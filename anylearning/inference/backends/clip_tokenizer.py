"""Bounded NumPy tokenizer for CLIP-compatible ONNX text encoders.

The byte-pair encoding algorithm follows OpenAI CLIP's MIT-licensed tokenizer.
Its copyright and license are shipped in ``anylearning/inference/assets``.
"""

from __future__ import annotations

import gzip
import html
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import ftfy
import numpy as np
import regex

_MAX_BPE_ARCHIVE_BYTES = 2 * 1024**2
_MAX_BPE_TEXT_BYTES = 8 * 1024**2
_MAX_TEXT_BYTES = 4096
_MERGE_COUNT = 49_152 - 256 - 2


@lru_cache(maxsize=1)
def _byte_alphabet() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values.extend(range(ord("¡"), ord("¬") + 1))
    byte_values.extend(range(ord("®"), ord("ÿ") + 1))
    unicode_values = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, unicode_values), strict=True))


def _adjacent_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class ClipTokenizer:
    """Tokenize bounded text without importing a training framework."""

    def __init__(
        self,
        bpe_path: str | Path | None = None,
        *,
        max_text_bytes: int = _MAX_TEXT_BYTES,
    ) -> None:
        if not 1 <= max_text_bytes <= 65_536:
            raise ValueError("max_text_bytes must be between 1 and 65536")
        path = (
            Path(bpe_path)
            if bpe_path is not None
            else Path(__file__).parents[1] / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )
        archive_size = path.stat().st_size
        if archive_size <= 0 or archive_size > _MAX_BPE_ARCHIVE_BYTES:
            raise ValueError("CLIP BPE vocabulary archive has an invalid size")
        with gzip.open(path, "rb") as stream:
            encoded_merges = stream.read(_MAX_BPE_TEXT_BYTES + 1)
        if len(encoded_merges) > _MAX_BPE_TEXT_BYTES:
            raise ValueError("CLIP BPE vocabulary expands beyond its configured limit")
        lines = encoded_merges.decode("utf-8").splitlines()
        merges = [tuple(line.split()) for line in lines[1 : _MERGE_COUNT + 1]]
        if len(merges) != _MERGE_COUNT or any(len(pair) != 2 for pair in merges):
            raise ValueError("CLIP BPE vocabulary has an unexpected merge contract")

        alphabet = _byte_alphabet()
        vocabulary = list(alphabet.values())
        vocabulary.extend(token + "</w>" for token in alphabet.values())
        vocabulary.extend("".join(pair) for pair in merges)
        vocabulary.extend(("<|startoftext|>", "<|endoftext|>"))
        self._byte_encoder = alphabet
        self._encoder = {token: index for index, token in enumerate(vocabulary)}
        self._merge_ranks = {pair: index for index, pair in enumerate(merges)}
        self._max_text_bytes = max_text_bytes
        self._piece_cache: OrderedDict[str, str] = OrderedDict()
        self._pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )

    def _encode_piece(self, token: str) -> str:
        cached = self._piece_cache.pop(token, None)
        if cached is not None:
            self._piece_cache[token] = cached
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _adjacent_pairs(word)
        while pairs:
            selected = min(
                pairs,
                key=lambda pair: self._merge_ranks.get(pair, len(self._merge_ranks)),
            )
            if selected not in self._merge_ranks:
                break
            first, second = selected
            merged: list[str] = []
            position = 0
            while position < len(word):
                try:
                    next_position = word.index(first, position)
                except ValueError:
                    merged.extend(word[position:])
                    break
                merged.extend(word[position:next_position])
                position = next_position
                if position + 1 < len(word) and word[position + 1] == second:
                    merged.append(first + second)
                    position += 2
                else:
                    merged.append(word[position])
                    position += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _adjacent_pairs(word)
        encoded = " ".join(word)
        self._piece_cache[token] = encoded
        while len(self._piece_cache) > 8192:
            self._piece_cache.popitem(last=False)
        return encoded

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("CLIP text must be a string")
        if not text.strip():
            raise ValueError("CLIP text must contain non-whitespace characters")
        if len(text.encode("utf-8")) > self._max_text_bytes:
            raise ValueError("Text prompt exceeds the configured UTF-8 byte limit")
        normalized = (
            regex.sub(
                r"\s+",
                " ",
                html.unescape(html.unescape(ftfy.fix_text(text))),
            )
            .strip()
            .lower()
        )
        if len(normalized.encode("utf-8")) > self._max_text_bytes:
            raise ValueError("Normalized text exceeds the configured UTF-8 byte limit")
        result: list[int] = []
        for token in regex.findall(self._pattern, normalized):
            encoded = "".join(
                self._byte_encoder[value] for value in token.encode("utf-8")
            )
            result.extend(
                self._encoder[piece] for piece in self._encode_piece(encoded).split(" ")
            )
        return result

    def tokenize(self, text: str, *, context_length: int) -> np.ndarray:
        if not 2 <= context_length <= 256:
            raise ValueError("context_length must be between 2 and 256")
        start = self._encoder["<|startoftext|>"]
        end = self._encoder["<|endoftext|>"]
        tokens = [start, *self.encode(text), end]
        if len(tokens) > context_length:
            raise ValueError(
                f"Text prompt needs {len(tokens)} tokens; model capacity is "
                f"{context_length}"
            )
        result = np.zeros((1, context_length), dtype=np.int64)
        result[0, : len(tokens)] = tokens
        return result


__all__ = ["ClipTokenizer"]
