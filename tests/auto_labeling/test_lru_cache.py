import pytest

from anylearning.auto_labeling.lru_cache import LRUCache


def test_lru_cache_init():
    cache = LRUCache(maxsize=5)
    assert cache.maxsize == 5
    assert len(cache._cache) == 0


def test_lru_cache_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="at least 1"):
        LRUCache(maxsize=0)


def test_lru_cache_put_and_get():
    cache = LRUCache(maxsize=2)

    # Test putting and getting single item
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test getting non-existent key
    assert cache.get("nonexistent") is None

    # Test eviction of oldest item
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    assert cache.get("key1") is None  # Should be evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_lru_cache_find():
    cache = LRUCache()

    cache.put("key1", "value1")
    assert cache.find("key1") is True
    assert cache.find("nonexistent") is False
    assert "key1" in cache
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0


def test_lru_cache_ordering():
    cache = LRUCache(maxsize=2)

    # Add two items
    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Access key1, making it most recently used
    cache.get("key1")

    # Add new item, should evict key2 instead of key1
    cache.put("key3", "value3")

    assert cache.find("key1") is True
    assert cache.find("key2") is False
    assert cache.find("key3") is True


def test_lru_cache_thread_safety():
    import threading
    import time

    cache = LRUCache(maxsize=100)
    errors = []

    def worker():
        try:
            for i in range(100):
                cache.put(f"key{i}", f"value{i}")
                time.sleep(0.001)  # Force thread switching
                assert cache.get(f"key{i}") == f"value{i}"
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety test failed with errors: {errors}"
