"""AnyLearning.

The one thing that happens at import: pointing torch, huggingface_hub and
iopath at the pretrained weights shipped with the app.

It has to be here rather than in an entry point. Those libraries read their
cache locations once, when they are imported, and by the time any entry point
has imported a router it is already too late. Putting it in the package's own
__init__ means anything that touches `anylearning` at all -- the app, the
training subprocess, a test -- gets it before torch can be loaded.

An environment variable that is already set is never overwritten; see
anylearning/weights.py.
"""

from anylearning import weights as _weights

_weights.use_bundled()
