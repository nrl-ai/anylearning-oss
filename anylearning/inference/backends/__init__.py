"""Model-specific inference backends.

Backends are imported lazily through the registry so optional runtimes do not
become dependencies of the lightweight inference contract surface.
"""
