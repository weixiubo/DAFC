"""Delay platform imports until an adapter is actually used."""
from importlib import import_module
class LazyModule:
    def __init__(self, name):
        self.name = name
    def __getattr__(self, name):
        try:
            module = import_module(self.name)
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                f"Optional adapter dependency {error.name!r} is missing. "
                "See docs/RESOURCES.md and install it in this project's environment."
            ) from error
        return getattr(module, name)
