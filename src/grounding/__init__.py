from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("grounding")
except PackageNotFoundError:
    __version__ = "dev"
