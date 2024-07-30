from pathlib import Path

from openfold3.model_implementations import registry


def _import_specific_py_files_from_dir(pattern: str):
    """Imports files with given pattern."""
    for path in Path(__file__).parent.resolve().glob(pattern):
        # Selects path starting from `openfold3` - will vary for different directories
        package_module = ".".join(list(path.parts[-4:-1]) + [str(path.stem)])
        __import__(package_module)
    return


# Import all `runner.py` from model directories to register the models
_import_specific_py_files_from_dir("*/runner.py")
MODEL_REGISTRY = registry.MODEL_REGISTRY
