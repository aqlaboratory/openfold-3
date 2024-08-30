# %%
from pathlib import Path


def _import_all_py_files_from_dir(directory: Path):
    """Imports all Python files in the specified directory."""
    for path in directory.glob("*.py"):  # Finds all `.py` files
        __import__(".".join(list(path.parts[-6:-1]) + [path.parts[-1].split(".")[0]]))
    return


_import_all_py_files_from_dir(
    Path(__file__).parent.parent / Path("framework/single_datasets")
)

# %%
