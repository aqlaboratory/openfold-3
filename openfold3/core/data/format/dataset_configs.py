import logging
from pathlib import Path
from typing import Annotated, Any, Optional, Union

from pydantic import BeforeValidator, DirectoryPath, FilePath

from openfold3.core.data.resources.residues import MoleculeType


def _convert_molecule_type(value: Any) -> Any:
    if isinstance(value, MoleculeType):
        return value
    elif isinstance(value, str):
        try:
            return MoleculeType[value.upper()]
        except KeyError:
            logging.warning(
                f"Found invalid {value=} for molecule type, skipping this example."
            )
            return None
    elif isinstance(value, int):
        try:
            return MoleculeType(value)
        except ValueError:
            logging.warning(
                f"Found invalid {value=} for molecule type, skipping this example."
            )
            return None


def is_path_none(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if isinstance(value, Path):
        return value
    elif value is None or value.lower() in ["none", "null"]:
        return None
    else:
        return Path(value)


FilePathOrNone = Annotated[Optional[FilePath], BeforeValidator(is_path_none)]
DirectoryPathOrNone = Annotated[Optional[DirectoryPath], BeforeValidator(is_path_none)]
