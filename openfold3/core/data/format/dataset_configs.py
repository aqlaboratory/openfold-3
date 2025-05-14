from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import BeforeValidator, DirectoryPath, FilePath


def is_path_none(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if isinstance(value, Path):
        return value
    elif value is None or value.lower() in ["none", "null"]:
        return None
    else:
        return Path(value)


FilePathOrNone = Annotated[Optional[FilePath], BeforeValidator(is_path_none)]
DirectoryPathOrNone = Annotated[Optional[DirectoryPath], BeforeValidator(is_path_none)]
