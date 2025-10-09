import inspect
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Literal, Optional

import click
from pydantic import BaseModel
from tqdm import tqdm

from openfold3.core.config.config_utils import load_yaml


class OstRunnerSettings(BaseModel):
    mode: Literal["protein_ligand"]
    n_processes: int = 1
    chunksize: int = 1
    log_dir: Optional[Path] = None


class OstPLRunnerSettings(OstRunnerSettings):
    """Protein-ligand OST pipeline settings."""

    lddt_pli: bool = True
    rmsd: bool = True
    use_amc: bool = True

    # @model_validator(mode="after")
    # def _prepare_output_directories(self) -> "OSTRunnerSettings":
    #     pass
    #     return self


class OstRunnerInput(BaseModel):
    pred_paths: list[Path]
    ref_paths: list[Path]
    output_paths: list[Path]

    def zip(self):
        return zip(self.pred_paths, self.ref_paths, self.output_paths)


class OstRunner:
    """
    Class for running OST on Openfold3 outputs.
    """

    def __init__(self, settings: OstRunnerSettings) -> None:
        self.settings = settings

        # only include function parameters
        func_params = set(
            inspect.signature(
                OST_PIPELINE_REGISTRY[self.settings.mode]
            ).parameters.keys()
        )
        ost_kwargs = {
            k: v for k, v in self.settings.model_dump().items() if k in func_params
        }
        ost_pipeline = partial(OST_PIPELINE_REGISTRY[self.settings.mode], **ost_kwargs)

        self.ost_pipeline = ost_pipeline

    def __call__(self, ost_input: OstRunnerInput) -> tuple[list[str], list[str]]:
        successful = []
        failed = []

        # Set up worker initialization if logging is enabled
        initializer = None
        initargs = ()
        if self.settings.log_dir is not None:
            initializer = self.worker_init
            initargs = (self.settings.log_dir,)

        with mp.Pool(
            self.settings.n_processes, initializer=initializer, initargs=initargs
        ) as pool:
            for id, success in tqdm(
                pool.imap_unordered(
                    self.ost_pipeline,
                    ost_input.zip(),
                    chunksize=self.settings.chunksize,
                ),
                total=len(ost_input.pred_paths),
                desc="Running OST protein-ligand comparison",
            ):
                if success:
                    successful.append(id)
                else:
                    failed.append(id)

        return successful, failed

    @staticmethod
    def worker_init(
        log_directory: Path, is_main: bool = False
    ) -> Optional[logging.Logger]:
        """Initialize logging for worker processes."""
        worker_logger = logging.getLogger(f"ost_logger_{os.getpid()}")
        worker_logger.setLevel(logging.INFO)
        if is_main:
            hname = f"main_{os.getpid()}.log"
        else:
            hname = f"{os.getpid()}.log"
        handler = logging.FileHandler(log_directory / hname)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        if not worker_logger.hasHandlers():
            worker_logger.addHandler(handler)
        worker_logger.propagate = False
        return worker_logger if is_main else None

    @staticmethod
    def prepare_input_paths(
        pred_dir: Path,
        ref_dir: Path,
        output_dir: Path,
        main_logger: Optional[logging.Logger] = None,
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Prepare input paths for OST processing."""
        pred_paths = []
        ref_paths = []
        output_paths = []
        for query_dir in sorted(pred_dir.iterdir()):
            if not query_dir.is_dir():  # Skip files
                continue
            query = query_dir.name
            for seed_dir in sorted(query_dir.iterdir()):
                pred_paths_i = sorted(seed_dir.glob("*.cif"))
                for pred_path in pred_paths_i:
                    try:
                        seed, sample = itemgetter(-4, -2)(pred_path.stem.split("_"))
                    except IndexError:
                        msg = (
                            f"WARNING: Cannot parse seed and sample from "
                            f"filename: {pred_path.stem}"
                        )
                        main_logger.warning(msg) if main_logger else print(msg)
                        continue
                    ref_path = ref_dir / f"{query}.cif"
                    if not ref_path.exists():
                        msg = f"WARNING: reference path does not exist: {ref_path}"
                        main_logger.warning(msg) if main_logger else print(msg)
                        continue
                    pred_paths.append(pred_path)
                    ref_paths.append(ref_dir / f"{query}.cif")
                    output_file = f"{query}_seed_{seed}_sample_{sample}.json"
                    output_paths.append(output_dir / output_file)

        return pred_paths, ref_paths, output_paths


# primitives - TODO move to primitives


def ost_compare_protein_ligand_complex(
    input_paths: list[Path, Path, Path],
    lddt_pli: bool = True,
    rmsd: bool = True,
    use_amc: bool = True,
) -> tuple[str, bool]:
    # Unpack input paths
    pred_path, ref_path, output_path = input_paths

    logger = logging.getLogger(f"ost_logger_{os.getpid()}")

    def log_message(message: str) -> None:
        """Log message using worker logger if available, otherwise print."""
        if logger.hasHandlers():
            logger.info(message)
        else:
            print(message)

    # Construct OST command
    cmd = [
        "ost",
        "compare-ligand-structures",
        "-m",
        str(pred_path),
        "-r",
        str(ref_path),
        "-mf",
        "cif",
        "-o",
        str(output_path),
    ]
    if lddt_pli:
        cmd.append("--lddt-pli")
        if use_amc:
            cmd.append("--lddt-pli-amc")
    if rmsd:
        cmd.append("--rmsd")

    # Run
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        log_message(f"SUCCESS: {output_path}")
        success = True
    except FileNotFoundError as _:
        log_message(
            "ERROR: 'ost' command not found. "
            "Please install OpenStructure (OST) to use this function."
        )
        success = False
    except subprocess.CalledProcessError as e:
        log_message(f"FAILED: {cmd}\nError: {e.stderr}")
        success = False

    return pred_path.stem, success


OST_SETTINGS_REGISTRY = {
    "protein_ligand": OstPLRunnerSettings,
}
OST_PIPELINE_REGISTRY = {
    "protein_ligand": ost_compare_protein_ligand_complex,
}


@click.command()
@click.option(
    "--pred-dir",
    required=True,
    help=("Path to a dir of OF3 outputs."),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--ref-dir",
    required=True,
    help=(
        "Path to a flat dir of reference cif files."
        " Filenames should match query names in pred-dir."
    ),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--output-dir",
    required=True,
    help=("Dir where the OST output jsons are saved."),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--runner-yml",
    required=True,
    help=("Runner.yml file to be parsed into settings for the OST runner."),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--log-dir",
    required=False,
    default=None,
    help=(
        "Dir where the OST logs are saved."
        " If not set, messages are printed to the terminal."
    ),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
def main(
    pred_dir: Path,
    ref_dir: Path,
    output_dir: Path,
    runner_yml: Path,
    log_dir: Optional[Path],
) -> None:
    """Run OST on Openfold3 outputs."""
    # Check if OST is available
    if shutil.which("ost") is None:
        raise RuntimeError(
            "ERROR: 'ost' command not found in PATH. "
            "Please install OpenStructure (OST) or load the module."
        )

    runner_args = load_yaml(runner_yml)
    if runner_args.get("log_dir"):
        raise ValueError(
            "log_dir should be set via the --log-dir command line argument, "
            "not in the runner_yml."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    main_logger = (
        OstRunner.worker_init(log_directory=log_dir, is_main=True) if log_dir else None
    )

    # Convert any Path objects to strings for JSON serialization
    serializable_args = {
        k: str(v) if isinstance(v, Path) else v for k, v in runner_args.items()
    }
    args_msg = f"Loaded runner args:\n{json.dumps(serializable_args, indent=4)}"
    main_logger.info(args_msg) if main_logger else print(args_msg)

    pred_paths, ref_paths, output_paths = OstRunner.prepare_input_paths(
        pred_dir=pred_dir,
        ref_dir=ref_dir,
        output_dir=output_dir,
        main_logger=main_logger,
    )

    # Set up runner and run OST
    ost_runner_settings = OST_SETTINGS_REGISTRY[runner_args["mode"]](
        log_dir=log_dir,
        **runner_args,
    )
    ost_runner = OstRunner(settings=ost_runner_settings)

    if not pred_paths:
        msg = "WARNING: No valid prediction files found to process"
        main_logger.warning(msg) if main_logger else print(msg)
        return

    ost_input = OstRunnerInput(
        pred_paths=pred_paths,
        ref_paths=ref_paths,
        output_paths=output_paths,
    )
    successful, failed = ost_runner(ost_input=ost_input)

    msg = (
        f"Done. Successful: {len(successful)}, Failed: {len(failed)}\n"
        "Successful entries:\n" + "\n".join(successful) + "\n"
        "Failed entries:\n" + "\n".join(failed) + "\n"
    )
    main_logger.info(msg) if main_logger else print(msg)


if __name__ == "__main__":
    main()
