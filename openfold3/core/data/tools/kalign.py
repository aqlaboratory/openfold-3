import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=512)
def run_kalign(
    a3m_string: str,
) -> str:
    """Runs Kalign on the provided A3M string and returns the aligned sequences.

    Args:
        a3m_string (str):
            A3M formatted string containing the sequences to be aligned. In the template
            pipeline, the first sequence is the query, and the rest are templates
            sequences to be realigned to it from hmmsearch.

    Raises:
        RuntimeError:
            If Kalign is not available.
        subprocess.CalledProcessError:
            If the Kalign command fails.

    Returns:
        str:
            The aligned sequences in A3M format as a string.
    """
    kalign_available = shutil.which("kalign") is not None

    if not kalign_available:
        raise RuntimeError(
            "Kalign is not available. Please install it and ensure it is in your PATH."
        )

    try:
        result = subprocess.run(
            ["kalign"], input=a3m_string, capture_output=True, text=True, check=True
        )

        # The resulting MSA is stored in the variable
        alignment_result = result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Kalign command failed:\n{e.stderr}")

    return alignment_result
