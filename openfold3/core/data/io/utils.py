import boto3
import numpy as np


def download_file_from_s3(
    bucket: str,
    prefix: str,
    filename: str,
    outfile: str,
    profile: str | None = None,
    session: boto3.Session | None = None,
):
    """Download a file from an s3 bucket, using the provided profile or session.
    Args:
        bucket (str): the name of the s3 bucket. Must not have s3:// prefix
        prefix (str): the path from the bucket root to the dir containing the file
        filename (str): name of the file to download
        outfile (str): file to save the downloaded file to
        profile (str | None, optional): profile to instantiate the boto3 session with
        session (boto3.Session | None, optional): instantiated boto3 session to use.
        one of profile or session must be provided

    """
    if session is None:
        if profile is None:
            raise ValueError("Either profile or session must be provided")
        session = boto3.Session(profile_name=profile)
    s3_client = session.client("s3")
    try:
        s3_client.download_file(bucket, f"{prefix}/{filename}", outfile)
    except Exception as e:
        print(f"Error downloading file from s3://{bucket}/{prefix}/{filename}")
        raise e
    return


def encode_numpy_types(obj: object):
    """An encoding function for NumPy -> standard types.

    This is useful for JSON serialisation for example, which can't deal with NumPy
    types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def is_intlike_string(value: str) -> bool:
    """Check if a string represents an integer.

    Args:
        value:
            The string to check.

    Returns:
        Whether the string represents an integer.
    """
    try:
        int(value)
        return True
    except ValueError:
        return False
