# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import botocore
import botocore.paginate
from botocore.config import Config


def parse_s3_config(s3_client_config_str: str) -> dict:
    """Converts a string representation of an S3 client config to a dictionary.

    Args:
        s3_client_config_str (str | None):
            The string representation of the S3 client config.

    Raises:
        ValueError:
            If the provided string is not a valid JSON.

    Returns:
        dict:
            The S3 client config as a dictionary.
    """
    try:
        s3_client_config = json.loads(s3_client_config_str)
        return s3_client_config
    except json.JSONDecodeError as e:
        raise ValueError("Invalid s3 client config provided.") from e


def start_s3_client(profile: str) -> boto3.client:
    """Starts an S3 client with the given profile.

    Args:
        profile (str):
            The AWS profile to use.

    Returns:
        boto3.client:
            The S3 client.
    """
    ### instantiate a boto3 session using adaptive retries
    ### this will automatically retry failed requests
    ### using an exponential backoff strategy, to avoid
    ### errors from rate limiting.
    session = boto3.Session(profile_name=profile)
    return session.client(
        "s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"})
    )


def create_paginated_bucket_iterator(
    bucket_name: str,
    prefix: str,
    profile: str,
    max_keys: int,
    enable_recursive_search: bool,
) -> botocore.paginate.PageIterator:
    """Creates an iterator for the contents of a bucket under a given prefix.

    Args:
        bucket_name (str):
            The name of the bucket.
        prefix (str):
            The prefix to list entries under.
        profile (str):
            The AWS profile to use.
        max_keys (int):
            The maximum number of keys to return.
        enable_recursive_search (bool):
            Whether to enable recursive search; recursively search for
            all files within the prefix. Enabling this option will slow down
            the search significantly; however, it permits
            searching for specific files within the prefix.

    Returns:
        botocore.paginate.PageIterator:
            The iterator for the contents of the bucket under the given prefix.

    """
    s3_client = start_s3_client(profile)
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {
        "Bucket": bucket_name,
        "Prefix": prefix,
        "Delimiter": "/",
        "MaxKeys": max_keys,
    }
    if enable_recursive_search:
        del operation_parameters["Delimiter"]

    return paginator.paginate(**operation_parameters)


def list_bucket_entries(
    bucket_name: str,
    prefix: str,
    profile: str,
    max_keys: int = 1000,
    check_filename_exists: str = None,
    num_workers: int = 1,
) -> list[str]:
    """Lists the paths of all files and subdirs in a bucket under a given prefix.

    Note entries are listed with maximum depth of 1.

    Args:
        bucket_name (str):
            The name of the bucket.
        prefix (str):
            The prefix to list entries under.
        profile (str):
            The AWS profile to use.
        max_keys (int):
            The maximum number of keys to return.
        check_filename_exists (str):
            File to search for among the listed entries. If provided, this function
            returns the list of parent directories that contain the specified file.
        num_workers (int):
            The number of workers to use for processing the paginated results.

    Returns:
        list[str]:
            A list of paths of all files and subdirs in the bucket under the given
            prefix.
    """
    if check_filename_exists:
        paginated_iterator = create_paginated_bucket_iterator(
            bucket_name, prefix, profile, max_keys, enable_recursive_search=True
        )
        valid_entries = []

        def process_page(page):
            entries = []
            if "Contents" in page:
                for obj in page["Contents"]:
                    if obj["Key"].endswith(check_filename_exists):
                        entries.append(Path(obj["Key"]).parent)
            return entries

        with ThreadPoolExecutor(num_workers) as executor:
            futures = [
                executor.submit(process_page, page) for page in paginated_iterator
            ]
            for future in futures:
                res = future.result()
                if res:
                    valid_entries.extend(res)

        return valid_entries
    else:
        paginated_iterator = create_paginated_bucket_iterator(
            bucket_name, prefix, profile, max_keys, enable_recursive_search=False
        )

        entries = []
        for page in paginated_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    entries.append(Path(obj["Key"]))

            if "CommonPrefixes" in page:
                for pfx in page["CommonPrefixes"]:
                    entries.append(Path(pfx["Prefix"]))

    return entries


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
        bucket (str):
            The name of the s3 bucket. Must not have s3:// prefix.
        prefix (str):
            The path from the bucket root to the dir containing the file.
        filename (str):
            Name of the file to download.
        outfile (str):
            File to save the downloaded file to.
        profile (str | None, optional):
            Profile to instantiate the boto3 session with
        session (boto3.Session | None, optional):
            Instantiated boto3 session to use.

    Raises:
        ValueError:
            If neither profile nor session is provided.
        Exception:
            If the download fails.

    """
    # TODO: rework with existing primitives
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


def open_local_or_s3(
    filepath: Path, profile: str | None, mode: str = "r"
) -> io.StringIO | io.BytesIO:
    """
    Return a file-like object for reading text/binary from either a local
    path or an S3 URI.

    Args:
        path (Path):
            Local file path or an S3 URI (e.g., "s3:/bucket/key"). Note the single
            forward slash after "s3:".
        profile (str, optional):
            Boto3 profile name (if needed).
        mode (str):
            "r" for text mode, "rb" for binary, etc.

    Returns:
        io.StringIO | io.BytesIO:
            A file-like object, which can be used just like an open() file.
    """
    # S3 path
    if str(filepath).startswith("s3:/"):
        # Parse out bucket and key
        s3_client = start_s3_client(profile)
        response = s3_client.get_object(
            Bucket=filepath.parts[1], Key="/".join(filepath.parts[2:])
        )
        body = response["Body"].read()  # this is bytes

        # If reading text, wrap bytes in a StringIO, else wrap in BytesIO
        if "b" in mode:
            return io.BytesIO(body)
        else:
            # decode bytes to string
            return io.StringIO(body.decode("utf-8"))
    # Normal local path
    else:
        return open(Path(filepath), mode)  # noqa: SIM115
