from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import botocore
import botocore.paginate


def start_client(profile: str) -> boto3.client:
    """Starts an S3 client with the given profile.

    Args:
        profile (str):
            The AWS profile to use.

    Returns:
        boto3.client:
            The S3 client.
    """
    session = boto3.Session(profile_name=profile)
    return session.client("s3")


def create_paginated_bucket_iterator(
    bucket_name: str,
    prefix: str,
    profile: str,
    max_keys: int,
    enable_recursive_search: bool = False,
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
            all files within the prefix.Enabling this option will slow down
            the search significantly; however, it permits
            searching for specific files within the prefix.

    Returns:
        botocore.paginate.PageIterator:
            The iterator for the contents of the bucket under the given prefix.

    """
    s3_client = start_client(profile)
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
            bucket_name, prefix, profile, max_keys
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
