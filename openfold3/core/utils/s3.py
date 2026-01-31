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

"""S3 utilities for checksum comparison without downloading."""

import base64
from pathlib import Path

import boto3
from awscrt import checksums
from botocore import UNSIGNED
from botocore.config import Config


def get_s3_checksum(bucket: str, key: str) -> str | None:
    """Get CRC64NVME checksum from S3 object metadata (HEAD request, no download)."""
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    response = s3.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")

    if "ChecksumCRC64NVME" in response:
        return response["ChecksumCRC64NVME"]
    return None


def compute_local_crc64nvme_base64(filepath: Path) -> str:
    """Compute CRC64NVME of local file, return as base64 (S3 format)."""
    crc = 0
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            crc = checksums.crc64nvme(chunk, crc)
    # Convert to bytes (big-endian) and base64 encode
    crc_bytes = crc.to_bytes(8, byteorder="big")
    return base64.b64encode(crc_bytes).decode()


def s3_file_matches_local(local_path: Path, bucket: str, key: str) -> bool:
    """
    Compare local file with S3 object using CRC64NVME checksum.

    Returns True if files match, False if they differ or comparison fails.
    """
    if not local_path.exists():
        return False

    s3_checksum = get_s3_checksum(bucket, key)
    if s3_checksum is None:
        # S3 object has no checksum, cannot compare
        return False

    local_checksum = compute_local_crc64nvme_base64(local_path)
    return local_checksum == s3_checksum
