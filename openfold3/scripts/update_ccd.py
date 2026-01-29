#!/usr/bin/env python3
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

"""
Script to update the Chemical Component Dictionary (CCD).

Downloads the latest CCD from WWPDB, processes it to BinaryCIF format,
and uploads to S3.
"""

import argparse
import logging
from pathlib import Path

import biotite.setup_ccd
import boto3

from openfold3.core.utils.s3 import compute_local_crc64nvme_base64
from openfold3.setup_openfold import S3_BUCKET, S3_KEY

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    """Upload file to S3 with CRC64NVME checksum."""
    logger.info(f"Uploading {local_path} to s3://{bucket}/{key}...")

    checksum = compute_local_crc64nvme_base64(local_path)
    logger.info(f"Local file checksum (CRC64NVME): {checksum}")

    s3 = boto3.client("s3")
    with open(local_path, "rb") as f:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ChecksumAlgorithm="CRC64NVME",
            ChecksumCRC64NVME=checksum,
        )

    logger.info(f"Upload complete: s3://{bucket}/{key}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update CCD: download from WWPDB, process, and upload to S3"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=S3_BUCKET,
        help=f"S3 bucket name (default: {S3_BUCKET})",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=S3_KEY,
        help=f"S3 object key (default: {S3_KEY})",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip S3 upload (useful for local testing)",
    )
    args = parser.parse_args()

    # Download and process CCD using biotite's setup_ccd
    logger.info("Downloading and processing CCD from WWPDB...")
    biotite.setup_ccd.main()
    output_path = biotite.setup_ccd.OUTPUT_CCD
    logger.info(f"CCD processed and saved to: {output_path}")

    if not args.skip_upload:
        upload_to_s3(output_path, args.bucket, args.key)
    else:
        logger.info("Skipping S3 upload (--skip-upload flag set)")


if __name__ == "__main__":
    main()
