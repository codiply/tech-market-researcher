"""MinIO (S3) artifact bucket provisioning."""

import logging
import os

import boto3

from bootstrapping.common.configuration import BootstrapConfig
from bootstrapping.common.exceptions import BootstrapError

logger = logging.getLogger(__name__)


def _create_buckets(config: BootstrapConfig) -> None:
    """Create the configured buckets if they do not exist (idempotent)."""
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        names = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    except s3.exceptions.ClientError as e:
        raise BootstrapError(f"Cannot reach S3 at {endpoint}: {e}") from e
    for bucket in config.s3.buckets_to_create:
        if bucket in names:
            logger.info("Bucket '%s' already exists.", bucket)
            continue
        s3.create_bucket(Bucket=bucket)
        logger.info("Created bucket '%s'.", bucket)


def bootstrap_s3(config: BootstrapConfig) -> None:
    """Provision the S3 (MinIO) resources."""
    _create_buckets(config)
