import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from src.domain.schemas import ListS3BucketResponse, Bucket, Owner

load_dotenv()

class S3UnitOfWork:
    """
    Manages an S3 client with optional custom configuration.

    This class builds a boto3 S3 client. If a parameter is left as None,
    boto3 will fall back to its default credential chain or environment
    variables.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the S3UnitOfWork with optional AWS client configuration.

        Args:
            **kwargs: Standard boto3.client keyword args (region_name, endpoint_url, etc.).

        Raises:
            RuntimeError: If the S3 client cannot be initialized.
        """
        params = {k: v for k, v in kwargs.items() if v is not None}
        try:
            self.client = boto3.client(service_name="s3", **params)
        except (BotoCoreError, NoCredentialsError, ClientError) as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}") from e

    def list_buckets(
        self,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
        max_buckets: int = 100,
    ) -> ListS3BucketResponse:
        """
        List S3 buckets, enrich with region + ARN, and return a Pydantic response.

        Args:
            prefix: If set, include only buckets whose names start with this value.
            region: If set, include only buckets whose resolved region equals this value.
            max_buckets: Maximum number of buckets to include after filtering.

        Returns:
            ListS3BucketResponse: DTO containing buckets and owner info.

        Raises:
            RuntimeError: If AWS calls fail.
        """
        try:
            raw = self.client.list_buckets()
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Failed to list buckets: {e}") from e

        raw_buckets: List[Dict[str, Any]] = raw.get("Buckets", []) or []
        owner_src: Dict[str, Any] = raw.get("Owner") or {}

        # Filter by prefix
        if prefix:
            raw_buckets = [b for b in raw_buckets if str(b.get("Name", "")).startswith(prefix)]

        buckets: List[Bucket] = []
        for b in raw_buckets:
            name = b["Name"]
            creation = b.get("CreationDate")

            try:
                loc = self.client.get_bucket_location(Bucket=name).get("LocationConstraint")
                resolved_region = Bucket.normalize_region(loc)
            except (BotoCoreError, ClientError):
                resolved_region = "us-east-1"

            if region and resolved_region != region:
                continue

            buckets.append(
                Bucket(
                    Name=name,
                    CreationDate=creation,
                    BucketRegion=resolved_region,
                    BucketArn=Bucket.arn_for(name),
                )
            )

            if len(buckets) >= max_buckets:
                break

        response = ListS3BucketResponse(
            Buckets=buckets,
            Owner=Owner(
                DisplayName=str(owner_src.get("DisplayName", "")),
                ID=str(owner_src.get("ID", "")),
            ),
            ContinuationToken=None,
            Prefix=prefix if prefix is not None else None,
        )
        return response
