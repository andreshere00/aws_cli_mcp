import datetime as dt
from typing import Any, Dict, List, Optional

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

# Module under test
import src.infrastructure.aws.s3_uow as s3_uow_mod
from src.domain.schemas import Bucket, ListS3BucketResponse
from src.infrastructure.aws.s3_uow import S3UnitOfWork

# -----------------------
# Test doubles / stubs
# -----------------------


class StubBoto3S3Client:
    """Programmable stub for boto3 S3 client."""

    def __init__(
        self,
        *,
        buckets: Optional[List[Dict[str, Any]]] = None,
        owner: Optional[Dict[str, Any]] = None,
    ):
        self._buckets = buckets if buckets is not None else []
        self._owner = owner if owner is not None else {"DisplayName": "owner", "ID": "owner-id"}
        self.calls: List[Dict[str, Any]] = []
        # behavior toggles
        self.raise_on_init: Optional[BaseException] = None
        self.raise_on_list: Optional[BaseException] = None
        self.locations: Dict[str, Optional[str]] = {}  # bucket -> LocationConstraint
        self.raise_on_location: Dict[str, BaseException] = {}  # bucket -> exception

    # boto3.client(slug) factory is monkeypatched to return an instance of this stub,
    # so __init__ is already executed at that time. We'll simulate init failures via the factory.

    def list_buckets(self):
        self.calls.append({"op": "list_buckets"})
        if self.raise_on_list:
            raise self.raise_on_list
        return {"Buckets": self._buckets, "Owner": self._owner}

    def get_bucket_location(self, Bucket: str):
        self.calls.append({"op": "get_bucket_location", "bucket": Bucket})
        if Bucket in self.raise_on_location:
            raise self.raise_on_location[Bucket]
        # Return shape with LocationConstraint key; can be None/''/'US'/'EU'/region
        return {"LocationConstraint": self.locations.get(Bucket)}


class StubBoto3Module:
    """Replaces boto3 module to control client creation and simulate init errors."""

    def __init__(
        self, stub_client: StubBoto3S3Client, *, raise_on_client: Optional[BaseException] = None
    ):
        self._client = stub_client
        self._raise_on_client = raise_on_client

    def client(self, service_name: str, **kwargs):
        assert service_name == "s3"
        if self._raise_on_client:
            raise self._raise_on_client
        return self._client


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def stub_client():
    return StubBoto3S3Client()


@pytest.fixture
def patch_boto3(monkeypatch, stub_client):
    """Patch boto3 in the module under test to use our stub."""
    monkeypatch.setattr(s3_uow_mod, "boto3", StubBoto3Module(stub_client))
    return stub_client


# -----------------------
# __init__
# -----------------------


def test_init_success_uses_boto3_client(patch_boto3):
    uow = S3UnitOfWork()
    assert hasattr(uow, "client")
    # The stub is used; no exception raised


def test_init_raises_runtimeerror_on_boto3_failure(monkeypatch, stub_client):
    # Make the factory raise NoCredentialsError (covered error types)
    monkeypatch.setattr(
        s3_uow_mod, "boto3", StubBoto3Module(stub_client, raise_on_client=NoCredentialsError())
    )
    with pytest.raises(RuntimeError) as ex:
        S3UnitOfWork()
    assert "Failed to initialize S3 client" in str(ex.value)


# -----------------------
# list_buckets happy path
# -----------------------


def test_list_buckets_returns_pydantic_response(patch_boto3):
    # Prepare buckets
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    patch_boto3._buckets[:] = [
        {"Name": "alpha", "CreationDate": now},
        {"Name": "beta", "CreationDate": now},
    ]
    # Regions for buckets
    patch_boto3.locations = {"alpha": "eu-west-1", "beta": "us-west-2"}

    uow = S3UnitOfWork()
    resp = uow.list_buckets()

    assert isinstance(resp, ListS3BucketResponse)
    assert len(resp.Buckets) == 2
    names = [b.Name for b in resp.Buckets]
    assert names == ["alpha", "beta"]
    # ARN & region
    alpha = resp.Buckets[0]
    assert isinstance(alpha, Bucket)
    assert alpha.BucketArn.endswith(":::alpha")
    assert alpha.BucketRegion == "eu-west-1"
    assert resp.Owner.DisplayName == "owner"
    assert resp.Prefix is None


def test_list_buckets_filters_by_prefix(patch_boto3):
    now = dt.datetime.now(dt.timezone.utc)
    patch_boto3._buckets[:] = [
        {"Name": "prod-logs", "CreationDate": now},
        {"Name": "dev-logs", "CreationDate": now},
        {"Name": "prod-data", "CreationDate": now},
    ]
    patch_boto3.locations = {
        "prod-logs": "eu-west-1",
        "dev-logs": "eu-west-1",
        "prod-data": "eu-west-1",
    }

    uow = S3UnitOfWork()
    resp = uow.list_buckets(prefix="prod-")
    assert [b.Name for b in resp.Buckets] == ["prod-logs", "prod-data"]
    assert resp.Prefix == "prod-"


def test_list_buckets_filters_by_region(patch_boto3):
    now = dt.datetime.now(dt.timezone.utc)
    patch_boto3._buckets[:] = [
        {"Name": "a", "CreationDate": now},
        {"Name": "b", "CreationDate": now},
        {"Name": "c", "CreationDate": now},
    ]
    patch_boto3.locations = {"a": "us-east-1", "b": "eu-west-1", "c": "eu-west-1"}

    uow = S3UnitOfWork()
    resp = uow.list_buckets(region="eu-west-1")
    assert [b.Name for b in resp.Buckets] == ["b", "c"]


def test_list_buckets_respects_max_buckets_limit(patch_boto3):
    now = dt.datetime.now(dt.timezone.utc)
    patch_boto3._buckets[:] = [{"Name": f"b{i}", "CreationDate": now} for i in range(10)]
    # All in same region
    patch_boto3.locations = {f"b{i}": "eu-west-1" for i in range(10)}

    uow = S3UnitOfWork()
    resp = uow.list_buckets(max_buckets=3)
    assert len(resp.Buckets) == 3
    assert [b.Name for b in resp.Buckets] == ["b0", "b1", "b2"]


# -----------------------
# Region normalization & fallbacks
# -----------------------


def test_region_normalization_special_cases(patch_boto3):
    now = dt.datetime.now(dt.timezone.utc)
    patch_boto3._buckets[:] = [
        {"Name": "n1", "CreationDate": now},
        {"Name": "n2", "CreationDate": now},
        {"Name": "n3", "CreationDate": now},
        {"Name": "n4", "CreationDate": now},
    ]
    # Simulate LocationConstraint returning None / "" / "US" / "EU"
    patch_boto3.locations = {"n1": None, "n2": "", "n3": "US", "n4": "EU"}

    uow = S3UnitOfWork()
    resp = uow.list_buckets()

    regions = [b.BucketRegion for b in resp.Buckets]
    assert regions == ["us-east-1", "us-east-1", "us-east-1", "eu-west-1"]


def test_get_bucket_location_error_falls_back_to_us_east_1(patch_boto3):
    now = dt.datetime.now(dt.timezone.utc)
    patch_boto3._buckets[:] = [{"Name": "x", "CreationDate": now}]
    # Force location API error
    patch_boto3.raise_on_location = {
        "x": ClientError({"Error": {"Code": "Oops"}}, "GetBucketLocation")
    }

    uow = S3UnitOfWork()
    resp = uow.list_buckets()
    assert resp.Buckets[0].BucketRegion == "us-east-1"


# -----------------------
# Error handling
# -----------------------


def test_list_buckets_raises_runtimeerror_on_client_failure(monkeypatch, stub_client):
    # Patch module's boto3 to a stub that raises on list_buckets
    failing = StubBoto3S3Client()
    failing.raise_on_list = ClientError({"Error": {"Code": "Auth"}}, "ListBuckets")
    monkeypatch.setattr(s3_uow_mod, "boto3", StubBoto3Module(failing))

    uow = S3UnitOfWork()
    with pytest.raises(RuntimeError) as ex:
        uow.list_buckets()
    assert "Failed to list buckets" in str(ex.value)
