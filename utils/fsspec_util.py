import os
from tosfsspec import TosFileSystem
from pydantic_settings import BaseSettings

class TosSettings(BaseSettings):
    volc_accesskey: str = ""
    volc_secretkey: str = ""
    volc_region: str = "cn-beijing"
    volc_endpoint: str = "tos-cn-beijing.ivolces.com"


def get_tosfs():
    """Initialize and return a TosFileSystem if credentials are available."""
    settings = TosSettings()
    if not settings.volc_accesskey or not settings.volc_secretkey:
        return None
    return TosFileSystem(
        key=settings.volc_accesskey,
        secret=settings.volc_secretkey,
        endpoint=settings.volc_endpoint,
        region=settings.volc_region,
        enable_crc=False,
    )

def open_with_fs(path, mode="rb", fs=None):
    """Open a file using the provided filesystem if path is on TOS."""
    if fs is not None and path.startswith("tos://"):
        return fs.open(path, mode)
    return open(path, mode)


def exists_with_fs(path, fs=None):
    """Check if a path exists, using the provided filesystem for TOS paths."""
    if fs is not None and path.startswith("tos://"):
        return fs.exists(path)
    return os.path.exists(path)


def getsize_with_fs(path, fs=None):
    """Get file size, using the provided filesystem for TOS paths."""
    if fs is not None and path.startswith("tos://"):
        return fs.size(path)
    return os.path.getsize(path)


def list_tos_directory(path, fs=None):
    """List directory contents.

    For tos:// paths, returns a detailed listing via TosFileSystem.ls(detail=True).
    For local paths, returns a simple listing via os.listdir().

    If no fs is provided for a tos:// path, the function attempts to create a
    TosFileSystem automatically.  If that fails (credentials not configured), a
    clear error is raised telling the user which environment variables to set.
    """
    if path.startswith("tos://"):
        if fs is None:
            fs = get_tosfs()
        if fs is None:
            raise ValueError(
                "TosFileSystem could not be initialized. "
                "Please set the VOLC_ACCESSKEY and VOLC_SECRETKEY environment variables."
            )
        return fs.ls(path, detail=True)
    return os.listdir(path)


def tos_to_s3_config(tos_path: str) -> tuple[str, dict]:
    """Convert a tos:// path to an S3-compatible URI and storage options.

    Returns (s3_uri, storage_options) suitable for passing to
    lance.dataset() or lance.write_dataset().
    """
    settings = TosSettings()
    bucket = tos_path.replace("tos://", "").split("/", 1)[0]
    uri = tos_path.replace("tos://", "s3://")
    opts = {
        "access_key_id": settings.volc_accesskey,
        "secret_access_key": settings.volc_secretkey,
        "aws_endpoint": f"https://{bucket}.tos-s3-{settings.volc_region}.ivolces.com",
        "virtual_hosted_style_request": "true",
    }
    return uri, opts
