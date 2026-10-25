"""Capture installed distribution versions without invoking pip or accessing indexes."""
from importlib.metadata import distributions


def dependency_versions() -> dict[str, str]:
    return dict(sorted(
        (dist.metadata["Name"], dist.version)
        for dist in distributions() if dist.metadata["Name"]
    ))
