"""Check that our minimum HA version isn't too far behind the latest stable release.

Fails if the gap exceeds MAX_GAP_MONTHS, signaling it's time to bump.
"""

import json
import sys
import urllib.request

MAX_GAP_MONTHS = 4
MANIFEST_PATH = "custom_components/solar_lens/manifest.json"


def get_latest_ha_version() -> str:
    """Fetch the latest stable Home Assistant version from PyPI."""
    url = "https://pypi.org/pypi/homeassistant/json"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())

    stable_versions = [
        v for v in data["releases"] if all(tag not in v for tag in ("a", "b", "dev", "rc"))
    ]
    stable_versions.sort(key=lambda v: [int(x) for x in v.split(".")])
    return stable_versions[-1]


def get_our_minimum_version() -> str:
    """Read the minimum HA version from our manifest.json."""
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    return manifest["homeassistant"]


def version_to_months(version: str) -> int:
    """Convert a YYYY.M.x version to total months for comparison."""
    parts = version.split(".")
    return int(parts[0]) * 12 + int(parts[1])


def main() -> None:
    latest = get_latest_ha_version()
    minimum = get_our_minimum_version()
    gap = version_to_months(latest) - version_to_months(minimum)

    print(f"Latest HA stable: {latest}")
    print(f"Our minimum:      {minimum}")
    print(f"Gap:              {gap} months (max allowed: {MAX_GAP_MONTHS})")

    if gap > MAX_GAP_MONTHS:
        print("\n❌ Minimum version is too old. Time to bump.")
        sys.exit(1)

    print("\n✅ Version is fresh.")


if __name__ == "__main__":
    main()
