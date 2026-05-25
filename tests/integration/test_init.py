"""Basic tests for Solar Lens integration."""

import sys
from pathlib import Path

# Add custom_components to path so we can import solar_lens
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "custom_components"))


def test_domain_constant() -> None:
    """Verify that DOMAIN constant is correct."""
    from solar_lens.const import DOMAIN

    assert DOMAIN == "solar_lens"
