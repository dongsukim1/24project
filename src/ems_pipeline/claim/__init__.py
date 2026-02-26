"""Claim-building subpackage.

`ems_pipeline.claim` remains the stable import path for claim-building utilities.
"""

from __future__ import annotations

from ems_pipeline.models import Claim, EntitiesDocument


def build_claim(entities_doc: EntitiesDocument) -> Claim:
    """Build a structured proto-claim JSON from extracted entities.

    Args:
        entities_doc: Output from `ems_pipeline extract`.

    Returns:
        Claim: structured fields plus provenance links to segment IDs.
    """

    raise NotImplementedError(
        "Claim building is not implemented. Implement `ems_pipeline.claim.build_claim()`."
    )

