"""Quality evaluation modules for generated samples."""

from flattune.teach.quality.deduplication import Deduplicator
from flattune.teach.quality.hallucination import HallucinationDetector
from flattune.teach.quality.scoring import QualityScorer

__all__ = [
    "Deduplicator",
    "QualityScorer",
    "HallucinationDetector",
]
