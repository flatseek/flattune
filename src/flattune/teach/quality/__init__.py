"""Quality evaluation modules for generated samples."""

from flattune.teach.quality.deduplication import Deduplicator
from flattune.teach.quality.scoring import QualityScorer
from flattune.teach.quality.hallucination import HallucinationDetector

__all__ = [
    "Deduplicator",
    "QualityScorer",
    "HallucinationDetector",
]
