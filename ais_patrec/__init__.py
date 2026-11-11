"""ais_patrec - Artificial Immune System pattern recognition for network security.

Components
----------
* :class:`VDetector` - real-valued negative selection (Ji & Dasgupta 2004).
* :class:`NearestSelfDetector` - positive-selection baseline (Stibor et al. 2005).
* :class:`ClonalSelectionClassifier` - CLONALG-style memory classifier
  (de Castro & Von Zuben 2002) with a negative-selection operator
  (Kim & Bentley 2001) and incremental "secondary response" learning.
* :class:`ImmuneRecognizer` - two-layer memory + tolerance recognizer.
* :class:`ShapeSpace` - NSL-KDD preprocessing into a [0,1]^k shape-space.
"""
from .clonal_selection import ClonalSelectionClassifier
from .data import FAMILIES, ShapeSpace, load_nsl_kdd
from .immune_recognizer import NOVEL, ImmuneRecognizer
from .negative_selection import NearestSelfDetector, VDetector

__all__ = [
    "ClonalSelectionClassifier", "ImmuneRecognizer", "NearestSelfDetector",
    "VDetector", "ShapeSpace", "load_nsl_kdd", "FAMILIES", "NOVEL",
]
__version__ = "0.1.0"
