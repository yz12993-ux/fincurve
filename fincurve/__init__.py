"""fincurve: find which functional form (or distribution) best characterises financial data."""
__version__ = "0.2.0"

from .core import analyze, analyze_distribution
from .library import list_candidates

__all__ = ["analyze", "analyze_distribution", "list_candidates"]
