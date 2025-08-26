from .central import (
    central,
)

from .partial import (
    partial_general_statistics,
    partial_aggregate_adjusted_deviation,
)

from .miscellaneous import (
    check_input_structure,
)

__all__ = [
    "central",
    "partial_general_statistics",
    "partial_aggregate_adjusted_deviation",
    "check_input_structure",
]

# Package information
__version__ = "1.0.0"
__author__ = "STRONGAYA"
__description__ = (
    "Vantage6 algorithm for descriptive statistics with comprehensive testing"
)
