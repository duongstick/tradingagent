"""Alpha research: operators, neutralization, IC, multiple-testing, research loop."""

from .ic import ICResult, evaluate, forward_returns
from .multiple_testing import benjamini_hochberg, deflated_sharpe_ratio
from .neutralizer import neutralize
from .research import FactorReport, research

__all__ = [
    "ICResult",
    "evaluate",
    "forward_returns",
    "benjamini_hochberg",
    "deflated_sharpe_ratio",
    "neutralize",
    "FactorReport",
    "research",
]
