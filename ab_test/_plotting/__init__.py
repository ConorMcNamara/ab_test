"""Draw backend-neutral plot specs with Plotly or Altair.

Plotly is a required dependency and the default backend. Altair is optional
(``pip install abtest-analysis[altair]``) and imported only when asked for.
"""

import importlib.util
from typing import TYPE_CHECKING, Any, TypeAlias

from ab_test._plotting.spec import (
    AxisFormat,
    Band,
    ForestPanel,
    ForestPoint,
    ForestSpec,
    HLine,
    LineSeries,
    LineSpec,
    Segment,
    VLine,
)

if TYPE_CHECKING:
    import altair as alt
    import plotly.graph_objects as go

__all__ = [
    "BACKENDS",
    "Figure",
    "validate_backend",
    "render",
    "AxisFormat",
    "Band",
    "ForestPanel",
    "ForestPoint",
    "ForestSpec",
    "HLine",
    "LineSeries",
    "LineSpec",
    "Segment",
    "VLine",
]

BACKENDS = ("plotly", "altair")

Figure: TypeAlias = "go.Figure | alt.TopLevelMixin"


def validate_backend(backend: str) -> None:
    """Check that ``backend`` names a known plotting backend that is installed.

    Called before any plot computes its numbers, so a bad backend fails fast.

    Raises
    ------
    ValueError
        If ``backend`` is not one of :data:`BACKENDS`.
    ImportError
        If ``backend`` is ``"altair"`` but Altair is not installed.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "altair" and importlib.util.find_spec("altair") is None:
        raise ImportError("backend='altair' needs Altair: pip install 'abtest-analysis[altair]'")


def render(spec: ForestSpec | LineSpec, backend: str = "plotly") -> Figure:
    """Draw ``spec`` with ``backend``, returning a Plotly figure or an Altair chart."""
    validate_backend(backend)
    module: Any
    if backend == "altair":
        from ab_test._plotting import _altair as module
    else:
        from ab_test._plotting import _plotly as module
    if isinstance(spec, ForestSpec):
        return module.forest(spec)  # type: ignore[no-any-return]
    return module.line(spec)  # type: ignore[no-any-return]
