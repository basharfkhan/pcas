"""Make the stated conflict probability mean what it says.

The multimodal model ranks conflicts well and states them badly: probabilities of 0.28, 0.49,
0.69 and 0.94 were followed by conflicts 7%, 15%, 27% and 70% of the time. Overconfident by 2
to 4x. Thresholding still works, because the ordering is right, but a number that reads as 30%
and means 7% cannot be shown to a pilot.

The fix is post-hoc and cheap: learn a monotone map from stated probability to observed
frequency on data the model did not train on, then apply it. Isotonic regression is the right
shape of tool because it assumes only monotonicity, which is exactly the property the
reliability table says the model has. It is fitted here with pool-adjacent-violators, in about
thirty lines, rather than by adding a dependency.

Two consequences worth being clear about:

- Calibration does not improve detection. An isotonic map is monotone, so the detection versus
  false alarm curve is unchanged: every operating point still exists, it is simply now labelled
  with a probability that means something. What improves is honesty, not skill.
- It must be fitted on held-out data. Fitting on the test days would make the reliability table
  a self-portrait.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _pool_adjacent_violators(y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Least-squares monotone fit to `y`, the classic PAVA loop.

    Walks left to right merging any block whose mean falls below its predecessor, which is the
    only way a monotone function can be violated once the inputs are sorted.
    """
    values = list(y.astype(float))
    counts = list(weights.astype(float))

    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        # Merge the violating pair into one block carrying their weighted mean.
        total = counts[i] + counts[i + 1]
        merged = (values[i] * counts[i] + values[i + 1] * counts[i + 1]) / total
        values[i : i + 2] = [merged]
        counts[i : i + 2] = [total]
        # The merge can break monotonicity behind us, so step back.
        i = max(i - 1, 0)

    out = np.empty(int(sum(counts)))
    position = 0
    for value, count in zip(values, counts, strict=True):
        out[position : position + int(count)] = value
        position += int(count)
    return out


@dataclass
class ProbabilityCalibrator:
    """A fitted monotone map from stated probability to calibrated probability."""

    knots: np.ndarray  # stated probabilities, ascending
    values: np.ndarray  # calibrated probabilities at those knots
    n_samples: int = 0

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the map, interpolating between knots and clamping outside them."""
        p = np.asarray(probabilities, dtype=float)
        if len(self.knots) == 0:
            return p
        return np.interp(p, self.knots, self.values, left=self.values[0], right=self.values[-1])

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "knots": self.knots.tolist(),
                    "values": self.values.tolist(),
                    "n_samples": self.n_samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ProbabilityCalibrator:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            knots=np.asarray(blob["knots"], dtype=float),
            values=np.asarray(blob["values"], dtype=float),
            n_samples=int(blob.get("n_samples", 0)),
        )


def fit_isotonic(stated: np.ndarray, outcome: np.ndarray) -> ProbabilityCalibrator:
    """Fit a calibration map from stated probabilities to observed outcomes.

    `outcome` is 1 where a conflict followed and 0 where it did not.
    """
    stated = np.asarray(stated, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if stated.size == 0:
        return ProbabilityCalibrator(knots=np.array([]), values=np.array([]))

    order = np.argsort(stated, kind="stable")
    fitted = _pool_adjacent_violators(outcome[order], np.ones(len(order)))

    # Collapse to one knot per distinct stated probability; that is all `predict` needs, and it
    # keeps the saved map small.
    unique, index = np.unique(stated[order], return_index=True)
    return ProbabilityCalibrator(knots=unique, values=fitted[index], n_samples=len(stated))


def reliability(
    stated: np.ndarray, outcome: np.ndarray, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
) -> pd.DataFrame:
    """Stated probability against observed frequency, per bin."""
    frame = pd.DataFrame({"stated": np.asarray(stated, dtype=float), "outcome": outcome})
    if frame.empty:
        return pd.DataFrame(columns=["bin", "n", "stated", "observed"])

    frame["bin"] = pd.cut(frame["stated"], bins=list(bins), include_lowest=True)
    grouped = frame.groupby("bin", observed=True).agg(
        n=("outcome", "size"), stated=("stated", "mean"), observed=("outcome", "mean")
    )
    return grouped.reset_index()


def expected_calibration_error(
    stated: np.ndarray, outcome: np.ndarray, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
) -> float:
    """Weighted mean gap between stated and observed probability, the usual single number."""
    table = reliability(stated, outcome, bins)
    if table.empty:
        return float("nan")
    weight = table["n"] / table["n"].sum()
    return float((weight * (table["stated"] - table["observed"]).abs()).sum())
