from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from artrec.monitoring.drift import check_drift


def test_drift_check_flags_large_numeric_shift():
    reference = pd.DataFrame({"price": [10, 11, 12, 13], "style": ["a", "a", "b", "b"]})
    current = pd.DataFrame(
        {"price": [100, 110, 120, 130], "style": ["c", "c", "c", "c"]}
    )

    report = check_drift(
        reference, current, numeric_features=["price"], categorical_features=["style"]
    )

    assert report["summary"]["has_alerts"] is True
    assert report["numeric"]["price"]["alert"] is True
    assert report["categorical"]["style"]["alert"] is True
