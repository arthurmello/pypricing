import numpy as np
import pandas as pd

from pypricing import add_event_columns, event_indicators


def test_weekly_black_friday_and_christmas():
    weeks = pd.date_range("2024-11-18", periods=6, freq="W-MON")
    flags = event_indicators(weeks)
    # Black Friday 2024-11-29 falls in the week starting 2024-11-25.
    np.testing.assert_array_equal(flags["black_friday"], [0, 1, 0, 0, 0, 0])
    # Cyber Monday 2024-12-02 is that week's Monday.
    np.testing.assert_array_equal(flags["cyber_monday"], [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(flags["christmas"], [0, 0, 0, 1, 1, 1])


def test_custom_span_and_drop_constant():
    df = pd.DataFrame({"period": pd.date_range("2024-03-01", periods=5, freq="D")})
    out = add_event_columns(
        df,
        events=("black_friday",),
        custom={"spring_sale": [("2024-03-02", "2024-03-03")]},
    )
    assert "control_event_black_friday" not in out.columns
    np.testing.assert_array_equal(out["control_event_spring_sale"], [0, 1, 1, 0, 0])
