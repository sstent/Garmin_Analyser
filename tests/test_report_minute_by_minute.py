import pytest
import pandas as pd
import numpy as np

from visualizers.report_generator import ReportGenerator


@pytest.fixture
def report_generator():
    return ReportGenerator()


def _create_synthetic_df(
    seconds,
    speed_mps=10,
    distance_m=None,
    hr=None,
    cadence=None,
    gradient=None,
    elevation=None,
    power=None,
    power_estimate=None,
):
    data = {
        "timestamp": pd.to_datetime(np.arange(seconds), unit="s"),
        "speed": np.full(seconds, speed_mps),
    }
    if distance_m is not None:
        data["distance"] = distance_m
    if hr is not None:
        data["heart_rate"] = hr
    if cadence is not None:
        data["cadence"] = cadence
    if gradient is not None:
        data["gradient"] = gradient
    if elevation is not None:
        data["elevation"] = elevation
    if power is not None:
        data["power"] = power
    if power_estimate is not None:
        data["power_estimate"] = power_estimate

    df = pd.DataFrame(data)
    df = df.set_index("timestamp").reset_index()
    return df


def test_aggregate_minute_by_minute_keys(report_generator):
    df = _create_synthetic_df(
        180,
        distance_m=np.linspace(0, 1000, 180),
        hr=np.full(180, 150),
        cadence=np.full(180, 90),
        gradient=np.full(180, 1.0),
        elevation=np.linspace(0, 10, 180),
        power=np.full(180, 200),
        power_estimate=np.full(180, 190),
    )
    result = report_generator._aggregate_minute_by_minute(df, {})
    expected_keys = [
        "minute_index",
        "distance_km",
        "avg_speed_kmh",
        "avg_cadence",
        "avg_hr",
        "max_hr",
        "avg_gradient",
        "elevation_change",
        "avg_real_power",
        "avg_power_estimate",
    ]
    assert len(result) == 3
    for row in result:
        for key in expected_keys:
            assert key in row


def test_speed_and_distance_conversion(report_generator):
    df = _create_synthetic_df(60, speed_mps=10)  # 10 m/s = 36 km/h
    result = report_generator._aggregate_minute_by_minute(df, {})
    assert len(result) == 1
    assert result[0]["avg_speed_kmh"] == pytest.approx(36.0, 0.01)
    # Distance integrated from speed: 10 m/s * 60s = 600m = 0.6 km
    assert "distance_km" not in result[0]


def test_distance_from_cumulative_column(report_generator):
    distance = np.linspace(0, 700, 120)  # 700m over 2 mins
    df = _create_synthetic_df(120, distance_m=distance)
    result = report_generator._aggregate_minute_by_minute(df, {})
    assert len(result) == 2
    # First minute: 350m travelled
    assert result[0]["distance_km"] == pytest.approx(0.35, 0.01)
    # Second minute: 350m travelled
    assert result[1]["distance_km"] == pytest.approx(0.35, 0.01)


def test_nan_safety_for_optional_metrics(report_generator):
    hr_with_nan = np.array([150, 155, np.nan, 160] * 15)  # 60s
    df = _create_synthetic_df(60, hr=hr_with_nan)
    result = report_generator._aggregate_minute_by_minute(df, {})
    assert len(result) == 1
    assert result[0]["avg_hr"] == pytest.approx(np.nanmean(hr_with_nan))
    assert result[0]["max_hr"] == 160
    assert "avg_cadence" not in result[0]
    assert "avg_gradient" not in result[0]


def test_all_nan_metrics(report_generator):
    hr_all_nan = np.full(60, np.nan)
    df = _create_synthetic_df(60, hr=hr_all_nan)
    result = report_generator._aggregate_minute_by_minute(df, {})
    assert len(result) == 1
    assert "avg_hr" not in result[0]
    assert "max_hr" not in result[0]


def test_rounding_precision(report_generator):
    df = _create_synthetic_df(60, speed_mps=10.12345, hr=[150.123] * 60)
    result = report_generator._aggregate_minute_by_minute(df, {})
    assert result[0]["avg_speed_kmh"] == 36.44  # 10.12345 * 3.6 rounded
    assert result[0]["distance_km"] == 0.61  # 607.407m / 1000 rounded
    assert result[0]["avg_hr"] == 150.1


def test_power_selection_logic(report_generator):
    # Case 1: Only real power
    df_real = _create_synthetic_df(60, power=[200] * 60)
    res_real = report_generator._aggregate_minute_by_minute(df_real, {})[0]
    assert res_real["avg_real_power"] == 200
    assert "avg_power_estimate" not in res_real

    # Case 2: Only estimated power
    df_est = _create_synthetic_df(60, power_estimate=[180] * 60)
    res_est = report_generator._aggregate_minute_by_minute(df_est, {})[0]
    assert "avg_real_power" not in res_est
    assert res_est["avg_power_estimate"] == 180

    # Case 3: Both present
    df_both = _create_synthetic_df(60, power=[200] * 60, power_estimate=[180] * 60)
    res_both = report_generator._aggregate_minute_by_minute(df_both, {})[0]
    assert res_both["avg_real_power"] == 200
    assert res_both["avg_power_estimate"] == 180

    # Case 4: None present
    df_none = _create_synthetic_df(60)
    res_none = report_generator._aggregate_minute_by_minute(df_none, {})[0]
    assert "avg_real_power" not in res_none
    assert "avg_power_estimate" not in res_none