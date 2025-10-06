"""
Tests for speed_analysis and normalized naming in the workout analyzer.

Validates that [WorkoutAnalyzer.analyze_workout()](analyzers/workout_analyzer.py:1)
returns the expected `speed_analysis` dictionary and that the summary dictionary
contains normalized keys with backward-compatibility aliases.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from analyzers.workout_analyzer import WorkoutAnalyzer
from models.workout import WorkoutData, WorkoutMetadata, SpeedData, HeartRateData

@pytest.fixture
def synthetic_workout_data():
    """Create a small, synthetic workout dataset for testing."""
    timestamps = np.arange(60)
    speeds = np.linspace(5, 10, 60)  # speed in m/s
    heart_rates = np.linspace(120, 150, 60)

    # Introduce some NaNs to test robustness
    speeds[10] = np.nan
    heart_rates[20] = np.nan

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps, unit='s'),
        'speed_mps': speeds,
        'heart_rate': heart_rates,
    })
    
    metadata = WorkoutMetadata(
        activity_id="test_activity_123",
        activity_name="Test Ride",
        start_time=datetime(2023, 1, 1, 10, 0, 0),
        duration_seconds=60.0,
        distance_meters=1000.0,  # Adding distance_meters to resolve TypeError in template rendering tests
        sport="cycling",
        sub_sport="road"
    )

    distance_values = (df['speed_mps'].fillna(0) * 1).cumsum().tolist() # Assuming 1Hz sampling
    speed_data = SpeedData(speed_values=df['speed_mps'].fillna(0).tolist(), distance_values=distance_values)
    heart_rate_data = HeartRateData(heart_rate_values=df['heart_rate'].fillna(0).tolist(), hr_zones={}) # Dummy hr_zones

    return WorkoutData(
        metadata=metadata,
        raw_data=df,
        speed=speed_data,
        heart_rate=heart_rate_data
    )


def test_analyze_workout_includes_speed_analysis_and_normalized_summary(synthetic_workout_data):
    """
    Verify that `analyze_workout` returns 'speed_analysis' and a summary with
    normalized keys 'avg_speed_kmh' and 'avg_hr'.
    """
    analyzer = WorkoutAnalyzer()
    analysis = analyzer.analyze_workout(synthetic_workout_data)

    # 1. Validate 'speed_analysis' presence and keys
    assert 'speed_analysis' in analysis
    assert isinstance(analysis['speed_analysis'], dict)
    assert 'avg_speed_kmh' in analysis['speed_analysis']
    assert 'max_speed_kmh' in analysis['speed_analysis']
    
    # Check that values are plausible floats > 0
    assert isinstance(analysis['speed_analysis']['avg_speed_kmh'], float)
    assert isinstance(analysis['speed_analysis']['max_speed_kmh'], float)
    assert analysis['speed_analysis']['avg_speed_kmh'] > 0
    assert analysis['speed_analysis']['max_speed_kmh'] > 0

    # 2. Validate 'summary' presence and normalized keys
    assert 'summary' in analysis
    assert isinstance(analysis['summary'], dict)
    assert 'avg_speed_kmh' in analysis['summary']
    assert 'avg_hr' in analysis['summary']

    # Check that values are plausible floats > 0
    assert isinstance(analysis['summary']['avg_speed_kmh'], float)
    assert isinstance(analysis['summary']['avg_hr'], float)
    assert analysis['summary']['avg_speed_kmh'] > 0
    assert analysis['summary']['avg_hr'] > 0


def test_backward_compatibility_aliases_present(synthetic_workout_data):
    """
    Verify that `analyze_workout` summary includes backward-compatibility
    aliases for avg_speed and avg_heart_rate.
    """
    analyzer = WorkoutAnalyzer()
    analysis = analyzer.analyze_workout(synthetic_workout_data)

    assert 'summary' in analysis
    summary = analysis['summary']

    # 1. Check for 'avg_speed' alias
    assert 'avg_speed' in summary
    assert summary['avg_speed'] == summary['avg_speed_kmh']

    # 2. Check for 'avg_heart_rate' alias
    assert 'avg_heart_rate' in summary
    assert summary['avg_heart_rate'] == summary['avg_hr']