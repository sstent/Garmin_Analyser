"""Gear estimation utilities for cycling workouts."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from config.settings import BikeConfig


def estimate_gear_series(
    df: pd.DataFrame,
    wheel_circumference_m: float = BikeConfig.TIRE_CIRCUMFERENCE_M,
    valid_configurations: dict = BikeConfig.VALID_CONFIGURATIONS,
) -> pd.Series:
    """Estimate gear per sample using speed and cadence data.

    Args:
        df: DataFrame with 'speed_mps' and 'cadence_rpm' columns
        wheel_circumference_m: Wheel circumference in meters
        valid_configurations: Dict of chainring -> list of cogs

    Returns:
        Series with gear strings (e.g., '38x16') aligned to input index
    """
    pass


def compute_gear_summary(gear_series: pd.Series) -> dict:
    """Compute summary statistics from gear series.

    Args:
        gear_series: Series of gear strings

    Returns:
        Dict with summary metrics
    """
    pass