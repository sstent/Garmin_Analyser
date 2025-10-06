"""Data models for workout analysis."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd


@dataclass
class WorkoutMetadata:
    """Metadata for a workout session."""
    
    activity_id: str
    activity_name: str
    start_time: datetime
    duration_seconds: float
    distance_meters: Optional[float] = None
    avg_heart_rate: Optional[float] = None
    max_heart_rate: Optional[float] = None
    avg_power: Optional[float] = None
    max_power: Optional[float] = None
    avg_speed: Optional[float] = None
    max_speed: Optional[float] = None
    elevation_gain: Optional[float] = None
    elevation_loss: Optional[float] = None
    calories: Optional[float] = None
    sport: str = "cycling"
    sub_sport: Optional[str] = None
    is_indoor: bool = False


@dataclass
class PowerData:
    """Power-related data for a workout."""
    
    power_values: List[float]
    estimated_power: List[float]
    power_zones: Dict[str, int]
    normalized_power: Optional[float] = None
    intensity_factor: Optional[float] = None
    training_stress_score: Optional[float] = None
    power_distribution: Dict[str, float] = None


@dataclass
class HeartRateData:
    """Heart rate data for a workout."""
    
    heart_rate_values: List[float]
    hr_zones: Dict[str, int]
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    hr_distribution: Dict[str, float] = None


@dataclass
class SpeedData:
    """Speed and distance data for a workout."""
    
    speed_values: List[float]
    distance_values: List[float]
    avg_speed: Optional[float] = None
    max_speed: Optional[float] = None
    total_distance: Optional[float] = None


@dataclass
class ElevationData:
    """Elevation and gradient data for a workout."""
    
    elevation_values: List[float]
    gradient_values: List[float]
    elevation_gain: Optional[float] = None
    elevation_loss: Optional[float] = None
    max_gradient: Optional[float] = None
    min_gradient: Optional[float] = None


@dataclass
class GearData:
    """Gear-related data for a workout."""

    series: pd.Series  # Per-sample gear selection with columns: chainring_teeth, cog_teeth, gear_ratio, confidence
    summary: Dict[str, Any]  # Time-in-gear distribution, top N gears by time, unique gears count


@dataclass
class WorkoutData:
    """Complete workout data structure."""
    
    metadata: WorkoutMetadata
    power: Optional[PowerData] = None
    heart_rate: Optional[HeartRateData] = None
    speed: Optional[SpeedData] = None
    elevation: Optional[ElevationData] = None
    gear: Optional[GearData] = None
    raw_data: Optional[pd.DataFrame] = None
    
    @property
    def has_power_data(self) -> bool:
        """Check if actual power data is available."""
        return self.power is not None and any(p > 0 for p in self.power.power_values)
    
    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.metadata.duration_seconds / 60
    
    @property
    def distance_km(self) -> Optional[float]:
        """Get distance in kilometers."""
        if self.metadata.distance_meters is None:
            return None
        return self.metadata.distance_meters / 1000
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workout."""
        return {
            "activity_id": self.metadata.activity_id,
            "activity_name": self.metadata.activity_name,
            "start_time": self.metadata.start_time.isoformat(),
            "duration_minutes": round(self.duration_minutes, 1),
            "distance_km": round(self.distance_km, 2) if self.distance_km else None,
            "avg_heart_rate": self.metadata.avg_heart_rate,
            "max_heart_rate": self.metadata.max_heart_rate,
            "avg_power": self.metadata.avg_power,
            "max_power": self.metadata.max_power,
            "elevation_gain": self.metadata.elevation_gain,
            "is_indoor": self.metadata.is_indoor,
            "has_power_data": self.has_power_data
        }