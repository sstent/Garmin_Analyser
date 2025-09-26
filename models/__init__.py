"""Data models for Garmin Analyser."""

from .workout import WorkoutData, WorkoutMetadata, PowerData, HeartRateData, SpeedData, ElevationData, GearData
from .zones import ZoneDefinition, ZoneCalculator

__all__ = [
    'WorkoutData', 
    'WorkoutMetadata', 
    'PowerData', 
    'HeartRateData', 
    'SpeedData', 
    'ElevationData', 
    'GearData',
    'ZoneDefinition', 
    'ZoneCalculator'
]