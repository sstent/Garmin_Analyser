"""Zone definitions and calculations for workouts."""

from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ZoneDefinition:
    """Definition of a training zone."""
    
    name: str
    min_value: float
    max_value: float
    color: str
    description: str


class ZoneCalculator:
    """Calculator for various training zones."""
    
    @staticmethod
    def get_power_zones() -> Dict[str, ZoneDefinition]:
        """Get power zone definitions."""
        return {
            'Recovery': ZoneDefinition(
                name='Recovery',
                min_value=0,
                max_value=150,
                color='lightblue',
                description='Active recovery, very light effort'
            ),
            'Endurance': ZoneDefinition(
                name='Endurance',
                min_value=150,
                max_value=200,
                color='green',
                description='Aerobic base, sustainable for hours'
            ),
            'Tempo': ZoneDefinition(
                name='Tempo',
                min_value=200,
                max_value=250,
                color='yellow',
                description='Sweet spot, sustainable for 20-60 minutes'
            ),
            'Threshold': ZoneDefinition(
                name='Threshold',
                min_value=250,
                max_value=300,
                color='orange',
                description='Functional threshold power, 20-60 minutes'
            ),
            'VO2 Max': ZoneDefinition(
                name='VO2 Max',
                min_value=300,
                max_value=1000,
                color='red',
                description='Maximum aerobic capacity, 3-8 minutes'
            )
        }
    
    @staticmethod
    def get_heart_rate_zones(lthr: int = 170) -> Dict[str, ZoneDefinition]:
        """Get heart rate zone definitions based on lactate threshold.
        
        Args:
            lthr: Lactate threshold heart rate in bpm
            
        Returns:
            Dictionary of heart rate zones
        """
        return {
            'Z1': ZoneDefinition(
                name='Zone 1',
                min_value=0,
                max_value=int(lthr * 0.8),
                color='lightblue',
                description='Active recovery, <80% LTHR'
            ),
            'Z2': ZoneDefinition(
                name='Zone 2',
                min_value=int(lthr * 0.8),
                max_value=int(lthr * 0.87),
                color='green',
                description='Aerobic base, 80-87% LTHR'
            ),
            'Z3': ZoneDefinition(
                name='Zone 3',
                min_value=int(lthr * 0.87) + 1,
                max_value=int(lthr * 0.93),
                color='yellow',
                description='Tempo, 88-93% LTHR'
            ),
            'Z4': ZoneDefinition(
                name='Zone 4',
                min_value=int(lthr * 0.93) + 1,
                max_value=int(lthr * 0.99),
                color='orange',
                description='Threshold, 94-99% LTHR'
            ),
            'Z5': ZoneDefinition(
                name='Zone 5',
                min_value=int(lthr * 0.99) + 1,
                max_value=300,
                color='red',
                description='VO2 Max, >99% LTHR'
            )
        }
    
    @staticmethod
    def calculate_zone_distribution(values: List[float], zones: Dict[str, ZoneDefinition]) -> Dict[str, float]:
        """Calculate time spent in each zone.
        
        Args:
            values: List of values (power, heart rate, etc.)
            zones: Zone definitions
            
        Returns:
            Dictionary with percentage time in each zone
        """
        if not values:
            return {zone_name: 0.0 for zone_name in zones.keys()}
        
        zone_counts = {zone_name: 0 for zone_name in zones.keys()}
        
        for value in values:
            for zone_name, zone_def in zones.items():
                if zone_def.min_value <= value <= zone_def.max_value:
                    zone_counts[zone_name] += 1
                    break
        
        total_count = len(values)
        return {
            zone_name: (count / total_count) * 100
            for zone_name, count in zone_counts.items()
        }
    
    @staticmethod
    def get_zone_for_value(value: float, zones: Dict[str, ZoneDefinition]) -> str:
        """Get the zone name for a given value.
        
        Args:
            value: The value to check
            zones: Zone definitions
            
        Returns:
            Zone name or 'Unknown' if not found
        """
        for zone_name, zone_def in zones.items():
            if zone_def.min_value <= value <= zone_def.max_value:
                return zone_name
        return 'Unknown'
    
    @staticmethod
    def get_cadence_zones() -> Dict[str, ZoneDefinition]:
        """Get cadence zone definitions."""
        return {
            'Recovery': ZoneDefinition(
                name='Recovery',
                min_value=0,
                max_value=80,
                color='lightblue',
                description='Low cadence, recovery pace'
            ),
            'Endurance': ZoneDefinition(
                name='Endurance',
                min_value=80,
                max_value=90,
                color='green',
                description='Comfortable cadence, sustainable'
            ),
            'Tempo': ZoneDefinition(
                name='Tempo',
                min_value=90,
                max_value=100,
                color='yellow',
                description='Moderate cadence, tempo effort'
            ),
            'Threshold': ZoneDefinition(
                name='Threshold',
                min_value=100,
                max_value=110,
                color='orange',
                description='High cadence, threshold effort'
            ),
            'Sprint': ZoneDefinition(
                name='Sprint',
                min_value=110,
                max_value=200,
                color='red',
                description='Maximum cadence, sprint effort'
            )
        }