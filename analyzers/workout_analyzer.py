"""Workout data analyzer for calculating metrics and insights."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta

from ..models.workout import WorkoutData, PowerData, HeartRateData, SpeedData, ElevationData
from ..models.zones import ZoneCalculator

logger = logging.getLogger(__name__)


class WorkoutAnalyzer:
    """Analyzer for workout data to calculate metrics and insights."""
    
    def __init__(self):
        """Initialize workout analyzer."""
        self.zone_calculator = ZoneCalculator()
        self.BIKE_WEIGHT_LBS = 18.0  # Default bike weight in lbs
        self.RIDER_WEIGHT_LBS = 170.0  # Default rider weight in lbs
        self.WHEEL_CIRCUMFERENCE = 2.105  # Standard 700c wheel circumference in meters
        self.CHAINRING_TEETH = 38  # Default chainring teeth
        self.CASSETTE_OPTIONS = [14, 16, 18, 20]  # Available cog sizes
        self.BIKE_WEIGHT_KG = 8.16  # Bike weight in kg
        self.TIRE_CIRCUMFERENCE_M = 2.105  # Tire circumference in meters
        self.POWER_DATA_AVAILABLE = False  # Flag for real power data availability
        self.IS_INDOOR = False  # Flag for indoor workouts
    
    def analyze_workout(self, workout: WorkoutData, cog_size: int = 16) -> Dict[str, Any]:
        """Analyze a workout and return comprehensive metrics."""
        # Estimate power if not available
        estimated_power = self._estimate_power(workout, cog_size)
        
        return {
            'metadata': workout.metadata.__dict__,
            'summary': self._calculate_summary_metrics(workout, estimated_power),
            'power_analysis': self._analyze_power(workout, estimated_power),
            'heart_rate_analysis': self._analyze_heart_rate(workout),
            'cadence_analysis': self._analyze_cadence(workout),
            'elevation_analysis': self._analyze_elevation(workout),
            'intervals': self._detect_intervals(workout, estimated_power),
            'zones': self._calculate_zone_distribution(workout, estimated_power),
            'efficiency': self._calculate_efficiency_metrics(workout, estimated_power),
            'cog_size': cog_size,
            'estimated_power': estimated_power
        }
    
    def _calculate_summary_metrics(self, workout: WorkoutData, estimated_power: List[float] = None) -> Dict[str, Any]:
        """Calculate basic summary metrics.
        
        Args:
            workout: WorkoutData object
            estimated_power: List of estimated power values (optional)
            
        Returns:
            Dictionary with summary metrics
        """
        df = workout.raw_data
        
        # Determine which power values to use
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
            power_source = 'real'
        elif estimated_power:
            power_values = estimated_power
            power_source = 'estimated'
        else:
            power_values = []
            power_source = 'none'
        
        summary = {
            'duration_minutes': workout.metadata.duration_seconds / 60,
            'distance_km': workout.metadata.distance_meters / 1000 if workout.metadata.distance_meters else None,
            'avg_speed_kmh': None,
            'max_speed_kmh': None,
            'avg_power': np.mean(power_values) if power_values else 0,
            'max_power': np.max(power_values) if power_values else 0,
            'avg_heart_rate': workout.metadata.avg_heart_rate,
            'max_heart_rate': workout.metadata.max_heart_rate,
            'elevation_gain_m': workout.metadata.elevation_gain,
            'calories': workout.metadata.calories,
            'work_kj': None,
            'normalized_power': None,
            'intensity_factor': None,
            'training_stress_score': None,
            'power_source': power_source
        }
        
        # Calculate speed metrics
        if workout.speed and workout.speed.speed_values:
            summary['avg_speed_kmh'] = np.mean(workout.speed.speed_values)
            summary['max_speed_kmh'] = np.max(workout.speed.speed_values)
        
        # Calculate work (power * time)
        if power_values:
            duration_hours = workout.metadata.duration_seconds / 3600
            summary['work_kj'] = np.mean(power_values) * duration_hours * 3.6  # kJ
            
            # Calculate normalized power
            summary['normalized_power'] = self._calculate_normalized_power(power_values)
            
            # Calculate IF and TSS (assuming FTP of 250W)
            ftp = 250  # Default FTP, should be configurable
            summary['intensity_factor'] = summary['normalized_power'] / ftp
            summary['training_stress_score'] = (
                (summary['duration_minutes'] * summary['normalized_power'] * summary['intensity_factor']) /
                (ftp * 3600) * 100
            )
        
        return summary
    
    def _analyze_power(self, workout: WorkoutData, estimated_power: List[float] = None) -> Dict[str, Any]:
        """Analyze power data.
        
        Args:
            workout: WorkoutData object
            estimated_power: List of estimated power values (optional)
            
        Returns:
            Dictionary with power analysis
        """
        # Determine which power values to use
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
            power_source = 'real'
        elif estimated_power:
            power_values = estimated_power
            power_source = 'estimated'
        else:
            return {}
        
        # Calculate power zones
        power_zones = self.zone_calculator.get_power_zones()
        zone_distribution = self.zone_calculator.calculate_zone_distribution(
            power_values, power_zones
        )
        
        # Calculate power metrics
        power_analysis = {
            'avg_power': np.mean(power_values),
            'max_power': np.max(power_values),
            'min_power': np.min(power_values),
            'power_std': np.std(power_values),
            'power_variability': np.std(power_values) / np.mean(power_values),
            'normalized_power': self._calculate_normalized_power(power_values),
            'power_zones': zone_distribution,
            'power_spikes': self._detect_power_spikes(power_values),
            'power_distribution': self._calculate_power_distribution(power_values),
            'power_source': power_source
        }
        
        return power_analysis
    
    def _analyze_heart_rate(self, workout: WorkoutData) -> Dict[str, Any]:
        """Analyze heart rate data.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Dictionary with heart rate analysis
        """
        if not workout.heart_rate or not workout.heart_rate.heart_rate_values:
            return {}
        
        hr_values = workout.heart_rate.heart_rate_values
        
        # Calculate heart rate zones
        hr_zones = self.zone_calculator.get_heart_rate_zones()
        zone_distribution = self.zone_calculator.calculate_zone_distribution(
            hr_values, hr_zones
        )
        
        # Calculate heart rate metrics
        hr_analysis = {
            'avg_hr': np.mean(hr_values),
            'max_hr': np.max(hr_values),
            'min_hr': np.min(hr_values),
            'hr_std': np.std(hr_values),
            'hr_zones': zone_distribution,
            'hr_recovery': self._calculate_hr_recovery(workout),
            'hr_distribution': self._calculate_hr_distribution(hr_values)
        }
        
        return hr_analysis
    
    def _analyze_speed(self, workout: WorkoutData) -> Dict[str, Any]:
        """Analyze speed data.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Dictionary with speed analysis
        """
        if not workout.speed or not workout.speed.speed_values:
            return {}
        
        speed_values = workout.speed.speed_values
        
        # Calculate speed zones
        speed_zones = {
            'Recovery': (0, 15),
            'Endurance': (15, 25),
            'Tempo': (25, 30),
            'Threshold': (30, 35),
            'VO2 Max': (35, 100)
        }
        
        zone_distribution = {}
        for zone_name, (min_speed, max_speed) in speed_zones.items():
            count = sum(1 for s in speed_values if min_speed <= s < max_speed)
            zone_distribution[zone_name] = (count / len(speed_values)) * 100
        
        speed_analysis = {
            'avg_speed_kmh': np.mean(speed_values),
            'max_speed_kmh': np.max(speed_values),
            'min_speed_kmh': np.min(speed_values),
            'speed_std': np.std(speed_values),
            'speed_zones': zone_distribution,
            'speed_distribution': self._calculate_speed_distribution(speed_values)
        }
        
        return speed_analysis
    
    def _analyze_elevation(self, workout: WorkoutData) -> Dict[str, Any]:
        """Analyze elevation data.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Dictionary with elevation analysis
        """
        if not workout.elevation or not workout.elevation.elevation_values:
            return {}
        
        elevation_values = workout.elevation.elevation_values
        
        # Calculate elevation metrics
        elevation_analysis = {
            'elevation_gain': workout.elevation.elevation_gain,
            'elevation_loss': workout.elevation.elevation_loss,
            'max_elevation': np.max(elevation_values),
            'min_elevation': np.min(elevation_values),
            'avg_gradient': np.mean(workout.elevation.gradient_values),
            'max_gradient': np.max(workout.elevation.gradient_values),
            'min_gradient': np.min(workout.elevation.gradient_values),
            'climbing_ratio': self._calculate_climbing_ratio(elevation_values)
        }
        
        return elevation_analysis
    
    def _detect_intervals(self, workout: WorkoutData, estimated_power: List[float] = None) -> List[Dict[str, Any]]:
        """Detect intervals in the workout.
        
        Args:
            workout: WorkoutData object
            estimated_power: List of estimated power values (optional)
            
        Returns:
            List of interval dictionaries
        """
        # Determine which power values to use
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
        elif estimated_power:
            power_values = estimated_power
        else:
            return []
        
        # Simple interval detection based on power
        threshold = np.percentile(power_values, 75)  # Top 25% as intervals
        
        intervals = []
        in_interval = False
        start_idx = 0
        
        for i, power in enumerate(power_values):
            if power >= threshold and not in_interval:
                # Start of interval
                in_interval = True
                start_idx = i
            elif power < threshold and in_interval:
                # End of interval
                in_interval = False
                if i - start_idx >= 30:  # Minimum 30 seconds
                    interval_data = {
                        'start_index': start_idx,
                        'end_index': i,
                        'duration_seconds': (i - start_idx) * 1,  # Assuming 1s intervals
                        'avg_power': np.mean(power_values[start_idx:i]),
                        'max_power': np.max(power_values[start_idx:i]),
                        'type': 'high_intensity'
                    }
                    intervals.append(interval_data)
        
        return intervals
    
    def _calculate_zone_distribution(self, workout: WorkoutData, estimated_power: List[float] = None) -> Dict[str, Any]:
        """Calculate time spent in each training zone.
        
        Args:
            workout: WorkoutData object
            estimated_power: List of estimated power values (optional)
            
        Returns:
            Dictionary with zone distributions
        """
        zones = {}
        
        # Power zones - use real power if available, otherwise estimated
        power_values = None
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
        elif estimated_power:
            power_values = estimated_power
            
        if power_values:
            power_zones = self.zone_calculator.get_power_zones()
            zones['power'] = self.zone_calculator.calculate_zone_distribution(
                power_values, power_zones
            )
        
        # Heart rate zones
        if workout.heart_rate and workout.heart_rate.heart_rate_values:
            hr_zones = self.zone_calculator.get_heart_rate_zones()
            zones['heart_rate'] = self.zone_calculator.calculate_zone_distribution(
                workout.heart_rate.heart_rate_values, hr_zones
            )
        
        # Speed zones
        if workout.speed and workout.speed.speed_values:
            speed_zones = {
                'Recovery': (0, 15),
                'Endurance': (15, 25),
                'Tempo': (25, 30),
                'Threshold': (30, 35),
                'VO2 Max': (35, 100)
            }
            zones['speed'] = self.zone_calculator.calculate_zone_distribution(
                workout.speed.speed_values, speed_zones
            )
        
        return zones
    
    def _calculate_efficiency_metrics(self, workout: WorkoutData, estimated_power: List[float] = None) -> Dict[str, Any]:
        """Calculate efficiency metrics.
        
        Args:
            workout: WorkoutData object
            estimated_power: List of estimated power values (optional)
            
        Returns:
            Dictionary with efficiency metrics
        """
        efficiency = {}
        
        # Determine which power values to use
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
        elif estimated_power:
            power_values = estimated_power
        else:
            return efficiency
        
        # Power-to-heart rate ratio
        if workout.heart_rate and workout.heart_rate.heart_rate_values:
            hr_values = workout.heart_rate.heart_rate_values
            
            # Align arrays (assuming same length)
            min_len = min(len(power_values), len(hr_values))
            if min_len > 0:
                power_efficiency = [
                    p / hr for p, hr in zip(power_values[:min_len], hr_values[:min_len])
                    if hr > 0
                ]
                
                if power_efficiency:
                    efficiency['power_to_hr_ratio'] = np.mean(power_efficiency)
        
        # Decoupling (power vs heart rate drift)
        if len(workout.raw_data) > 100:
            df = workout.raw_data.copy()
            
            # Add estimated power to dataframe if provided
            if estimated_power and len(estimated_power) == len(df):
                df['power'] = estimated_power
            
            # Split workout into halves
            mid_point = len(df) // 2
            
            if 'power' in df.columns and 'heart_rate' in df.columns:
                first_half = df.iloc[:mid_point]
                second_half = df.iloc[mid_point:]
                
                if not first_half.empty and not second_half.empty:
                    first_power = first_half['power'].mean()
                    second_power = second_half['power'].mean()
                    first_hr = first_half['heart_rate'].mean()
                    second_hr = second_half['heart_rate'].mean()
                    
                    if first_power > 0 and first_hr > 0:
                        power_ratio = second_power / first_power
                        hr_ratio = second_hr / first_hr
                        efficiency['decoupling'] = (hr_ratio - power_ratio) * 100
        
        return efficiency
    
    def _calculate_normalized_power(self, power_values: List[float]) -> float:
        """Calculate normalized power using 30-second rolling average.
        
        Args:
            power_values: List of power values
            
        Returns:
            Normalized power value
        """
        if not power_values:
            return 0.0
        
        # Convert to pandas Series for rolling calculation
        power_series = pd.Series(power_values)
        
        # 30-second rolling average (assuming 1Hz data)
        rolling_avg = power_series.rolling(window=30, min_periods=1).mean()
        
        # Raise to 4th power, average, then 4th root
        normalized = (rolling_avg ** 4).mean() ** 0.25
        
        return float(normalized)
    
    def _detect_power_spikes(self, power_values: List[float]) -> List[Dict[str, Any]]:
        """Detect power spikes in the data.
        
        Args:
            power_values: List of power values
            
        Returns:
            List of spike dictionaries
        """
        if not power_values:
            return []
        
        mean_power = np.mean(power_values)
        std_power = np.std(power_values)
        
        # Define spike as > 2 standard deviations above mean
        spike_threshold = mean_power + 2 * std_power
        
        spikes = []
        for i, power in enumerate(power_values):
            if power > spike_threshold:
                spikes.append({
                    'index': i,
                    'power': power,
                    'deviation': (power - mean_power) / std_power
                })
        
        return spikes
    
    def _calculate_power_distribution(self, power_values: List[float]) -> Dict[str, float]:
        """Calculate power distribution statistics.
        
        Args:
            power_values: List of power values
            
        Returns:
            Dictionary with power distribution metrics
        """
        if not power_values:
            return {}
        
        percentiles = [5, 25, 50, 75, 95]
        distribution = {}
        
        for p in percentiles:
            distribution[f'p{p}'] = float(np.percentile(power_values, p))
        
        return distribution
    
    def _calculate_hr_distribution(self, hr_values: List[float]) -> Dict[str, float]:
        """Calculate heart rate distribution statistics.
        
        Args:
            hr_values: List of heart rate values
            
        Returns:
            Dictionary with HR distribution metrics
        """
        if not hr_values:
            return {}
        
        percentiles = [5, 25, 50, 75, 95]
        distribution = {}
        
        for p in percentiles:
            distribution[f'p{p}'] = float(np.percentile(hr_values, p))
        
        return distribution
    
    def _calculate_speed_distribution(self, speed_values: List[float]) -> Dict[str, float]:
        """Calculate speed distribution statistics.
        
        Args:
            speed_values: List of speed values
            
        Returns:
            Dictionary with speed distribution metrics
        """
        if not speed_values:
            return {}
        
        percentiles = [5, 25, 50, 75, 95]
        distribution = {}
        
        for p in percentiles:
            distribution[f'p{p}'] = float(np.percentile(speed_values, p))
        
        return distribution
    
    def _calculate_hr_recovery(self, workout: WorkoutData) -> Optional[float]:
        """Calculate heart rate recovery (not implemented).
        
        Args:
            workout: WorkoutData object
            
        Returns:
            HR recovery value or None
        """
        # This would require post-workout data
        return None
    
    def _calculate_climbing_ratio(self, elevation_values: List[float]) -> float:
        """Calculate climbing ratio (elevation gain per km).
        
        Args:
            elevation_values: List of elevation values
            
        Returns:
            Climbing ratio in m/km
        """
        if not elevation_values:
            return 0.0
        
        total_elevation_gain = max(elevation_values) - min(elevation_values)
        # Assume 10m between points for distance calculation
        total_distance_km = len(elevation_values) * 10 / 1000
        
        return total_elevation_gain / total_distance_km if total_distance_km > 0 else 0.0
    
    def _analyze_cadence(self, workout: WorkoutData) -> Dict[str, Any]:
        """Analyze cadence data.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Dictionary with cadence analysis
        """
        if not workout.raw_data.empty and 'cadence' in workout.raw_data.columns:
            cadence_values = workout.raw_data['cadence'].dropna().tolist()
            if cadence_values:
                return {
                    'avg_cadence': np.mean(cadence_values),
                    'max_cadence': np.max(cadence_values),
                    'min_cadence': np.min(cadence_values),
                    'cadence_std': np.std(cadence_values)
                }
        return {}
    
    def _estimate_power(self, workout: WorkoutData, cog_size: int = 16) -> List[float]:
        """Estimate power based on speed, cadence, and elevation data.
        
        Args:
            workout: WorkoutData object
            cog_size: Cog size in teeth for power estimation
            
        Returns:
            List of estimated power values
        """
        if workout.raw_data.empty:
            return []
        
        df = workout.raw_data
        
        # Check if real power data is available
        if 'power' in df.columns and df['power'].notna().any():
            self.POWER_DATA_AVAILABLE = True
            return df['power'].fillna(0).tolist()
        
        # Estimate power based on available data
        estimated_power = []
        
        for idx, row in df.iterrows():
            speed = row.get('speed', 0)
            cadence = row.get('cadence', 0)
            elevation = row.get('elevation', 0)
            gradient = row.get('grade', 0)
            
            # Basic power estimation formula
            # Power = (rolling resistance + air resistance + gravity) * speed
            
            # Constants
            rolling_resistance_coeff = 0.005  # Coefficient of rolling resistance
            air_density = 1.225  # kg/m³
            drag_coeff = 0.5  # Drag coefficient
            frontal_area = 0.5  # m²
            
            # Calculate forces
            total_weight = (self.RIDER_WEIGHT_LBS + self.BIKE_WEIGHT_LBS) * 0.453592  # Convert to kg
            
            # Rolling resistance
            rolling_force = rolling_resistance_coeff * total_weight * 9.81
            
            # Air resistance (simplified)
            air_force = 0.5 * air_density * drag_coeff * frontal_area * (speed / 3.6) ** 2
            
            # Gravity component
            gravity_force = total_weight * 9.81 * np.sin(np.arctan(gradient / 100))
            
            # Total power in watts
            total_power = (rolling_force + air_force + gravity_force) * (speed / 3.6)
            
            # Adjust based on cadence and gear ratio
            if cadence > 0:
                gear_ratio = self.CHAINRING_TEETH / cog_size
                cadence_factor = min(cadence / 90, 1.5)  # Normalize cadence
                total_power *= cadence_factor
            
            estimated_power.append(max(total_power, 0))
        
        return estimated_power