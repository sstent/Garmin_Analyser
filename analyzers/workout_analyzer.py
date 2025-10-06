"""Workout data analyzer for calculating metrics and insights."""

import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta

from models.workout import WorkoutData, PowerData, HeartRateData, SpeedData, ElevationData
from models.zones import ZoneCalculator, ZoneDefinition
from config.settings import BikeConfig, INDOOR_KEYWORDS

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
    
    def analyze_workout(self, workout: WorkoutData, cog_size: Optional[int] = None) -> Dict[str, Any]:
        """Analyze a workout and return comprehensive metrics."""
        self.workout = workout

        if cog_size is None:
            if workout.gear and workout.gear.cassette_teeth:
                cog_size = workout.gear.cassette_teeth[0]
            else:
                cog_size = 16

        # Estimate power if not available
        estimated_power = self._estimate_power(workout, cog_size)

        analysis = {
            'metadata': workout.metadata.__dict__,
            'summary': self._calculate_summary_metrics(workout, estimated_power),
            'power_analysis': self._analyze_power(workout, estimated_power),
            'heart_rate_analysis': self._analyze_heart_rate(workout),
            'speed_analysis': self._analyze_speed(workout),
            'cadence_analysis': self._analyze_cadence(workout),
            'elevation_analysis': self._analyze_elevation(workout),
            'gear_analysis': self._analyze_gear(workout),
            'intervals': self._detect_intervals(workout, estimated_power),
            'zones': self._calculate_zone_distribution(workout, estimated_power),
            'efficiency': self._calculate_efficiency_metrics(workout, estimated_power),
            'cog_size': cog_size,
            'estimated_power': estimated_power
        }

        # Add power_estimate summary when real power is missing
        if not workout.power or not workout.power.power_values:
            analysis['power_estimate'] = {
                'avg_power': np.mean(estimated_power) if estimated_power else 0,
                'max_power': np.max(estimated_power) if estimated_power else 0
            }

        return analysis
    
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
            'avg_hr': workout.metadata.avg_heart_rate if workout.metadata.avg_heart_rate else (np.mean(workout.heart_rate.heart_rate_values) if workout.heart_rate and workout.heart_rate.heart_rate_values else 0),
            'max_hr': workout.metadata.max_heart_rate,
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
            summary['avg_speed'] = summary['avg_speed_kmh'] # Backward compatibility alias
            summary['avg_heart_rate'] = summary['avg_hr'] # Backward compatibility alias
        
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
            'avg_hr': np.mean(hr_values) if hr_values else 0,
            'max_hr': np.max(hr_values) if hr_values else 0,
            'min_hr': np.min(hr_values) if hr_values else 0,
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
        
        # Calculate speed zones (using ZoneDefinition objects)
        speed_zones = {
            'Recovery': ZoneDefinition(name='Recovery', min_value=0, max_value=15, color='blue', description=''),
            'Endurance': ZoneDefinition(name='Endurance', min_value=15, max_value=25, color='green', description=''),
            'Tempo': ZoneDefinition(name='Tempo', min_value=25, max_value=30, color='yellow', description=''),
            'Threshold': ZoneDefinition(name='Threshold', min_value=30, max_value=35, color='orange', description=''),
            'VO2 Max': ZoneDefinition(name='VO2 Max', min_value=35, max_value=100, color='red', description='')
        }
        
        zone_distribution = self.zone_calculator.calculate_zone_distribution(speed_values, speed_zones)

        zone_distribution = self.zone_calculator.calculate_zone_distribution(speed_values, speed_zones)
        
        speed_analysis = {
            'avg_speed_kmh': np.mean(speed_values),
            'max_speed_kmh': np.max(speed_values),
            'min_speed_kmh': np.min(speed_values),
            'speed_std': np.std(speed_values),
            'moving_time_s': len(speed_values),  # Assuming 1 Hz sampling
            'distance_km': workout.metadata.distance_meters / 1000 if workout.metadata.distance_meters else None,
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
                'Recovery': ZoneDefinition(name='Recovery', min_value=0, max_value=15, color='blue', description=''),
                'Endurance': ZoneDefinition(name='Endurance', min_value=15, max_value=25, color='green', description=''),
                'Tempo': ZoneDefinition(name='Tempo', min_value=25, max_value=30, color='yellow', description=''),
                'Threshold': ZoneDefinition(name='Threshold', min_value=30, max_value=35, color='orange', description=''),
                'VO2 Max': ZoneDefinition(name='VO2 Max', min_value=35, max_value=100, color='red', description='')
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
    
    def _analyze_gear(self, workout: WorkoutData) -> Dict[str, Any]:
        """Analyze gear data.

        Args:
            workout: WorkoutData object

        Returns:
            Dictionary with gear analysis
        """
        if not workout.gear or not workout.gear.series:
            return {}

        gear_series = workout.gear.series
        summary = workout.gear.summary

        # Use the summary if available, otherwise compute basic stats
        if summary:
            return {
                'time_in_top_gear_s': summary.get('time_in_top_gear_s', 0),
                'top_gears': summary.get('top_gears', []),
                'unique_gears_count': summary.get('unique_gears_count', 0),
                'gear_distribution': summary.get('gear_distribution', {})
            }

        # Fallback: compute basic gear distribution
        if not gear_series.empty:
            gear_counts = gear_series.value_counts().sort_index()
            total_samples = len(gear_series)
            gear_distribution = {
                gear: (count / total_samples) * 100
                for gear, count in gear_counts.items()
            }

            return {
                'unique_gears_count': len(gear_counts),
                'gear_distribution': gear_distribution,
                'top_gears': gear_counts.head(3).index.tolist(),
                'time_in_top_gear_s': gear_counts.iloc[0] if not gear_counts.empty else 0
            }

        return {}

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
        """Estimate power using physics-based model for indoor and outdoor workouts.

        Args:
            workout: WorkoutData object
            cog_size: Cog size in teeth (unused in this implementation)

        Returns:
            List of estimated power values
        """
        if workout.raw_data.empty:
            return []

        df = workout.raw_data.copy()

        # Check if real power data is available - prefer real power when available
        if 'power' in df.columns and df['power'].notna().any():
            logger.debug("Real power data available, skipping estimation")
            return df['power'].fillna(0).tolist()

        # Determine if this is an indoor workout
        is_indoor = workout.metadata.is_indoor if workout.metadata.is_indoor is not None else False
        if not is_indoor and workout.metadata.activity_name:
            activity_name = workout.metadata.activity_name.lower()
            is_indoor = any(keyword in activity_name for keyword in INDOOR_KEYWORDS)

        logger.info(f"Using {'indoor' if is_indoor else 'outdoor'} power estimation model")

        # Prepare speed data (prefer speed_mps, derive from distance if needed)
        if 'speed' in df.columns:
            speed_mps = df['speed'].fillna(0)
        elif 'distance' in df.columns:
            # Derive speed from cumulative distance (assuming 1 Hz sampling)
            distance_diff = df['distance'].diff().fillna(0)
            speed_mps = distance_diff.clip(lower=0)  # Ensure non-negative
        else:
            logger.warning("No speed or distance data available for power estimation")
            return [0.0] * len(df)

        # Prepare gradient data (prefer gradient_percent, derive from elevation if needed)
        if 'gradient_percent' in df.columns:
            gradient_percent = df['gradient_percent'].fillna(0)
        elif 'elevation' in df.columns:
            # Derive gradient from elevation changes (assuming 1 Hz sampling)
            elevation_diff = df['elevation'].diff().fillna(0)
            distance_diff = speed_mps  # Approximation: distance per second ≈ speed
            gradient_percent = np.where(distance_diff > 0,
                                      (elevation_diff / distance_diff) * 100,
                                      0).clip(-50, 50)  # Reasonable bounds
        else:
            logger.warning("No gradient or elevation data available for power estimation")
            gradient_percent = pd.Series([0.0] * len(df), index=df.index)

        # Indoor handling: disable aero, set gradient to 0 for unrealistic values, add baseline
        if is_indoor:
            gradient_percent = gradient_percent.where(
                (gradient_percent >= -10) & (gradient_percent <= 10), 0
            )  # Clamp unrealistic gradients
            aero_enabled = False
        else:
            aero_enabled = True

        # Constants
        g = 9.80665  # gravity m/s²
        theta = np.arctan(gradient_percent / 100)  # slope angle in radians
        m = BikeConfig.BIKE_MASS_KG  # total mass kg
        Crr = BikeConfig.BIKE_CRR
        CdA = BikeConfig.BIKE_CDA if aero_enabled else 0.0
        rho = BikeConfig.AIR_DENSITY
        eta = BikeConfig.DRIVE_EFFICIENCY

        # Compute acceleration (centered difference for smoothness)
        accel_mps2 = speed_mps.diff().fillna(0)  # Simple diff, assuming 1 Hz

        # Power components
        P_roll = Crr * m * g * speed_mps
        P_aero = 0.5 * rho * CdA * speed_mps**3
        P_grav = m * g * np.sin(theta) * speed_mps
        P_accel = m * accel_mps2 * speed_mps

        # Total power (clamp acceleration contribution to non-negative)
        P_total = (P_roll + P_aero + P_grav + np.maximum(P_accel, 0)) / eta

        # Indoor baseline
        if is_indoor:
            P_total += BikeConfig.INDOOR_BASELINE_WATTS

        # Clamp and smooth
        P_total = np.maximum(P_total, 0)  # Non-negative
        P_total = np.minimum(P_total, BikeConfig.MAX_POWER_WATTS)  # Cap spikes

        # Apply smoothing
        window = BikeConfig.POWER_ESTIMATE_SMOOTHING_WINDOW_SAMPLES
        if window > 1:
            P_total = P_total.rolling(window=window, center=True, min_periods=1).mean()

        # Fill any remaining NaN and convert to list
        power_estimate = P_total.fillna(0).tolist()

        return power_estimate