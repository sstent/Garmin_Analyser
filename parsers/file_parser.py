"""File parser for various workout formats (FIT, TCX, GPX)."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

try:
    from fitparse import FitFile
except ImportError:
    raise ImportError("fitparse package required. Install with: pip install fitparse")

from models.workout import WorkoutData, WorkoutMetadata, PowerData, HeartRateData, SpeedData, ElevationData, GearData
from config.settings import SUPPORTED_FORMATS, BikeConfig, INDOOR_KEYWORDS
from utils.gear_estimation import estimate_gear_series, compute_gear_summary

logger = logging.getLogger(__name__)


class FileParser:
    """Parser for workout files in various formats."""
    
    def __init__(self):
        """Initialize file parser."""
        pass
    
    def parse_file(self, file_path: Path) -> Optional[WorkoutData]:
        """Parse a workout file and return structured data.
        
        Args:
            file_path: Path to the workout file
            
        Returns:
            WorkoutData object or None if parsing failed
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format: {file_extension}")
            return None
        
        try:
            if file_extension == '.fit':
                return self._parse_fit(file_path)
            elif file_extension == '.tcx':
                return self._parse_tcx(file_path)
            elif file_extension == '.gpx':
                return self._parse_gpx(file_path)
            else:
                logger.error(f"Parser not implemented for format: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return None
    
    def _parse_fit(self, file_path: Path) -> Optional[WorkoutData]:
        """Parse FIT file format.
        
        Args:
            file_path: Path to FIT file
            
        Returns:
            WorkoutData object or None if parsing failed
        """
        try:
            fit_file = FitFile(str(file_path))
            
            # Extract session data
            session_data = self._extract_fit_session(fit_file)
            if not session_data:
                logger.error("No session data found in FIT file")
                return None
            
            # Extract record data (timestamp-based data)
            records = list(fit_file.get_messages('record'))
            if not records:
                logger.error("No record data found in FIT file")
                return None
            
            # Create DataFrame from records
            df = self._fit_records_to_dataframe(records)
            if df.empty:
                logger.error("No valid data extracted from FIT records")
                return None
            
            # Create metadata
            metadata = WorkoutMetadata(
                activity_id=str(session_data.get('activity_id', 'unknown')),
                activity_name=session_data.get('activity_name', 'Workout'),
                start_time=session_data.get('start_time', pd.Timestamp.now()),
                duration_seconds=session_data.get('total_timer_time', 0),
                distance_meters=session_data.get('total_distance'),
                avg_heart_rate=session_data.get('avg_heart_rate'),
                max_heart_rate=session_data.get('max_heart_rate'),
                avg_power=session_data.get('avg_power'),
                max_power=session_data.get('max_power'),
                avg_speed=session_data.get('avg_speed'),
                max_speed=session_data.get('max_speed'),
                elevation_gain=session_data.get('total_ascent'),
                elevation_loss=session_data.get('total_descent'),
                calories=session_data.get('total_calories'),
                sport=session_data.get('sport', 'cycling'),
                sub_sport=session_data.get('sub_sport'),
                is_indoor=session_data.get('is_indoor', False)
            )

            if not metadata.is_indoor and metadata.activity_name:
                metadata.is_indoor = any(keyword in metadata.activity_name.lower() for keyword in INDOOR_KEYWORDS)
            
            # Create workout data
            workout_data = WorkoutData(
                metadata=metadata,
                raw_data=df
            )
            
            # Add processed data if available
            if not df.empty:
                workout_data.power = self._extract_power_data(df)
                workout_data.heart_rate = self._extract_heart_rate_data(df)
                workout_data.speed = self._extract_speed_data(df)
                workout_data.elevation = self._extract_elevation_data(df)
                workout_data.gear = self._extract_gear_data(df)
            
            return workout_data
            
        except Exception as e:
            logger.error(f"Failed to parse FIT file {file_path}: {e}")
            return None
    
    def _extract_fit_session(self, fit_file) -> Optional[Dict[str, Any]]:
        """Extract session data from FIT file.
        
        Args:
            fit_file: FIT file object
            
        Returns:
            Dictionary with session data
        """
        try:
            sessions = list(fit_file.get_messages('session'))
            if not sessions:
                return None
            
            session = sessions[0]
            data = {}
            
            for field in session:
                if field.name and field.value is not None:
                    data[field.name] = field.value
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to extract session data: {e}")
            return None
    
    def _fit_records_to_dataframe(self, records) -> pd.DataFrame:
        """Convert FIT records to pandas DataFrame.
        
        Args:
            records: List of FIT record messages
            
        Returns:
            DataFrame with workout data
        """
        data = []
        
        for record in records:
            record_data = {}
            for field in record:
                if field.name and field.value is not None:
                    record_data[field.name] = field.value
            data.append(record_data)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            df = df.reset_index(drop=True)
        
        return df
    
    def _extract_power_data(self, df: pd.DataFrame) -> Optional[PowerData]:
        """Extract power data from DataFrame.
        
        Args:
            df: DataFrame with workout data
            
        Returns:
            PowerData object or None
        """
        if 'power' not in df.columns:
            return None
        
        power_values = df['power'].dropna().tolist()
        if not power_values:
            return None
        
        return PowerData(
            power_values=power_values,
            estimated_power=[],  # Will be calculated later
            power_zones={}
        )
    
    def _extract_heart_rate_data(self, df: pd.DataFrame) -> Optional[HeartRateData]:
        """Extract heart rate data from DataFrame.
        
        Args:
            df: DataFrame with workout data
            
        Returns:
            HeartRateData object or None
        """
        if 'heart_rate' not in df.columns:
            return None
        
        hr_values = df['heart_rate'].dropna().tolist()
        if not hr_values:
            return None
        
        return HeartRateData(
            heart_rate_values=hr_values,
            hr_zones={},
            avg_hr=np.mean(hr_values),
            max_hr=np.max(hr_values)
        )
    
    def _extract_speed_data(self, df: pd.DataFrame) -> Optional[SpeedData]:
        """Extract speed data from DataFrame.
        
        Args:
            df: DataFrame with workout data
            
        Returns:
            SpeedData object or None
        """
        if 'speed' not in df.columns:
            return None
        
        speed_values = df['speed'].dropna().tolist()
        if not speed_values:
            return None
        
        # Convert m/s to km/h if needed
        if max(speed_values) < 50:  # Likely m/s
            speed_values = [s * 3.6 for s in speed_values]
        
        # Calculate distance if available
        distance_values = []
        if 'distance' in df.columns:
            distance_values = df['distance'].dropna().tolist()
            # Convert to km if in meters
            if distance_values and max(distance_values) > 1000:
                distance_values = [d / 1000 for d in distance_values]
        
        return SpeedData(
            speed_values=speed_values,
            distance_values=distance_values,
            avg_speed=np.mean(speed_values),
            max_speed=np.max(speed_values),
            total_distance=distance_values[-1] if distance_values else None
        )
    
    def _extract_elevation_data(self, df: pd.DataFrame) -> Optional[ElevationData]:
        """Extract elevation data from DataFrame.
        
        Args:
            df: DataFrame with workout data
            
        Returns:
            ElevationData object or None
        """
        if 'altitude' not in df.columns and 'elevation' not in df.columns:
            return None
        
        # Use 'altitude' or 'elevation' column
        elevation_col = 'altitude' if 'altitude' in df.columns else 'elevation'
        elevation_values = df[elevation_col].dropna().tolist()
        
        if not elevation_values:
            return None
        
        # Calculate gradients
        gradient_values = self._calculate_gradients(df)

        # Add gradient column to DataFrame
        df['gradient_percent'] = gradient_values

        return ElevationData(
            elevation_values=elevation_values,
            gradient_values=gradient_values,
            elevation_gain=max(elevation_values) - min(elevation_values),
            elevation_loss=0,  # Will be calculated more accurately
            max_gradient=np.max(gradient_values),
            min_gradient=np.min(gradient_values)
        )
    
    def _extract_gear_data(self, df: pd.DataFrame) -> Optional[GearData]:
        """Extract gear data from DataFrame.

        Args:
            df: DataFrame with workout data

        Returns:
            GearData object or None
        """
        if 'cadence_rpm' not in df.columns or 'speed_mps' not in df.columns:
            logger.info("Gear estimation skipped: missing speed_mps or cadence_rpm columns")
            return None

        # Estimate gear series
        gear_series = estimate_gear_series(
            df,
            wheel_circumference_m=BikeConfig.TIRE_CIRCUMFERENCE_M,
            valid_configurations=BikeConfig.VALID_CONFIGURATIONS
        )

        if gear_series.empty:
            logger.info("Gear estimation skipped: no valid data for estimation")
            return None

        # Compute summary
        summary = compute_gear_summary(gear_series)

        return GearData(
            series=gear_series,
            summary=summary
        )
    
    def _distance_window_indices(self, distance: np.ndarray, half_window_m: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute backward and forward indices for distance-based windowing.

        For each sample i, find the closest indices j <= i and k >= i such that
        distance[i] - distance[j] >= half_window_m and distance[k] - distance[i] >= half_window_m.

        Args:
            distance: Monotonic array of cumulative distances in meters
            half_window_m: Half window size in meters

        Returns:
            Tuple of (j_indices, k_indices) arrays
        """
        n = len(distance)
        j_indices = np.full(n, -1, dtype=int)
        k_indices = np.full(n, -1, dtype=int)

        for i in range(n):
            # Find largest j <= i where distance[i] - distance[j] >= half_window_m
            j = i
            while j >= 0 and distance[i] - distance[j] < half_window_m:
                j -= 1
            j_indices[i] = max(j, 0)

            # Find smallest k >= i where distance[k] - distance[i] >= half_window_m
            k = i
            while k < n and distance[k] - distance[i] < half_window_m:
                k += 1
            k_indices[i] = min(k, n - 1)

        return j_indices, k_indices

    def _calculate_gradients(self, df: pd.DataFrame) -> List[float]:
        """Calculate smoothed, distance-referenced gradients from elevation data.

        Computes gradients using a distance-based smoothing window, handling missing
        distance/speed/elevation data gracefully. Assumes 1 Hz sampling for distance
        derivation if speed is available but distance is not.

        Args:
            df: DataFrame containing elevation, distance, and speed columns

        Returns:
            List of gradient values in percent, with NaN for invalid computations
        """
        from config.settings import SMOOTHING_WINDOW

        n = len(df)
        if n < 2:
            return [np.nan] * n

        # Derive distance array
        if 'distance' in df.columns:
            distance = df['distance'].values.astype(float)
            if not np.all(distance[1:] >= distance[:-1]):
                logger.warning("Distance not monotonic, deriving from speed")
                distance = None  # Fall through to speed derivation
        else:
            distance = None

        if distance is None:
            if 'speed' in df.columns:
                speed = df['speed'].values.astype(float)
                distance = np.cumsum(speed)  # dt=1 assumed
            else:
                logger.warning("No distance or speed available, cannot compute gradients")
                return [np.nan] * n

        # Get elevation
        elevation_col = 'altitude' if 'altitude' in df.columns else 'elevation'
        elevation = df[elevation_col].values.astype(float)

        half_window = SMOOTHING_WINDOW / 2
        j_arr, k_arr = self._distance_window_indices(distance, half_window)

        gradients = []
        for i in range(n):
            j, k = j_arr[i], k_arr[i]
            if distance[k] - distance[j] >= 1 and not (pd.isna(elevation[j]) or pd.isna(elevation[k])):
                delta_elev = elevation[k] - elevation[j]
                delta_dist = distance[k] - distance[j]
                grad = 100 * delta_elev / delta_dist
                grad = np.clip(grad, -30, 30)
                gradients.append(grad)
            else:
                gradients.append(np.nan)

        # Light smoothing: rolling median over 5 samples, interpolate isolated NaNs
        grad_series = pd.Series(gradients)
        smoothed = grad_series.rolling(5, center=True, min_periods=1).median()
        smoothed = smoothed.interpolate(limit=3, limit_direction='both')

        return smoothed.tolist()
    
    def _parse_tcx(self, file_path: Path) -> Optional[WorkoutData]:
        """Parse TCX file format.
        
        Args:
            file_path: Path to TCX file
            
        Returns:
            WorkoutData object or None if parsing failed
        """
        raise NotImplementedError("TCX file parsing is not yet implemented.")
    
    def _parse_gpx(self, file_path: Path) -> Optional[WorkoutData]:
        """Parse GPX file format.
        
        Args:
            file_path: Path to GPX file
            
        Returns:
            WorkoutData object or None if parsing failed
        """
        raise NotImplementedError("GPX file parsing is not yet implemented.")