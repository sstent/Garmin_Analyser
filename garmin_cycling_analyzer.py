#!/usr/bin/env python3
"""
Enhanced Garmin Workout Analyzer
Downloads workouts from Garmin Connect and generates detailed markdown reports with charts.

Features:
- Download specific workouts by ID or latest cycling workout
- Enhanced power estimation with physics-based calculations
- Improved gear calculation using actual wheel specifications
- Generate comprehensive reports with minute-by-minute analysis
- Support for FIT, TCX, and GPX file formats
- Workout visualization charts
- Smoothed gradient calculation for better accuracy

Requirements:
pip install garminconnect fitparse python-dotenv pandas numpy matplotlib
"""

import os
import sys
import zipfile
import magic
from datetime import datetime, timedelta
from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional
import tempfile
import errno

# Required packages
try:
    from garminconnect import Garmin
    from fitparse import FitFile
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install garminconnect fitparse python-dotenv pandas numpy matplotlib")
    sys.exit(1)


class GarminWorkoutAnalyzer:
    """Main class for analyzing Garmin workout data."""
    
    def __init__(self, is_indoor=False):
        # Load environment variables
        load_dotenv()
        
        # Initialize magic file type detection
        self.magic = magic.Magic(mime=True)
        
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
        
        # Track last activity ID for filename
        self.last_activity_id = None
        
        # Bike specifications
        self.VALID_CONFIGURATIONS = {
            38: [14, 16, 18, 20],
            46: [16]
        }
        self.is_indoor = is_indoor
        self.selected_chainring = None
        self.power_data_available = False
        self.CHAINRING_TEETH = 38  # Default, will be updated
        self.BIKE_WEIGHT_LBS = 22
        self.BIKE_WEIGHT_KG = self.BIKE_WEIGHT_LBS * 0.453592
        
        # HR Zones (based on LTHR 170 bpm)
        self.HR_ZONES = {
            'Z1': (0, 136),
            'Z2': (136, 148),
            'Z3': (149, 158),
            'Z4': (159, 168),
            'Z5': (169, 300)
        }
        
        # Power Zones (in watts) for visualization
        self.POWER_ZONES = {
            'Recovery': (0, 150),
            'Endurance': (150, 200),
            'Tempo': (200, 250),
            'Threshold': (250, 300),
            'VO2 Max': (300, 1000)
        }
        
        # Colors for power zones
        self.POWER_ZONE_COLORS = {
            'Recovery': 'lightblue',
            'Endurance': 'green',
            'Tempo': 'yellow',
            'Threshold': 'orange',
            'VO2 Max': 'red'
        }

        # Wheel specifications for 700c + 46mm tires
        self.WHEEL_DIAMETER_MM = 700
        self.TIRE_WIDTH_MM = 46
        self.TIRE_CIRCUMFERENCE_MM = math.pi * (self.WHEEL_DIAMETER_MM + 2 * self.TIRE_WIDTH_MM)
        self.TIRE_CIRCUMFERENCE_M = self.TIRE_CIRCUMFERENCE_MM / 1000  # ~2.23m
        
        
    def connect_to_garmin(self) -> bool:
        """Connect to Garmin Connect using credentials from .env file."""
        username = os.getenv('GARMIN_USERNAME')
        password = os.getenv('GARMIN_PASSWORD')
        
        if not username or not password:
            print("Error: GARMIN_USERNAME and GARMIN_PASSWORD must be set in .env file")
            return False
            
        try:
            self.garmin_client = Garmin(username, password)
            self.garmin_client.login()
            print("Successfully connected to Garmin Connect")
            return True
        except Exception as e:
            print(f"Error connecting to Garmin: {e}")
            return False
    
    def download_specific_workout(self, activity_id: int) -> Optional[str]:
        """Download a specific workout by activity ID in FIT format."""
        try:
            print(f"Downloading workout ID: {activity_id}")
            self.last_activity_id = activity_id
            return self._download_workout(activity_id)
        except Exception as e:
            print(f"Error downloading workout {activity_id}: {e}")
            return None
            
    def download_latest_workout(self) -> Optional[str]:
        """Download the latest cycling workout in FIT format."""
        self.last_activity_id = None
        try:
            activities = self.garmin_client.get_activities(0, 20)
            
            print(f"Found {len(activities)} recent activities:")
            
            for i, activity in enumerate(activities[:10]):
                activity_type = activity.get('activityType', {})
                type_key = activity_type.get('typeKey', 'unknown')
                type_name = activity_type.get('typeId', 'unknown')
                activity_name = activity.get('activityName', 'Unnamed')
                start_time = activity.get('startTimeLocal', 'unknown')
                print(f"  {i+1}. {activity_name} - Type: {type_key} ({type_name}) - {start_time}")
            
            cycling_keywords = ['cycling', 'bike', 'road_biking', 'mountain_biking', 'indoor_cycling', 'biking']
            cycling_activity = None
            
            for activity in activities:
                activity_type = activity.get('activityType', {})
                type_key = activity_type.get('typeKey', '').lower()
                type_name = str(activity_type.get('typeId', '')).lower()
                activity_name = activity.get('activityName', '').lower()
                
                if any(keyword in type_key or keyword in type_name or keyword in activity_name 
                       for keyword in cycling_keywords):
                    cycling_activity = activity
                    print(f"Selected cycling activity: {activity['activityName']} (Type: {type_key})")
                    break
                    
            if not cycling_activity:
                print("No cycling activities found automatically.")
                print("Available activity types:")
                unique_types = set()
                for activity in activities[:10]:
                    type_key = activity.get('activityType', {}).get('typeKey', 'unknown')
                    unique_types.add(type_key)
                print(f"  {sorted(unique_types)}")
                
                choice = input("Enter 'y' to see all activities and select one, or 'n' to exit: ").strip().lower()
                
                if choice == 'y':
                    cycling_activity = self._manual_activity_selection(activities)
                    if not cycling_activity:
                        return None
                else:
                    return None
            
            activity_id = cycling_activity['activityId']
            self.last_activity_id = activity_id
            print(f"Found cycling activity: {cycling_activity['activityName']} ({activity_id})")
            return self._download_workout(activity_id)
            
        except Exception as e:
            print(f"Error downloading workout: {e}")
            return None

    def _download_workout(self, activity_id: int) -> Optional[str]:
        """Helper method to download a workout given an activity ID."""
        formats_to_try = [
            (self.garmin_client.ActivityDownloadFormat.ORIGINAL, '.fit'),
            (self.garmin_client.ActivityDownloadFormat.TCX, '.tcx'),
            (self.garmin_client.ActivityDownloadFormat.GPX, '.gpx')
        ]
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        for dl_format, extension in formats_to_try:
            try:
                print(f"Trying to download in {dl_format} format...")
                fit_data = self.garmin_client.download_activity(activity_id, dl_fmt=dl_format)
                
                if not fit_data or len(fit_data) == 0:
                    print(f"No data received for format {dl_format}")
                    continue
                
                print(f"Downloaded {len(fit_data)} bytes")
                
                # Save directly to data directory - use enum name for clean filename
                format_name = dl_format.name.lower()
                file_path = os.path.join("data", f"{activity_id}_{format_name}{extension}")
                with open(file_path, 'wb') as f:
                    f.write(fit_data)
                
                if dl_format == self.garmin_client.ActivityDownloadFormat.ORIGINAL and extension == '.fit':
                    # Validate real file type with python-magic
                    file_type = self.magic.from_file(file_path)
                    
                    if "application/zip" in file_type or zipfile.is_zipfile(file_path):
                        print("Extracting ZIP archive...")
                        extracted_files = []
                        temp_dir = tempfile.mkdtemp(dir="data")
                        
                        # Extract initial ZIP
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                            extracted_files = [os.path.join(temp_dir, name) for name in zip_ref.namelist()]
                        
                        # Recursive extraction
                        final_files = []
                        for extracted_path in extracted_files:
                            if zipfile.is_zipfile(extracted_path):
                                with zipfile.ZipFile(extracted_path, 'r') as nested_zip:
                                    nested_dir = os.path.join(temp_dir, "nested")
                                    os.makedirs(nested_dir, exist_ok=True)
                                    nested_zip.extractall(nested_dir)
                                    final_files.extend([os.path.join(nested_dir, name) for name in nested_zip.namelist()])
                            else:
                                final_files.append(extracted_path)
                        
                        # Find valid FIT files
                        fit_files = []
                        for file_path in final_files:
                            # Check by file header
                            try:
                                with open(file_path, 'rb') as f:
                                    if f.read(12).endswith(b'.FIT'):
                                        fit_files.append(file_path)
                            except:
                                continue
                        
                        if not fit_files:
                            print("No valid FIT file found after extraction")
                            continue
                        
                        # Use first valid FIT file
                        fit_file = fit_files[0]
                        print(f"Using FIT file: {fit_file}")
                    
                    elif is_fit_file_by_header(file_path):
                        print("Valid FIT file detected by header")
                        fit_file = file_path
                    
                    else:
                        print(f"Unexpected file format: {file_type}")
                        continue
                    
                    # Helper function for FIT header validation
                    def is_fit_file_by_header(path):
                        try:
                            with open(path, 'rb') as f:
                                header = f.read(12)
                                return header.endswith(b'.FIT')
                        except:
                            return False
                    
                    # Final validation with fitparse
                    try:
                        fit_obj = FitFile(fit_file)
                        list(fit_obj.get_messages())[:1]  # Test parse
                        print(f"Successfully validated FIT file: {fit_file}")
                        return fit_file
                    except Exception as fit_error:
                        print(f"FIT file validation failed: {fit_error}")
                        continue
                else:
                    print(f"Downloaded {dl_format} file to {file_path}")
                    return file_path
                    
            except Exception as format_error:
                print(f"Failed to download in {dl_format} format: {format_error}")
                continue
        
        print("Failed to download activity in any supported format")
        return None
    
    def download_all_workouts(self) -> None:
        """Download all cycling activities without analysis."""
        if not self.garmin_client:
            if not self.connect_to_garmin():
                return
        
        try:
            activities = self.garmin_client.get_activities(0, 1000)  # Get up to 1000 activities
            if not activities:
                print("No activities found")
                return
                
            cycling_keywords = ['cycling', 'bike', 'road_biking', 'mountain_biking', 'indoor_cycling', 'biking']
            cycling_activities = []
            
            for activity in activities:
                activity_type = activity.get('activityType', {})
                type_key = activity_type.get('typeKey', '').lower()
                type_name = str(activity_type.get('typeId', '')).lower()
                activity_name = activity.get('activityName', '').lower()
                
                if any(keyword in type_key or keyword in type_name or keyword in activity_name 
                       for keyword in cycling_keywords):
                    cycling_activities.append(activity)
            
            if not cycling_activities:
                print("No cycling activities found")
                return
                
            print(f"Found {len(cycling_activities)} cycling activities")
            os.makedirs("data", exist_ok=True)
            
            for activity in cycling_activities:
                activity_id = activity['activityId']
                activity_name = activity.get('activityName', 'Unnamed')
                print(f"\nDownloading activity: {activity_name} (ID: {activity_id})")
                
                # Check if already downloaded
                existing_files = [f for f in os.listdir("data") if str(activity_id) in f]
                if existing_files:
                    print(f"  Already exists: {existing_files[0]}")
                    continue
                
                self._download_workout(activity_id)
                
            print("\nAll cycling activities downloaded")
            
        except Exception as e:
            print(f"Error downloading workouts: {e}")
            return

    def _manual_activity_selection(self, activities: List[Dict]) -> Optional[Dict]:
        """Allow user to manually select an activity from the list."""
        print("\nRecent activities:")
        for i, activity in enumerate(activities[:15]):
            activity_type = activity.get('activityType', {})
            type_key = activity_type.get('typeKey', 'unknown')
            activity_name = activity.get('activityName', 'Unnamed')
            start_time = activity.get('startTimeLocal', 'unknown')
            distance = activity.get('distance', 0)
            distance_km = distance / 1000 if distance else 0
            
            print(f"  {i+1:2d}. {activity_name} ({type_key}) - {start_time} - {distance_km:.1f}km")
        
        while True:
            try:
                selection = input(f"\nSelect activity number (1-{min(15, len(activities))}), or 'q' to quit: ").strip()
                if selection.lower() == 'q':
                    return None
                
                index = int(selection) - 1
                if 0 <= index < min(15, len(activities)):
                    return activities[index]
                else:
                    print(f"Please enter a number between 1 and {min(15, len(activities))}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def estimate_gear_from_speed_cadence(self, speed_ms: float, cadence_rpm: float) -> float:
        """Calculate actual gear ratio from speed and cadence."""
        if cadence_rpm <= 0 or speed_ms <= 0:
            return 0
        
        # Distance per pedal revolution
        distance_per_rev = speed_ms * 60 / cadence_rpm
        
        # Gear ratio calculation
        gear_ratio = self.TIRE_CIRCUMFERENCE_M / distance_per_rev
        cog_teeth = self.CHAINRING_TEETH / gear_ratio
        
        return cog_teeth
    
    def enhanced_chainring_cog_estimation(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Estimate chainring and cog using actual data and valid configurations."""
        if df.empty or 'cadence' not in df.columns or 'speed' not in df.columns:
            return 38, 16  # Default fallback
        
        # For each valid configuration, calculate error
        config_errors = []
        
        for chainring, cogs in self.VALID_CONFIGURATIONS.items():
            for cog in cogs:
                error = 0
                count = 0
                
                for _, row in df.iterrows():
                    if pd.notna(row['cadence']) and pd.notna(row['speed']) and row['cadence'] > 30:
                        # Theoretical speed calculation
                        distance_per_rev = self.TIRE_CIRCUMFERENCE_M * (chainring / cog)
                        theoretical_speed = distance_per_rev * row['cadence'] * 60 / 1000  # km/h
                        
                        # Accumulate squared error
                        error += (theoretical_speed - row['speed'] * 3.6) ** 2
                        count += 1
                
                if count > 0:
                    avg_error = error / count
                    config_errors.append((chainring, cog, avg_error))
        
        # Find configuration with minimum error
        if config_errors:
            best_config = min(config_errors, key=lambda x: x[2])
            return best_config[0], best_config[1]
        
        return 38, 16  # Default if no valid estimation

    def enhanced_cog_estimation(self, df: pd.DataFrame) -> int:
        """Estimate cog size using speed and cadence data."""
        if df.empty or 'cadence' not in df.columns or 'speed' not in df.columns:
            return 16  # Default fallback
        
        gear_estimates = []
        valid_points = 0
        
        for _, row in df.iterrows():
            if (pd.notna(row['cadence']) and pd.notna(row['speed']) and 
                row['cadence'] > 60 and row['speed'] > 1.5):
                cog_estimate = self.estimate_gear_from_speed_cadence(row['speed'], row['cadence'])
                if 12 <= cog_estimate <= 22:
                    gear_estimates.append(cog_estimate)
                    valid_points += 1
        
        if valid_points > 10:  # Ensure sufficient valid data points
            avg_cog = np.mean(gear_estimates)
            return min(self.CASSETTE_OPTIONS, key=lambda x: abs(x - avg_cog))
        
        return 16  # Default if not enough data
    
    def estimate_cog_from_cadence(self, file_path: str) -> int:
        """Analyze workout file to estimate the most likely cog size based on cadence patterns."""
        try:
            if file_path.lower().endswith('.fit'):
                return self._analyze_fit_cadence(file_path)
            elif file_path.lower().endswith('.tcx'):
                return self._analyze_tcx_cadence(file_path)
            elif file_path.lower().endswith('.gpx'):
                print("GPX files don't contain cadence data, using default estimate")
                return 16
            else:
                return 16
        except Exception as e:
            print(f"Error analyzing cadence: {e}")
            return 16
    
    def _analyze_fit_cadence(self, fit_file_path: str) -> int:
        """Analyze FIT file for cadence patterns."""
        fit_file = FitFile(fit_file_path)
        
        cadence_values = []
        speed_values = []
        
        for record in fit_file.get_messages('record'):
            cadence = record.get_value('cadence')
            speed = record.get_value('enhanced_speed')
            
            if cadence and speed and cadence > 0 and speed > 0:
                cadence_values.append(cadence)
                speed_values.append(speed * 3.6)
        
        if not cadence_values:
            return 16
        
        avg_cadence = np.mean(cadence_values)
        avg_speed = np.mean(speed_values)
        
        if avg_cadence > 85:
            estimated_cog = 14
        elif avg_cadence > 75:
            estimated_cog = 16
        elif avg_cadence > 65:
            estimated_cog = 18
        else:
            estimated_cog = 20
            
        print(f"Analysis: Avg cadence {avg_cadence:.1f} RPM, Avg speed {avg_speed:.1f} km/h")
        return estimated_cog
    
    def _analyze_tcx_cadence(self, tcx_file_path: str) -> int:
        """Analyze TCX file for cadence patterns."""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(tcx_file_path)
            root = tree.getroot()
            
            ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
            
            cadence_values = []
            speed_values = []
            
            for trackpoint in root.findall('.//tcx:Trackpoint', ns):
                cadence_elem = trackpoint.find('tcx:Cadence', ns)
                speed_elem = trackpoint.find('tcx:Extensions//*[local-name()="Speed"]')
                
                if cadence_elem is not None and speed_elem is not None:
                    try:
                        cadence = int(cadence_elem.text)
                        speed = float(speed_elem.text)
                        
                        if cadence > 0 and speed > 0:
                            cadence_values.append(cadence)
                            speed_values.append(speed * 3.6)
                    except (ValueError, TypeError):
                        continue
            
            if not cadence_values:
                return 16
            
            avg_cadence = np.mean(cadence_values)
            avg_speed = np.mean(speed_values)
            
            if avg_cadence > 85:
                estimated_cog = 14
            elif avg_cadence > 75:
                estimated_cog = 16
            elif avg_cadence > 65:
                estimated_cog = 18
            else:
                estimated_cog = 20
                
            print(f"Analysis: Avg cadence {avg_cadence:.1f} RPM, Avg speed {avg_speed:.1f} km/h")
            return estimated_cog
            
        except Exception as e:
            print(f"Error parsing TCX file: {e}")
            return 16
    
    def get_user_cog_confirmation(self, estimated_cog: int) -> int:
        """Ask user to confirm the cog size."""
        print(f"\nBased on the workout data, estimated cog size: {estimated_cog}t")
        print("Available cog sizes: 14t, 16t, 18t, 20t")
        
        while True:
            try:
                user_input = input(f"Confirm cog size (press Enter for {estimated_cog}t, or enter 14/16/18/20): ").strip()
                
                if not user_input:
                    return estimated_cog
                
                cog = int(user_input)
                if cog in self.CASSETTE_OPTIONS:
                    return cog
                else:
                    print(f"Invalid cog size. Choose from: {self.CASSETTE_OPTIONS}")
            except ValueError:
                print("Please enter a valid number")
    
    def calculate_power(self, speed_ms: float, cadence: float, gradient: float, 
                       rider_weight_kg: float = 90.7, 
                       temperature_c: float = 20.0) -> float:
        """
        Calculate power using physics-based model. For indoor workouts, this estimates
        power based on cadence and resistance simulation.
        """
        # Handle None values to prevent comparison errors
        cadence = cadence if cadence is not None else 0
        speed_ms = speed_ms if speed_ms is not None else 0
        gradient = gradient if gradient is not None else 0
        temperature_c = temperature_c if temperature_c is not None else 20.0
        
        if self.power_data_available and cadence > 0:
            # Use real power data if available and valid
            return cadence  # This is just a placeholder
            
        if self.is_indoor:
            # Indoor-specific power estimation based on cadence and simulated resistance
            if cadence <= 0:
                return 0
                
            # Base resistance for stationary bike (equivalent to 2% grade)
            base_resistance = 0.02
            
            # Increase resistance effect at higher cadences
            resistance_factor = base_resistance * (1 + 0.01 * max(0, cadence - 80))
            
            # Calculate effective grade based on cadence
            simulated_grade = resistance_factor * 100
            
            # Simulate speed based on cadence and gear ratio (using fixed indoor gear)
            simulated_speed = cadence * (self.TIRE_CIRCUMFERENCE_M / 60) * 3.6
            
            # Apply the outdoor power model with simulated parameters
            return self._physics_power_model(
                simulated_speed / 3.6, 
                cadence, 
                simulated_grade, 
                temperature_c,
                rider_weight_kg
            )
            
        return self._physics_power_model(
            speed_ms, 
            cadence, 
            gradient, 
            temperature_c,
            rider_weight_kg
        )
    
    def _physics_power_model(self, 
                           speed_ms: float, 
                           cadence: float, 
                           gradient: float, 
                           temperature_c: float,
                           rider_weight_kg: float) -> float:
        """Physics-based power calculation model used for both indoor and outdoor."""
        if speed_ms <= 0:
            return 0
        
        # Temperature-adjusted air density
        rho = 1.225 * (288.15 / (temperature_c + 273.15))
        
        # Speed-dependent CdA (accounting for position changes)
        base_CdA = 0.324
        CdA = base_CdA * (1 + 0.02 * max(0, speed_ms - 10))
        
        # Rolling resistance varies with speed
        base_Cr = 0.0063
        Cr = base_Cr * (1 + 0.0001 * speed_ms**2)
        
        efficiency = 0.97
        total_weight = (rider_weight_kg + self.BIKE_WEIGHT_KG) * 9.81
        
        # Calculate the angle of the slope (in radians)
        slope_angle = math.atan(gradient / 100.0)
        
        # Force components
        F_rolling = Cr * total_weight * math.cos(slope_angle)
        F_air = 0.5 * CdA * rho * speed_ms**2
        F_gravity = total_weight * math.sin(slope_angle)
        
        # Mechanical losses
        mechanical_loss = 5 + 0.1 * speed_ms
        
        F_total = F_rolling + F_air + F_gravity
        power_watts = (F_total * speed_ms) / efficiency + mechanical_loss
        
        return max(power_watts, 0)
    
    def calculate_smoothed_gradient(self, df: pd.DataFrame, window_size: int = 5) -> pd.Series:
        """Calculate smoothed gradient with robust null safety."""
        gradients = []
        
        for i in range(len(df)):
            # Handle beginning of dataset
            if i < window_size:
                gradients.append(0.0)
                continue
                
            start_idx = i - window_size
            
            # Safe retrieval of values with defaults
            current_alt = df.iloc[i].get('altitude', 0)
            start_alt = df.iloc[start_idx].get('altitude', 0)
            current_dist = df.iloc[i].get('distance', 0)
            start_dist = df.iloc[start_idx].get('distance', 0)
            
            # Handle None and NaN values
            current_alt = 0 if current_alt is None or np.isnan(current_alt) else current_alt
            start_alt = 0 if start_alt is None or np.isnan(start_alt) else start_alt
            current_dist = 0 if current_dist is None or np.isnan(current_dist) else current_dist
            start_dist = 0 if start_dist is None or np.isnan(start_dist) else start_dist
            
            alt_diff = current_alt - start_alt
            dist_diff = current_dist - start_dist
            
            if dist_diff > 0:
                gradient = (alt_diff / dist_diff) * 100
                gradient = max(-20, min(20, gradient))  # Limit extreme gradients
                gradients.append(gradient)
            else:
                gradients.append(gradients[-1] if gradients else 0.0)
        
        return pd.Series(gradients)
    
    def analyze_fit_file(self, file_path: str, cog_size: int) -> Dict:
        """Analyze workout file and extract comprehensive workout data."""
        try:
            if file_path.lower().endswith('.fit'):
                return self._analyze_fit_format(file_path, cog_size)
            elif file_path.lower().endswith('.tcx'):
                return self._analyze_tcx_format(file_path, cog_size)
            elif file_path.lower().endswith('.gpx'):
                return self._analyze_gpx_format(file_path, cog_size)
            else:
                print(f"Unsupported file format: {file_path}")
                return None
        except Exception as e:
            print(f"Error analyzing workout file: {e}")
            return None
    
    def _analyze_fit_format(self, fit_file_path: str, cog_size: int) -> Dict:
        """Analyze FIT file format with robust missing value handling."""
        fit_file = FitFile(fit_file_path)
        
        records = []
        session_data = {}
        
        # Process session data with defaults
        for session in fit_file.get_messages('session'):
            session_data = {
                'start_time': session.get_value('start_time'),
                'total_elapsed_time': session.get_value('total_elapsed_time') or 0,
                'total_distance': session.get_value('total_distance') or 0,
                'total_calories': session.get_value('total_calories') or 0,
                'max_heart_rate': session.get_value('max_heart_rate') or 0,
                'avg_heart_rate': session.get_value('avg_heart_rate') or 0,
                'total_ascent': session.get_value('total_ascent') or 0,
                'total_descent': session.get_value('total_descent') or 0,
                'max_speed': session.get_value('max_speed') or 0,
                'avg_speed': session.get_value('avg_speed') or 0,
                'avg_cadence': session.get_value('avg_cadence') or 0,
            }
        
        # Process records with robust null handling
        for record in fit_file.get_messages('record'):
            # Handle potential missing values using get_value_safely
            record_data = {
                'timestamp': record.get_value('timestamp'),
                'heart_rate': self.get_value_safely(record, 'heart_rate'),
                'cadence': self.get_value_safely(record, 'cadence'),
                'speed': self.get_value_safely(record, 'enhanced_speed'),
                'distance': self.get_value_safely(record, 'distance'),
                'altitude': self.get_value_safely(record, 'enhanced_altitude'),
                'temperature': self.get_value_safely(record, 'temperature'),
            }
            records.append(record_data)
        
        # Create DataFrame and handle timestamps
        df = pd.DataFrame(records)
        
        # Drop records without timestamps
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
        
        # Fill other missing values with defaults
        for col in ['heart_rate', 'cadence', 'speed', 'distance', 'altitude', 'temperature']:
            if col in df.columns:
                df[col].fillna(0, inplace=True)
        
        return self._process_workout_data(df, session_data, cog_size)
    
    def get_value_safely(self, record, field_name, default=0):
        """Safely get value from FIT record field with error handling."""
        try:
            value = record.get_value(field_name)
            # Convert any NaNs or None to default
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            return value
        except (KeyError, ValueError, TypeError):
            return default
    
    def _analyze_tcx_format(self, tcx_file_path: str, cog_size: int) -> Dict:
        """Analyze TCX file format with robust namespace handling."""
        import xml.etree.ElementTree as ET
        import re
        
        try:
            tree = ET.parse(tcx_file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Error parsing TCX file: {e}")
            return None
            
        # Extract all namespaces from root element
        namespaces = {}
        for attr, value in root.attrib.items():
            if attr.startswith('xmlns'):
                # Extract prefix (or set as default)
                prefix = re.findall(r'\{?([^:]+)}?$', attr)[0] if ':' in attr else 'default'
                namespaces[prefix] = value
                
        # Create default namespace if missing
        if 'default' not in namespaces:
            namespaces['default'] = 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'
        
        # Create consistent namespace mapping
        ns_map = {
            'tcd': namespaces.get('default', 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'),
            'ae': namespaces.get('ns3', 'http://www.garmin.com/xmlschemas/ActivityExtension/v2')
        }
        
        records = []
        session_data = {}
        
        # Find activity using the default namespace prefix
        activity = root.find('.//tcd:Activity', ns_map)
        if activity is None:
            # Fallback to default namespace search
            activity = root.find('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Activity')
        if activity is None:
            # Final fallback to element name only
            activity = root.find('.//Activity')
            
        if activity is not None:
            total_time = 0
            total_distance = 0
            total_calories = 0
            max_hr = 0
            hr_values = []
            
            # Find laps with consistent namespace
            laps = activity.findall('tcd:Lap', ns_map)
            if not laps:
                laps = activity.findall('.//Lap')
                
            for lap in laps:
                # Use XPath-like syntax for deeper elements
                time_elem = lap.find('tcd:TotalTimeSeconds', ns_map)
                dist_elem = lap.find('tcd:DistanceMeters', ns_map)
                cal_elem = lap.find('tcd:Calories', ns_map)
                
                # Handle nested elements with namespace
                max_hr_elem = lap.find('tcd:MaximumHeartRateBpm/tcd:Value', ns_map) or lap.find('.//HeartRateBpm/Value')
                avg_hr_elem = lap.find('tcd:AverageHeartRateBpm/tcd:Value', ns_map) or lap.find('.//AverageHeartRateBpm/Value')
                
                if time_elem is not None and time_elem.text:
                    total_time += float(time_elem.text)
                if dist_elem is not None and dist_elem.text:
                    total_distance += float(dist_elem.text)
                if cal_elem is not None and cal_elem.text:
                    total_calories += int(cal_elem.text)
                if max_hr_elem is not None and max_hr_elem.text:
                    max_hr = max(max_hr, int(max_hr_elem.text))
                if avg_hr_elem is not None and avg_hr_elem.text:
                    hr_values.append(int(avg_hr_elem.text))
            
            session_data = {
                'start_time': None,
                'total_elapsed_time': total_time,
                'total_distance': total_distance,
                'total_calories': total_calories,
                'max_heart_rate': max_hr if max_hr > 0 else None,
                'avg_heart_rate': np.mean(hr_values) if hr_values else None,
                'total_ascent': None,
                'total_descent': None,
                'max_speed': None,
                'avg_speed': None,
                'avg_cadence': None,
            }
        
        # Find trackpoints using namespace or fallbacks
        trackpoints = root.findall('.//tcd:Trackpoint', ns_map)
        if not trackpoints:
            trackpoints = root.findall('.//{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}Trackpoint')
        if not trackpoints:
            trackpoints = root.findall('.//Trackpoint')
        if not trackpoints:
            # Fallback to recursive search
            trackpoints = [e for e in root.iter() if 'Trackpoint' in e.tag]
        
        for trackpoint in trackpoints:
            record_data = {
                'timestamp': None,
                'heart_rate': None,
                'cadence': None,
                'speed': None,
                'distance': None,
                'altitude': None,
                'temperature': None
            }
            
            # Handle timestamp
            time_elem = trackpoint.find('tcd:Time', ns_map) or trackpoint.find('.//Time')
            if time_elem is not None and time_elem.text:
                try:
                    record_data['timestamp'] = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                except:
                    pass
            
            # Handle heart rate with namespace fallbacks
            hr_elem = trackpoint.find('tcd:HeartRateBpm/tcd:Value', ns_map) or trackpoint.find('.//HeartRateBpm/Value')
            if hr_elem is not None and hr_elem.text:
                try:
                    record_data['heart_rate'] = int(hr_elem.text)
                except ValueError:
                    pass
            
            # Handle altitude
            alt_elem = trackpoint.find('tcd:AltitudeMeters', ns_map) or trackpoint.find('.//AltitudeMeters')
            if alt_elem is not None and alt_elem.text:
                try:
                    record_data['altitude'] = float(alt_elem.text)
                except ValueError:
                    pass
            
            # Handle distance
            dist_elem = trackpoint.find('tcd:DistanceMeters', ns_map) or trackpoint.find('.//DistanceMeters')
            if dist_elem is not None and dist_elem.text:
                try:
                    record_data['distance'] = float(dist_elem.text)
                except ValueError:
                    pass
            
            # Handle extensions (cadence and speed)
            extensions = trackpoint.find('tcd:Extensions', ns_map) or trackpoint.find('.//Extensions')
            if extensions is not None:
                cadence_elem = extensions.find('ae:Cadence', ns_map) or extensions.find('.//Cadence')
                speed_elem = extensions.find('ae:Speed', ns_map) or extensions.find('.//Speed')
                
                if cadence_elem is not None and cadence_elem.text:
                    try:
                        record_data['cadence'] = int(cadence_elem.text)
                    except ValueError:
                        pass
                if speed_elem is not None and speed_elem.text:
                    try:
                        record_data['speed'] = float(speed_elem.text)
                    except ValueError:
                        pass
            
            records.append(record_data)
        
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
        
        return self._process_workout_data(df, session_data, cog_size)
    
    def _find_element(self, element, tags, namespaces, is_path=False):
        """Helper to find element with multiple possible tags/namespaces."""
        for ns in namespaces:
            if is_path:
                current = element
                for tag in tags:
                    current = current.find(f'ns:{tag}', namespaces=ns) if ns else current.find(tag)
                    if current is None:
                        break
                if current is not None:
                    return current
            else:
                for tag in tags:
                    elem = element.find(f'ns:{tag}', namespaces=ns) if ns else element.find(tag)
                    if elem is not None:
                        return elem
        return None

    def _analyze_gpx_format(self, gpx_file_path: str, cog_size: int) -> Dict:
        """Analyze GPX file format (limited data available)."""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(gpx_file_path)
        root = tree.getroot()
        
        namespaces = [
            {'ns': 'http://www.topografix.com/GPX/1/1'},
            {'ns': ''}
        ]
        
        records = []
        
        # Find trackpoints
        trackpoints = []
        for ns in namespaces:
            trackpoints = root.findall('.//ns:trkpt', namespaces=ns)
            if trackpoints:
                break
        if not trackpoints:
            trackpoints = root.findall('.//trkpt')
        
        for trkpt in trackpoints:
            record_data = {
                'timestamp': None,
                'heart_rate': None,
                'cadence': None,
                'speed': None,
                'distance': None,
                'altitude': None,
                'temperature': None,
                'lat': None,
                'lon': None
            }
            
            try:
                record_data['lat'] = float(trkpt.get('lat'))
                record_data['lon'] = float(trkpt.get('lon'))
            except (ValueError, TypeError):
                pass
            
            ele_elem = trkpt.find('gpx:ele', ns)
            if ele_elem is not None:
                try:
                    record_data['altitude'] = float(ele_elem.text)
                except ValueError:
                    pass
            
            time_elem = trkpt.find('gpx:time', ns)
            if time_elem is not None:
                try:
                    record_data['timestamp'] = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                except:
                    pass
            
            records.append(record_data)
        
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
        
        session_data = {
            'start_time': df['timestamp'].iloc[0] if len(df) > 0 and 'timestamp' in df.columns else None,
            'total_elapsed_time': None,
            'total_distance': None,
            'total_calories': None,
            'max_heart_rate': None,
            'avg_heart_rate': None,
            'total_ascent': None,
            'total_descent': None,
            'max_speed': None,
            'avg_speed': None,
            'avg_cadence': None,
        }
        
        return self._process_workout_data(df, session_data, cog_size)
    
    def _process_workout_data(self, df: pd.DataFrame, session_data: Dict, cog_size: int) -> Dict:
        """
        Enhanced workout data processing with comprehensive validation:
        1. Validates essential columns
        2. Ensures proper data types
        3. Fills missing values with appropriate defaults
        4. Validates data ranges
        5. Handles outdoor vs indoor processing
        """
        # Initialize power_data_available flag
        self.power_data_available = False
        
        # Check if df has data
        if df is None or df.empty:
            print("Error: Workout data is empty")
            return None
        
        # Validate session_data exists
        if not session_data:
            session_data = {}
            print("Warning: Empty session data provided")
        
        # Validate and preprocess DataFrame columns
        required_columns = ['timestamp', 'cadence', 'speed', 'distance', 'altitude', 'temperature']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
                print(f"Warning: Missing column '{col}', filled with default 0")
        
        # Ensure numeric columns have proper types
        numeric_cols = ['cadence', 'speed', 'distance', 'altitude', 'temperature', 'heart_rate']
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric and fill NaNs (using assignment instead of inplace=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Validate data ranges
        df['cadence'] = df['cadence'].clip(lower=0, upper=200)
        df['speed'] = df['speed'].clip(lower=0, upper=100)
        df['altitude'] = df['altitude'].clip(lower=-100, upper=5000)
        df['temperature'] = df['temperature'].clip(lower=-20, upper=50)
        if 'heart_rate' in df.columns:
            df['heart_rate'] = df['heart_rate'].clip(lower=0, upper=250)
        
        # Check for real power data availability with validation
        if 'power' in df.columns:
            valid_power = df['power'].dropna()
            self.power_data_available = len(valid_power) > 10 and valid_power.mean() > 0
            if self.power_data_available:
                df['power'] = pd.to_numeric(df['power'], errors='coerce').fillna(0)
                df['power'] = df['power'].clip(lower=0, upper=2000)
        
        if len(df) > 0:
            if 'speed' in df.columns:
                df['speed_kmh'] = df['speed'] * 3.6
            else:
                df['speed_kmh'] = 0
                print("Warning: Speed data missing, created default speed_kmh=0")
            
            # Indoor-specific processing
            if self.is_indoor:
                # For indoor workouts, gradient calculation is simulated
                df['gradient'] = 0
                
                # Fixed gear configuration for indoor bike
                self.selected_chainring = 38
                cog_size = 16
                self.CHAINRING_TEETH = self.selected_chainring
                
                # Use physics model for indoor power estimation
                if not self.power_data_available:
                    if 'cadence' in df.columns and not df['cadence'].isna().all():
                        # For indoor, speed data might not be reliable - set to 0
                        df['power_estimate'] = df.apply(lambda row: 
                            self.calculate_power(0, row.get('cadence', 0), 0, row.get('temperature', 20)), 
                            axis=1
                        )
                    else:
                        df['power_estimate'] = 0
                        print("Warning: Cadence data missing for indoor power estimation")
                else:
                    # Use real power data when available
                    df['power_estimate'] = df['power']
            else:
                # Outdoor-specific processing
                if 'altitude' in df.columns and 'distance' in df.columns:
                    df['gradient'] = self.calculate_smoothed_gradient(df)
                else:
                    df['gradient'] = 0
                    print("Warning: Missing altitude/distance data for gradient calculation")
                
                # Enhanced cog and chainring estimation
                if 'cadence' in df.columns and 'speed_kmh' in df.columns:
                    estimated_chainring, estimated_cog = self.enhanced_chainring_cog_estimation(df)
                    self.selected_chainring = estimated_chainring
                    cog_size = estimated_cog
                    self.CHAINRING_TEETH = self.selected_chainring
                else:
                    print("Warning: Missing cadence/speed data for gear estimation")
                
                # Power estimation for outdoor workouts
                if 'speed' in df.columns and 'cadence' in df.columns and 'gradient' in df.columns:
                    df['power_estimate'] = df.apply(lambda row: 
                        self.calculate_power(
                            row.get('speed', 0), 
                            row.get('cadence', 0),
                            row.get('gradient', 0),
                            row.get('temperature', 20)
                        ), axis=1)
                else:
                    df['power_estimate'] = 0
                    print("Warning: Missing required data for outdoor power estimation")
            
            # Re-estimate cog size based on actual data for outdoor
            if not self.is_indoor and 'cadence' in df.columns and 'speed' in df.columns:
                estimated_cog = self.enhanced_cog_estimation(df)
                if estimated_cog != cog_size:
                    print(f"Data-based cog estimate: {estimated_cog}t (using {cog_size}t)")
        
        # Final data validation
        df = df.dropna(subset=['timestamp'], how='all')
        df = df.fillna(0)
        
        return {
            'session': session_data,
            'records': df,
            'cog_size': cog_size,
            'activity_id': self.last_activity_id
        }
    
    def calculate_hr_zones(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate time spent in each HR zone."""
        hr_zones_time = {zone: 0.0 for zone in self.HR_ZONES.keys()}
        
        if 'heart_rate' not in df.columns:
            return hr_zones_time
        
        total_records = len(df)
        time_per_record = 1.0  # seconds
        
        for _, record in df.iterrows():
            hr = record['heart_rate']
            if pd.isna(hr):
                continue
                
            for zone, (min_hr, max_hr) in self.HR_ZONES.items():
                if min_hr <= hr <= max_hr:
                    hr_zones_time[zone] += time_per_record
                    break
        
        # Convert to minutes
        for zone in hr_zones_time:
            hr_zones_time[zone] = hr_zones_time[zone] / 60.0
        
        return hr_zones_time
    
    def create_minute_by_minute_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Create minute-by-minute breakdown of the ride."""
        if len(df) == 0:
            return []
        
        minute_data = []
        start_time = df['timestamp'].iloc[0] if 'timestamp' in df.columns else None
        
        # Group data by minute
        for minute in range(int(len(df) / 60) + 1):
            start_idx = minute * 60
            end_idx = min((minute + 1) * 60, len(df))
            
            if start_idx >= len(df):
                break
                
            minute_df = df.iloc[start_idx:end_idx]
            
            if len(minute_df) == 0:
                continue
            
            # Calculate metrics for this minute
            minute_stats = {
                'minute': minute + 1,
                'distance_km': 0,
                'avg_speed_kmh': 0,
                'avg_cadence': 0,
                'avg_hr': 0,
                'max_hr': 0,
                'avg_gradient': 0,
                'elevation_change': 0,
                'avg_power_estimate': 0
            }
            
            # Distance cycled this minute
            if 'distance' in minute_df.columns and not minute_df['distance'].isna().all():
                distance_start = minute_df['distance'].iloc[0]
                distance_end = minute_df['distance'].iloc[-1]
                minute_stats['distance_km'] = (distance_end - distance_start) / 1000
            
            # Average metrics
            for col, stat_key in [
                ('speed_kmh', 'avg_speed_kmh'),
                ('cadence', 'avg_cadence'),
                ('heart_rate', 'avg_hr'),
                ('gradient', 'avg_gradient'),
                ('power_estimate', 'avg_power_estimate')
            ]:
                if col in minute_df.columns and not minute_df[col].isna().all():
                    minute_stats[stat_key] = minute_df[col].mean()
            
            # Max HR
            if 'heart_rate' in minute_df.columns and not minute_df['heart_rate'].isna().all():
                minute_stats['max_hr'] = minute_df['heart_rate'].max()
            
            # Elevation change
            if 'altitude' in minute_df.columns and not minute_df['altitude'].isna().all():
                alt_start = minute_df['altitude'].iloc[0]
                alt_end = minute_df['altitude'].iloc[-1]
                minute_stats['elevation_change'] = alt_end - alt_start
            
            minute_data.append(minute_stats)
        
        return minute_data
    
    def generate_workout_charts(self, analysis_data: Dict, output_dir: str = "."):
        """Generate workout visualization charts."""
        df = analysis_data['records']
        session = analysis_data['session']
        activity_id = analysis_data.get('activity_id', 'workout')
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if df.empty:
            print("No data available for chart generation")
            return
        
        # Prepare data
        df['distance_km'] = df['distance'] / 1000 if 'distance' in df.columns else range(len(df))
        
        # Set matplotlib style for better looking charts
        plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Cycling Workout Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Power + Elevation over Distance
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Plot elevation (background)
        if 'altitude' in df.columns and not df['altitude'].isna().all():
            ax1.fill_between(df['distance_km'], df['altitude'], alpha=0.3, color='gray', label='Elevation')
            ax1.plot(df['distance_km'], df['altitude'], color='gray', linewidth=1)
            ax1.set_ylabel('Elevation (m)', color='gray')
        else:
            ax1.set_ylabel('Distance (km)')
        
        # Power Visualization
        lines_to_show = []
        labels = []
        
        # Plot real power if available
        if self.power_data_available:
            real_power_line = ax1_twin.plot(df['distance_km'], df['power'], color='blue', linewidth=2, label='Real Power')
            lines_to_show.append(real_power_line[0])
            labels.append('Real Power')
        
        # Plot estimated power
        if 'power_estimate' in df.columns and not df['power_estimate'].isna().all():
            # Create power zone shading
            for power_zone, (low, high) in self.POWER_ZONES.items():
                ax1_twin.fill_between(df['distance_km'], low, high, 
                                     where=(df['power_estimate'] >= low) & (df['power_estimate'] <= high),
                                     color=self.POWER_ZONE_COLORS[power_zone], alpha=0.1)
            
            # Plot estimated power
            rolling_power = df['power_estimate'].rolling(window=15, min_periods=1).mean()
            power_line = ax1_twin.plot(df['distance_km'], df['power_estimate'], 
                                      color='red', linewidth=1, alpha=0.5, label='Estimated Power (Raw)')
            smooth_line = ax1_twin.plot(df['distance_km'], rolling_power, 
                                       color='darkred', linewidth=3, label='Estimated Power (Rolling Avg)')
            
            lines_to_show.extend([power_line[0], smooth_line[0]])
            labels.extend(['Estimated Power (Raw)', 'Estimated Power (Rolling Avg)'])
        
        # Add legend if we have lines to show
        if lines_to_show:
            ax1_twin.legend(lines_to_show, labels, loc='upper right', frameon=True, framealpha=0.8)
            ax1_twin.set_ylabel('Power (W)', color='black')
            ax1_twin.tick_params(axis='y', labelcolor='black')
            ax1_twin.spines['right'].set_color('black')
        
        ax1.set_xlabel('Distance (km)')
        ax1.set_title('Power and Elevation Profile')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='gray')
        
        # Chart 2: Temperature + HR + Elevation
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        ax2_triple = ax2.twinx()
        
        # Offset the third y-axis
        ax2_triple.spines['right'].set_position(('outward', 60))
        
        # Plot elevation (background)
        if 'altitude' in df.columns and not df['altitude'].isna().all():
            ax2.fill_between(df['distance_km'], df['altitude'], alpha=0.2, color='gray')
            ax2.plot(df['distance_km'], df['altitude'], color='gray', linewidth=1, alpha=0.7)
            ax2.set_ylabel('Elevation (m)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
        else:
            ax2.set_ylabel('Distance (km)')
        
        # Plot heart rate
        if 'heart_rate' in df.columns and not df['heart_rate'].isna().all():
            hr_line = ax2_twin.plot(df['distance_km'], df['heart_rate'], 
                                   color='blue', linewidth=2, label='Heart Rate')
            ax2_twin.set_ylabel('Heart Rate (bpm)', color='blue')
            ax2_twin.tick_params(axis='y', labelcolor='blue')
        
        # Plot temperature
        if 'temperature' in df.columns and not df['temperature'].isna().all():
            temp_line = ax2_triple.plot(df['distance_km'], df['temperature'], 
                                       color='orange', linewidth=2, label='Temperature')
            ax2_triple.set_ylabel('Temperature (°C)', color='orange')
            ax2_triple.tick_params(axis='y', labelcolor='orange')
        
        ax2.set_xlabel('Distance (km)')
        ax2.set_title('Heart Rate, Temperature, and Elevation Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"{output_dir}/{activity_id}_workout_charts.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        print(f"Charts saved to: {chart_filename}")
        plt.close()
        
        return chart_filename
    
    def reanalyze_all_workouts(self) -> None:
        """Re-analyze all downloaded activities and generate reports."""
        data_dir = Path("data")
        if not data_dir.exists():
            print("Data directory does not exist. Nothing to re-analyze.")
            return
        
        # Get all activity files in data directory
        activity_files = list(data_dir.glob('*.[fF][iI][tT]')) + \
                        list(data_dir.glob('*.[tT][cC][xX]')) + \
                        list(data_dir.glob('*.[gG][pP][xX]'))
        
        if not activity_files:
            print("No activity files found in data directory")
            return
            
        print(f"Found {len(activity_files)} activity files to analyze")
        
        for file_path in activity_files:
            try:
                # Extract activity ID from filename (filename format: {activity_id}_...)
                activity_id = None
                filename_parts = file_path.stem.split('_')
                if filename_parts and filename_parts[0].isdigit():
                    activity_id = int(filename_parts[0])
                
                print(f"\nAnalyzing: {file_path.name}")
                print("------------------------------------------------")
                
                # Estimate cog size
                estimated_cog = self.estimate_cog_from_cadence(str(file_path))
                print(f"Estimated cog size from file: {estimated_cog}t")
                
                # Run analysis
                analysis_data = self.analyze_fit_file(str(file_path), estimated_cog)
                if not analysis_data:
                    print(f"Failed to analyze file: {file_path.name}")
                    continue
                
                # Generate report (use activity ID if available, else use filename)
                report_id = activity_id if activity_id else file_path.stem
                self.generate_markdown_report(analysis_data, activity_id=report_id)
                print(f"Generated report for activity {report_id}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\nAll activities re-analyzed")

    def generate_markdown_report(self, analysis_data: Dict, activity_id: int = None, output_file: str = None):
        """Generate comprehensive markdown report with enhanced power analysis."""
        session = analysis_data['session']
        df = analysis_data['records']
        cog_size = analysis_data['cog_size']
        chainring = self.selected_chainring or 38
        
        # Create report directory structure
        report_dir = Path("reports")
        
        # Add indoor bike indicator to report
        indoor_indicator = " (Indoor Bike)" if self.is_indoor else ""
        if self.is_indoor and not self.power_data_available:
            power_source = "Estimated Power (Physics Model)"
        else:
            power_source = "Real Power Data" if self.power_data_available else "Estimated Power"
        if activity_id and session.get('start_time'):
            date_str = session['start_time'].strftime('%Y-%m-%d')
            report_dir = report_dir / f"{date_str}_{activity_id}"
            report_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(report_dir / f"{activity_id}_workout_analysis.md")
        else:
            output_file = 'workout_report.md'
            report_dir = Path(".")
            
        # Generate charts in the report directory
        chart_filename = self.generate_workout_charts(analysis_data, output_dir=str(report_dir))
        
        # Calculate additional metrics
        hr_zones = self.calculate_hr_zones(df)
        minute_analysis = self.create_minute_by_minute_analysis(df)
        
        # Chart generation moved to beginning of method
        
        # Generate report
        report = []
        report.append("# Cycling Workout Analysis Report")
        report.append(f"\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append(f"\n**Bike Configuration{indoor_indicator}:** {chainring}t chainring, {cog_size}t cog, {self.BIKE_WEIGHT_LBS}lbs bike weight")
        report.append(f"**Power Source:** {power_source}")
        report.append(f"**Wheel Specs:** 700c wheel + {self.TIRE_WIDTH_MM}mm tires (circumference: {self.TIRE_CIRCUMFERENCE_M:.2f}m)\n")
        
        # Basic metrics table
        report.append("## Basic Workout Metrics")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        
        # Format session data
        if session.get('start_time'):
            report.append(f"| Date | {session['start_time'].strftime('%Y-%m-%d %H:%M:%S')} |")
        
        if session.get('total_elapsed_time'):
            total_time = str(timedelta(seconds=session['total_elapsed_time']))
            report.append(f"| Total Time | {total_time} |")
        
        if session.get('total_distance'):
            distance_km = session['total_distance'] / 1000
            report.append(f"| Distance | {distance_km:.2f} km |")
        
        if session.get('total_ascent'):
            report.append(f"| Elevation Gain | {session['total_ascent']:.0f} m |")
        
        if session.get('avg_heart_rate'):
            report.append(f"| Average HR | {session['avg_heart_rate']:.0f} bpm |")
        
        if session.get('max_heart_rate'):
            report.append(f"| Max HR | {session['max_heart_rate']:.0f} bpm |")
        
        if session.get('avg_speed'):
            avg_speed_kmh = session['avg_speed'] * 3.6
            report.append(f"| Average Speed | {avg_speed_kmh:.1f} km/h |")
        
        if session.get('max_speed'):
            max_speed_kmh = session['max_speed'] * 3.6
            report.append(f"| Max Speed | {max_speed_kmh:.1f} km/h |")
        
        if session.get('avg_cadence'):
            report.append(f"| Average Cadence | {session['avg_cadence']:.0f} rpm |")
        
        # Power Metrics - prioritize real power data if available
        if self.power_data_available:
            # Real power data
            power_data = df['power']
            avg_power = power_data.mean()
            max_power = power_data.max()
            power_95th = np.percentile(power_data, 95)
            power_75th = np.percentile(power_data, 75)
            
            report.append(f"| **Real Avg Power** | **{avg_power:.0f} W** |")
            report.append(f"| **Real Max Power** | **{max_power:.0f} W** |")
            report.append(f"| Real 95th Percentile | {power_95th:.0f} W |")
            report.append(f"| Real 75th Percentile | {power_75th:.0f} W |")
        elif not df.empty and 'power_estimate' in df.columns:
            # Estimated power
            power_data = df[df['power_estimate'] > 0]['power_estimate']
            if len(power_data) > 0:
                avg_power = power_data.mean()
                max_power = power_data.max()
                power_95th = np.percentile(power_data, 95)
                power_75th = np.percentile(power_data, 75)
                
                report.append(f"| **Estimated Avg Power** | **{avg_power:.0f} W** |")
                report.append(f"| **Estimated Max Power** | **{max_power:.0f} W** |")
                report.append(f"| Estimated 95th Percentile | {power_95th:.0f} W |")
                report.append(f"| Estimated 75th Percentile | {power_75th:.0f} W |")
        
        # Temperature
        if not df.empty and 'temperature' in df.columns and not df['temperature'].isna().all():
            min_temp = df['temperature'].min()
            max_temp = df['temperature'].max()
            avg_temp = df['temperature'].mean()
            report.append(f"| Temperature Range | {min_temp:.0f}°C - {max_temp:.0f}°C (avg {avg_temp:.0f}°C) |")
        
        if session.get('total_calories'):
            report.append(f"| Calories | {session['total_calories']:.0f} cal |")
        
        # HR Zones table
        report.append("\n## Heart Rate Zones")
        report.append("*Based on LTHR 170 bpm*")
        report.append("\n| Zone | Range (bpm) | Time (min) | Percentage |")
        report.append("|------|-------------|------------|------------|")
        
        total_time_min = sum(hr_zones.values())
        for zone, (min_hr, max_hr) in self.HR_ZONES.items():
            time_min = hr_zones[zone]
            percentage = (time_min / total_time_min * 100) if total_time_min > 0 else 0
            range_str = f"{min_hr}-{max_hr}" if max_hr < 300 else f"{min_hr}+"
            report.append(f"| {zone} | {range_str} | {time_min:.1f} | {percentage:.1f}% |")
        
        # Power Distribution
        if not df.empty and 'power_estimate' in df.columns:
            power_data = df[df['power_estimate'] > 0]['power_estimate']
            if len(power_data) > 0:
                power_type = "Real" if self.power_data_available else "Estimated"
                report.append(f"\n## {power_type} Power Distribution")
                power_zones = {
                    'Recovery (<150W)': len(power_data[power_data < 150]) / len(power_data) * 100,
                    'Endurance (150-200W)': len(power_data[(power_data >= 150) & (power_data < 200)]) / len(power_data) * 100,
                    'Tempo (200-250W)': len(power_data[(power_data >= 200) & (power_data < 250)]) / len(power_data) * 100,
                    'Threshold (250-300W)': len(power_data[(power_data >= 250) & (power_data < 300)]) / len(power_data) * 100,
                    'VO2 Max (>300W)': len(power_data[power_data >= 300]) / len(power_data) * 100
                }
                
                report.append("| Power Zone | Percentage | Time (min) |")
                report.append("|------------|------------|------------|")
                for zone, percentage in power_zones.items():
                    time_in_zone = (percentage / 100) * (len(power_data) / 60)  # Convert to minutes
                    report.append(f"| {zone} | {percentage:.1f}% | {time_in_zone:.1f} |")
        
        # Charts section - add at the bottom as requested
        if chart_filename:
            report.append(f"\n## Workout Analysis Charts")
            report.append(f"Detailed charts showing power output, heart rate, and elevation profiles:")
            report.append(f"![Workout Analysis Charts]({os.path.basename(chart_filename)})")
        
        # Minute-by-minute analysis
        if minute_analysis:
            report.append("\n## Minute-by-Minute Analysis")
            
            if self.power_data_available:
                # Show both power columns when real data is available
                report.append("| Min | Dist (km) | Avg Speed (km/h) | Avg Cadence | Avg HR | Max HR | Avg Gradient (%) | Elevation Δ (m) | Real Avg Power (W) | Est Avg Power (W) |")
                report.append("|-----|-----------|------------------|-------------|--------|--------|------------------|-----------------|-------------------|-------------------|")
                
                for minute_data in minute_analysis:
                    report.append(
                        f"| {minute_data['minute']:2d} | "
                        f"{minute_data['distance_km']:.2f} | "
                        f"{minute_data['avg_speed_kmh']:.1f} | "
                        f"{minute_data['avg_cadence']:.0f} | "
                        f"{minute_data['avg_hr']:.0f} | "
                        f"{minute_data['max_hr']:.0f} | "
                        f"{minute_data['avg_gradient']:.1f} | "
                        f"{minute_data['elevation_change']:.1f} | "
                        f"{(minute_data.get('avg_real_power') or 0):.0f} | "
                        f"{minute_data['avg_power_estimate']:.0f} |"
                    )
            else:
                # Only show estimated power column when no real data
                report.append("| Min | Dist (km) | Avg Speed (km/h) | Avg Cadence | Avg HR | Max HR | Avg Gradient (%) | Elevation Δ (m) | Est Avg Power (W) |")
                report.append("|-----|-----------|------------------|-------------|--------|--------|------------------|-----------------|-------------------|")
                
                for minute_data in minute_analysis:
                    report.append(
                        f"| {minute_data['minute']:2d} | "
                        f"{minute_data['distance_km']:.2f} | "
                        f"{minute_data['avg_speed_kmh']:.1f} | "
                        f"{minute_data['avg_cadence']:.0f} | "
                        f"{minute_data['avg_hr']:.0f} | "
                        f"{minute_data['max_hr']:.0f} | "
                        f"{minute_data['avg_gradient']:.1f} | "
                        f"{minute_data['elevation_change']:.1f} | "
                        f"{minute_data['avg_power_estimate']:.0f} |"
                    )
        
        # Technical Notes
        report.append("\n## Technical Notes")
        if self.is_indoor and not self.power_data_available:
            report.append("- **INDOOR POWER ESTIMATION:** Uses physics-based model simulating 2% base grade ")
            report.append("  with increasing resistance at higher cadences (>80 RPM)")
        else:
            if self.power_data_available:
                report.append("- Power metrics use direct power meter measurements")
            else:
                report.append("- Power metrics use enhanced physics model with temperature-adjusted air density")
        report.append("- Gradient calculations are smoothed over 5-point windows to reduce GPS noise")
        report.append("- Gear ratios calculated using actual wheel circumference and drive train specifications")
        report.append("- Power zones based on typical cycling power distribution ranges")
        
        # Write report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Report generated: {output_file}")
        return output_file


import argparse

def main():
    """Main function to run the workout analyzer."""
    parser = argparse.ArgumentParser(
        description='Garmin Cycling Analyzer - Download and analyze workouts with enhanced power estimation',
        epilog=(
            'Examples:\n'
            '  Download & analyze latest workout: python garmin_cycling_analyzer.py\n'
            '  Analyze specific workout: python garmin_cycling_analyzer.py -w 123456789\n'
            '  Download all cycling workouts: python garmin_cycling_analyzer.py --download-all\n'
            '  Re-analyze downloaded workouts: python garmin_cycling_analyzer.py --reanalyze-all\n\n'
            'After downloading workouts, find files in data/ directory\n'
            'Generated reports are saved in reports/ directory'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-w', '--workout-id', type=int, help='Analyze specific workout by ID')
    parser.add_argument('--indoor', action='store_true', help='Process as indoor cycling workout')
    parser.add_argument('--download-all', action='store_true', help='Download all cycling activities (no analysis)')
    parser.add_argument('--reanalyze-all', action='store_true', help='Re-analyze all downloaded activities')
    args = parser.parse_args()
    
    analyzer = GarminWorkoutAnalyzer(is_indoor=args.indoor)
    
    # Step 1: Connect to Garmin
    if not analyzer.connect_to_garmin():
        return
    
    # Process command line arguments
    if args.download_all:
        print("Downloading all cycling workouts...")
        analyzer.download_all_workouts()
        print("\nAll downloads completed!")
    elif args.reanalyze_all:
        print("Re-analyzing all downloaded workouts...")
        analyzer.reanalyze_all_workouts()
    elif args.workout_id:
        activity_id = args.workout_id
        print(f"Processing workout ID: {activity_id}")
        fit_file_path = analyzer.download_specific_workout(activity_id)
        
        if not fit_file_path:
            print(f"Failed to download workout {activity_id}")
            return
            
        estimated_cog = analyzer.estimate_cog_from_cadence(fit_file_path)
        confirmed_cog = analyzer.get_user_cog_confirmation(estimated_cog)
        print("Analyzing workout with enhanced power calculations...")
        analysis_data = analyzer.analyze_fit_file(fit_file_path, confirmed_cog)
        
        if not analysis_data:
            print("Error: Could not analyze workout data")
            return
            
        print("Generating comprehensive report...")
        report_file = analyzer.generate_markdown_report(analysis_data, activity_id=activity_id)
        
        print(f"\nAnalysis complete for workout {activity_id}!")
        print(f"Report saved: {report_file}")
    else:
        print("Processing latest cycling workout")
        fit_file_path = analyzer.download_latest_workout()
        activity_id = analyzer.last_activity_id

        if not fit_file_path:
            print("Failed to download latest workout")
            return
        
        estimated_cog = analyzer.estimate_cog_from_cadence(fit_file_path)
        confirmed_cog = analyzer.get_user_cog_confirmation(estimated_cog)
        print("Analyzing with enhanced power model...")
        analysis_data = analyzer.analyze_fit_file(fit_file_path, confirmed_cog)
        
        if not analysis_data:
            print("Error: Could not analyze workout data")
            return
        
        print("Generating report with visualization...")
        report_file = analyzer.generate_markdown_report(analysis_data, activity_id=activity_id)
        
        print(f"\nAnalysis complete for activity {activity_id}!")
        print(f"Report saved: {report_file}")

if __name__ == "__main__":
    # Create example .env file if it doesn't exist
    env_file = Path('.env')
    if not env_file.exists():
        with open('.env', 'w') as f:
            f.write("# Garmin Connect Credentials\n")
            f.write("GARMIN_USERNAME=your_username_here\n")
            f.write("GARMIN_PASSWORD=your_password_here\n")
        print("Created .env file template. Please add your Garmin credentials.")
        sys.exit(1)
    
    main()
