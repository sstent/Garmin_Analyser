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
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Track last activity ID for filename
        self.last_activity_id = None
        
        # Bike specifications
        self.CHAINRING_TEETH = 38
        self.BIKE_WEIGHT_LBS = 22
        self.BIKE_WEIGHT_KG = self.BIKE_WEIGHT_LBS * 0.453592
        
        # Wheel specifications for 700c + 46mm tires
        self.WHEEL_DIAMETER_MM = 700
        self.TIRE_WIDTH_MM = 46
        self.TIRE_CIRCUMFERENCE_MM = math.pi * (self.WHEEL_DIAMETER_MM + 2 * self.TIRE_WIDTH_MM)
        self.TIRE_CIRCUMFERENCE_M = self.TIRE_CIRCUMFERENCE_MM / 1000  # ~2.23m
        
        # HR Zones (based on LTHR 170 bpm)
        self.HR_ZONES = {
            'Z1': (0, 136),
            'Z2': (136, 148),
            'Z3': (149, 158),
            'Z4': (159, 168),
            'Z5': (169, 300)
        }
        
        # Cassette options
        self.CASSETTE_OPTIONS = [14, 16, 18, 20]
        
        self.garmin_client = None
        
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
                
                # Save directly to data directory
                file_path = os.path.join("data", f"{activity_id}_{dl_format}{extension}")
                with open(file_path, 'wb') as f:
                    f.write(fit_data)
                
                if dl_format == self.garmin_client.ActivityDownloadFormat.ORIGINAL and extension == '.fit':
                    if zipfile.is_zipfile(file_path):
                        print(f"Downloaded file is a ZIP archive, extracting FIT file...")
                        
                        # Extract to same data directory
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall("data")
                        
                        fit_files = list(Path("data").glob('*.fit'))
                        if not fit_files:
                            print("No FIT file found in ZIP archive")
                            continue
                        
                        extracted_fit = fit_files[0]
                        print(f"Extracted FIT file: {extracted_fit}")
                        file_path = str(extracted_fit)
                    
                    try:
                        test_fit = FitFile(file_path)
                        list(test_fit.get_messages())[:1]
                        print(f"Downloaded valid FIT file to {file_path}")
                        return file_path
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
    
    def enhanced_cog_estimation(self, df: pd.DataFrame) -> int:
        """Enhanced cog estimation using actual gear calculations."""
        if df.empty or 'cadence' not in df.columns or 'speed' not in df.columns:
            return 16
        
        gear_estimates = []
        
        for _, row in df.iterrows():
            if (pd.notna(row['cadence']) and pd.notna(row['speed']) and 
                row['cadence'] > 60 and row['speed'] > 1.5):
                
                cog_estimate = self.estimate_gear_from_speed_cadence(row['speed'], row['cadence'])
                if 12 <= cog_estimate <= 22:
                    gear_estimates.append(cog_estimate)
        
        if gear_estimates:
            avg_cog = np.mean(gear_estimates)
            return min(self.CASSETTE_OPTIONS, key=lambda x: abs(x - avg_cog))
        
        return 16
    
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
    
    def calculate_enhanced_power_estimate(self, speed_ms: float, gradient: float, 
                                        rider_weight_kg: float = 90.7, 
                                        temperature_c: float = 20.0) -> float:
        """Enhanced power estimation with better physics and environmental factors."""
        if speed_ms <= 0:
            return 0
        
        speed_ms = max(speed_ms, 0.1)
        
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
        
        # Force components
        F_rolling = Cr * total_weight * math.cos(math.atan(gradient / 100))
        F_air = 0.5 * CdA * rho * speed_ms**2
        F_gravity = total_weight * math.sin(math.atan(gradient / 100))
        
        # Mechanical losses
        mechanical_loss = 5 + 0.1 * speed_ms
        
        F_total = F_rolling + F_air + F_gravity
        power_watts = (F_total * speed_ms) / efficiency + mechanical_loss
        
        return max(power_watts, 0)
    
    def calculate_smoothed_gradient(self, df: pd.DataFrame, window_size: int = 5) -> pd.Series:
        """Calculate smoothed gradient to reduce noise."""
        gradients = []
        
        for i in range(len(df)):
            if i < window_size:
                gradients.append(0.0)
                continue
                
            start_idx = i - window_size
            
            if (pd.notna(df.iloc[i]['altitude']) and pd.notna(df.iloc[start_idx]['altitude']) and
                pd.notna(df.iloc[i]['distance']) and pd.notna(df.iloc[start_idx]['distance'])):
                
                alt_diff = df.iloc[i]['altitude'] - df.iloc[start_idx]['altitude']
                dist_diff = df.iloc[i]['distance'] - df.iloc[start_idx]['distance']
                
                if dist_diff > 0:
                    gradient = (alt_diff / dist_diff) * 100
                    gradient = max(-20, min(20, gradient))  # Limit extreme gradients
                    gradients.append(gradient)
                else:
                    gradients.append(gradients[-1] if gradients else 0.0)
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
        """Analyze FIT file format."""
        fit_file = FitFile(fit_file_path)
        
        records = []
        session_data = {}
        
        for session in fit_file.get_messages('session'):
            session_data = {
                'start_time': session.get_value('start_time'),
                'total_elapsed_time': session.get_value('total_elapsed_time'),
                'total_distance': session.get_value('total_distance'),
                'total_calories': session.get_value('total_calories'),
                'max_heart_rate': session.get_value('max_heart_rate'),
                'avg_heart_rate': session.get_value('avg_heart_rate'),
                'total_ascent': session.get_value('total_ascent'),
                'total_descent': session.get_value('total_descent'),
                'max_speed': session.get_value('max_speed'),
                'avg_speed': session.get_value('avg_speed'),
                'avg_cadence': session.get_value('avg_cadence'),
            }
        
        for record in fit_file.get_messages('record'):
            record_data = {
                'timestamp': record.get_value('timestamp'),
                'heart_rate': record.get_value('heart_rate'),
                'cadence': record.get_value('cadence'),
                'speed': record.get_value('enhanced_speed'),
                'distance': record.get_value('distance'),
                'altitude': record.get_value('enhanced_altitude'),
                'temperature': record.get_value('temperature'),
            }
            records.append(record_data)
        
        df = pd.DataFrame(records)
        df = df.dropna(subset=['timestamp'])
        
        return self._process_workout_data(df, session_data, cog_size)
    
    def _analyze_tcx_format(self, tcx_file_path: str, cog_size: int) -> Dict:
        """Analyze TCX file format."""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(tcx_file_path)
        root = tree.getroot()
        
        ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
        
        records = []
        session_data = {}
        
        activity = root.find('.//tcx:Activity', ns)
        if activity is not None:
            total_time = 0
            total_distance = 0
            total_calories = 0
            max_hr = 0
            hr_values = []
            
            for lap in activity.findall('tcx:Lap', ns):
                time_elem = lap.find('tcx:TotalTimeSeconds', ns)
                dist_elem = lap.find('tcx:DistanceMeters', ns)
                cal_elem = lap.find('tcx:Calories', ns)
                max_hr_elem = lap.find('tcx:MaximumHeartRateBpm/tcx:Value', ns)
                avg_hr_elem = lap.find('tcx:AverageHeartRateBpm/tcx:Value', ns)
                
                if time_elem is not None:
                    total_time += float(time_elem.text)
                if dist_elem is not None:
                    total_distance += float(dist_elem.text)
                if cal_elem is not None:
                    total_calories += int(cal_elem.text)
                if max_hr_elem is not None:
                    max_hr = max(max_hr, int(max_hr_elem.text))
                if avg_hr_elem is not None:
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
        
        for trackpoint in root.findall('.//tcx:Trackpoint', ns):
            record_data = {'timestamp': None, 'heart_rate': None, 'cadence': None, 
                          'speed': None, 'distance': None, 'altitude': None, 'temperature': None}
            
            time_elem = trackpoint.find('tcx:Time', ns)
            if time_elem is not None:
                try:
                    record_data['timestamp'] = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                except:
                    pass
            
            hr_elem = trackpoint.find('tcx:HeartRateBpm/tcx:Value', ns)
            if hr_elem is not None:
                try:
                    record_data['heart_rate'] = int(hr_elem.text)
                except ValueError:
                    pass
            
            alt_elem = trackpoint.find('tcx:AltitudeMeters', ns)
            if alt_elem is not None:
                try:
                    record_data['altitude'] = float(alt_elem.text)
                except ValueError:
                    pass
            
            dist_elem = trackpoint.find('tcx:DistanceMeters', ns)
            if dist_elem is not None:
                try:
                    record_data['distance'] = float(dist_elem.text)
                except ValueError:
                    pass
            
            extensions = trackpoint.find('tcx:Extensions', ns)
            if extensions is not None:
                cadence_elem = extensions.find('.//*[local-name()="Cadence"]')
                if cadence_elem is not None:
                    try:
                        record_data['cadence'] = int(cadence_elem.text)
                    except ValueError:
                        pass
                
                speed_elem = extensions.find('.//*[local-name()="Speed"]')
                if speed_elem is not None:
                    try:
                        record_data['speed'] = float(speed_elem.text)
                    except ValueError:
                        pass
            
            records.append(record_data)
        
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df = df.dropna(subset=['timestamp'])
        
        return self._process_workout_data(df, session_data, cog_size)
    
    def _analyze_gpx_format(self, gpx_file_path: str, cog_size: int) -> Dict:
        """Analyze GPX file format (limited data available)."""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(gpx_file_path)
        root = tree.getroot()
        
        ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
        
        records = []
        
        for trkpt in root.findall('.//gpx:trkpt', ns):
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
        """Enhanced workout data processing with improved calculations."""
        if len(df) > 0:
            if 'speed' in df.columns:
                df['speed_kmh'] = df['speed'] * 3.6
            else:
                df['speed_kmh'] = 0
                
            # Enhanced gradient calculation
            df['gradient'] = self.calculate_smoothed_gradient(df)
            
            # Enhanced power calculation
            df['power_estimate'] = df.apply(lambda row: 
                self.calculate_enhanced_power_estimate(
                    row.get('speed', 0), 
                    row.get('gradient', 0),
                    temperature_c=row.get('temperature', 20)
                ), axis=1)
            
            # Re-estimate cog size based on actual data
            estimated_cog = self.enhanced_cog_estimation(df)
            if estimated_cog != cog_size:
                print(f"Data-based cog estimate: {estimated_cog}t (using {cog_size}t)")
        
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
        
        # Plot power
        if 'power_estimate' in df.columns and not df['power_estimate'].isna().all():
            power_line = ax1_twin.plot(df['distance_km'], df['power_estimate'], 
                                      color='red', linewidth=2, label='Power (W)')
            ax1_twin.set_ylabel('Power (W)', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
        
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
    
    def generate_markdown_report(self, analysis_data: Dict, activity_id: int = None, output_file: str = None):
        """Generate comprehensive markdown report with enhanced power analysis."""
        session = analysis_data['session']
        df = analysis_data['records']
        cog_size = analysis_data['cog_size']
        
        # Create report directory structure
        report_dir = Path("reports")
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
        report.append(f"\n**Bike Configuration:** {self.CHAINRING_TEETH}t chainring, {cog_size}t cog, {self.BIKE_WEIGHT_LBS}lbs bike weight")
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
        
        # Enhanced Power estimates
        if not df.empty and 'power_estimate' in df.columns:
            power_data = df[df['power_estimate'] > 0]['power_estimate']
            if len(power_data) > 0:
                avg_power = power_data.mean()
                max_power = power_data.max()
                power_95th = np.percentile(power_data, 95)
                power_75th = np.percentile(power_data, 75)
                
                report.append(f"| **Enhanced Avg Power** | **{avg_power:.0f} W** |")
                report.append(f"| **Enhanced Max Power** | **{max_power:.0f} W** |")
                report.append(f"| Power 95th Percentile | {power_95th:.0f} W |")
                report.append(f"| Power 75th Percentile | {power_75th:.0f} W |")
        
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
        
        # Enhanced Power Distribution
        if not df.empty and 'power_estimate' in df.columns:
            power_data = df[df['power_estimate'] > 0]['power_estimate']
            if len(power_data) > 0:
                report.append("\n## Enhanced Power Distribution")
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
        report.append("- Power estimates use enhanced physics model with temperature-adjusted air density")
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
    parser = argparse.ArgumentParser(description='Analyze Garmin cycling workouts with enhanced power estimation and charts')
    parser.add_argument('-w', '--workout-id', type=int, help='Specific workout ID to analyze')
    args = parser.parse_args()
    
    analyzer = GarminWorkoutAnalyzer()
    
    # Step 1: Connect to Garmin
    if not analyzer.connect_to_garmin():
        return
    
    # Step 2: Download workout
    if args.workout_id:
        activity_id = args.workout_id
        fit_file_path = analyzer.download_specific_workout(activity_id)
        
        if not fit_file_path:
            return
            
        # Run analysis for specific workout
        estimated_cog = analyzer.estimate_cog_from_cadence(fit_file_path)
        confirmed_cog = analyzer.get_user_cog_confirmation(estimated_cog)
        print("Analyzing workout data with enhanced power calculations...")
        analysis_data = analyzer.analyze_fit_file(fit_file_path, confirmed_cog)
        
        if analysis_data is None:
            print("Error: Could not analyze workout data")
            return
            
        print("Generating enhanced report with charts...")
        report_file = analyzer.generate_markdown_report(analysis_data, activity_id=activity_id)
        
        print(f"\nWorkout analysis complete!")
        print(f"Report saved as: {report_file}")
        print(f"Charts saved in report directory")
    else:
        fit_file_path = analyzer.download_latest_workout()
        activity_id = analyzer.last_activity_id

        if not fit_file_path:
            return
        
        # Step 3: Estimate cog size and get user confirmation
        estimated_cog = analyzer.estimate_cog_from_cadence(fit_file_path)
        confirmed_cog = analyzer.get_user_cog_confirmation(estimated_cog)
        
        # Step 4: Analyze workout file with enhanced processing
        print("Analyzing workout data with enhanced power calculations...")
        analysis_data = analyzer.analyze_fit_file(fit_file_path, confirmed_cog)
        
        if analysis_data is None:
            print("Error: Could not analyze workout data")
            return
        
        # Step 5: Generate enhanced markdown report with charts
        print("Generating enhanced report with charts...")
        report_file = analyzer.generate_markdown_report(analysis_data, activity_id=activity_id)
        
        print(f"\nWorkout analysis complete!")
        print(f"Report saved as: {report_file}")
        print(f"Charts saved in report directory")

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
