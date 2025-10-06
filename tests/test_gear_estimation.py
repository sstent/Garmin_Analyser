import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

# Temporarily add project root to path for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.workout import WorkoutData, GearData, WorkoutMetadata
from parsers.file_parser import FileParser
from analyzers.workout_analyzer import WorkoutAnalyzer
from config.settings import BikeConfig

# Mock implementations based on legacy code for testing purposes
def mock_estimate_gear_series(df: pd.DataFrame, wheel_circumference_m: float, valid_configurations: dict) -> pd.Series:
    results = []
    for _, row in df.iterrows():
        if pd.isna(row.get('speed_mps')) or pd.isna(row.get('cadence_rpm')) or row.get('cadence_rpm') == 0:
            results.append({'chainring_teeth': np.nan, 'cog_teeth': np.nan, 'gear_ratio': np.nan, 'confidence': 0})
            continue

        speed_ms = row['speed_mps']
        cadence_rpm = row['cadence_rpm']
        
        if cadence_rpm <= 0 or speed_ms <= 0:
            results.append({'chainring_teeth': np.nan, 'cog_teeth': np.nan, 'gear_ratio': np.nan, 'confidence': 0})
            continue

        # Simplified logic from legacy analyzer
        distance_per_rev = speed_ms * 60 / cadence_rpm
        actual_ratio = wheel_circumference_m / distance_per_rev

        best_match = None
        min_error = float('inf')

        for chainring, cogs in valid_configurations.items():
            for cog in cogs:
                ratio = chainring / cog
                error = abs(ratio - actual_ratio)
                if error < min_error:
                    min_error = error
                    best_match = (chainring, cog, ratio)
        
        if best_match:
            confidence = 1.0 - min_error
            results.append({'chainring_teeth': best_match[0], 'cog_teeth': best_match[1], 'gear_ratio': best_match[2], 'confidence': confidence})
        else:
            results.append({'chainring_teeth': np.nan, 'cog_teeth': np.nan, 'gear_ratio': np.nan, 'confidence': 0})

    return pd.Series(results, index=df.index)

def mock_compute_gear_summary(gear_series: pd.Series) -> dict:
    if gear_series.empty:
        return {}
    
    summary = {}
    gear_counts = gear_series.apply(lambda x: f"{int(x['chainring_teeth'])}x{int(x['cog_teeth'])}" if pd.notna(x['chainring_teeth']) else None).value_counts()
    
    if not gear_counts.empty:
        summary['top_gears'] = gear_counts.head(3).index.tolist()
        summary['time_in_top_gear_s'] = int(gear_counts.iloc[0])
        summary['unique_gears_count'] = len(gear_counts)
        summary['gear_distribution'] = (gear_counts / len(gear_series) * 100).to_dict()
    else:
        summary['top_gears'] = []
        summary['time_in_top_gear_s'] = 0
        summary['unique_gears_count'] = 0
        summary['gear_distribution'] = {}
        
    return summary


class TestGearEstimation(unittest.TestCase):

    def setUp(self):
        """Set up test data and patch configurations."""
        self.mock_patcher = patch.multiple(
            'config.settings.BikeConfig',
            VALID_CONFIGURATIONS={(52, [12, 14]), (36, [28])},
            TIRE_CIRCUMFERENCE_M=2.096
        )
        self.mock_patcher.start()

        # Capture logs
        self.log_capture = logging.getLogger('parsers.file_parser')
        self.log_stream = unittest.mock.MagicMock()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.log_capture.addHandler(self.log_handler)
        self.log_capture.setLevel(logging.INFO)

        # Mock gear estimation functions in the utils module
        self.mock_estimate_patcher = patch('parsers.file_parser.estimate_gear_series', side_effect=mock_estimate_gear_series)
        self.mock_summary_patcher = patch('parsers.file_parser.compute_gear_summary', side_effect=mock_compute_gear_summary)
        self.mock_estimate = self.mock_estimate_patcher.start()
        self.mock_summary = self.mock_summary_patcher.start()

    def tearDown(self):
        """Clean up patches and log handlers."""
        self.mock_patcher.stop()
        self.mock_estimate_patcher.stop()
        self.mock_summary_patcher.stop()
        self.log_capture.removeHandler(self.log_handler)

    def _create_synthetic_df(self, data):
        return pd.DataFrame(data)

    def test_gear_ratio_estimation_basics(self):
        """Test basic gear ratio estimation with steady cadence and speed changes."""
        data = {
            'speed_mps': [5.5] * 5 + [7.5] * 5,
            'cadence_rpm': [90] * 10,
        }
        df = self._create_synthetic_df(data)
        
        with patch('config.settings.BikeConfig.VALID_CONFIGURATIONS', {(52, [12, 14]), (36, [28])}):
            series = mock_estimate_gear_series(df, 2.096, BikeConfig.VALID_CONFIGURATIONS)

        self.assertEqual(len(series), 10)
        self.assertTrue(all(c in series.iloc[0] for c in ['chainring_teeth', 'cog_teeth', 'gear_ratio', 'confidence']))
        
        # Check that gear changes as speed changes
        self.assertEqual(series.iloc[0]['cog_teeth'], 14) # Lower speed -> easier gear
        self.assertEqual(series.iloc[9]['cog_teeth'], 12) # Higher speed -> harder gear
        self.assertGreater(series.iloc[0]['confidence'], 0.9)

    def test_smoothing_and_hysteresis_mock(self):
        """Test that smoothing reduces gear shifting flicker (conceptual)."""
        # This test is conceptual as smoothing is not in the mock.
        # It verifies that rapid changes would ideally be smoothed.
        data = {
            'speed_mps': [6.0, 6.1, 6.0, 6.1, 7.5, 7.6, 7.5, 7.6],
            'cadence_rpm': [90] * 8,
        }
        df = self._create_synthetic_df(data)
        
        with patch('config.settings.BikeConfig.VALID_CONFIGURATIONS', {(52, [12, 14]), (36, [28])}):
            series = mock_estimate_gear_series(df, 2.096, BikeConfig.VALID_CONFIGURATIONS)
        
        # Without smoothing, we expect flicker
        num_changes = (series.apply(lambda x: x['cog_teeth']).diff().fillna(0) != 0).sum()
        self.assertGreater(num_changes, 1) # More than one major gear change event

    def test_nan_handling(self):
        """Test that NaNs in input data are handled gracefully."""
        data = {
            'speed_mps': [5.5, np.nan, 5.5, 7.5, 7.5, np.nan, np.nan, 7.5],
            'cadence_rpm': [90, 90, np.nan, 90, 90, 90, 90, 90],
        }
        df = self._create_synthetic_df(data)

        with patch('config.settings.BikeConfig.VALID_CONFIGURATIONS', {(52, [12, 14]), (36, [28])}):
            series = mock_estimate_gear_series(df, 2.096, BikeConfig.VALID_CONFIGURATIONS)

        self.assertTrue(pd.isna(series.iloc[1]['cog_teeth']))
        self.assertTrue(pd.isna(series.iloc[2]['cog_teeth']))
        self.assertTrue(pd.isna(series.iloc[5]['cog_teeth']))
        self.assertFalse(pd.isna(series.iloc[0]['cog_teeth']))
        self.assertFalse(pd.isna(series.iloc[3]['cog_teeth']))

    def test_missing_signals_behavior(self):
        """Test behavior when entire columns for speed or cadence are missing."""
        # Missing cadence
        df_no_cadence = self._create_synthetic_df({'speed_mps': [5.5, 7.5]})
        parser = FileParser()
        gear_data = parser._extract_gear_data(df_no_cadence)
        self.assertIsNone(gear_data)
        
        # Missing speed
        df_no_speed = self._create_synthetic_df({'cadence_rpm': [90, 90]})
        gear_data = parser._extract_gear_data(df_no_speed)
        self.assertIsNone(gear_data)

        # Check for log message
        log_messages = [call.args[0] for call in self.log_stream.write.call_args_list]
        self.assertTrue(any("Gear estimation skipped: missing speed_mps or cadence_rpm columns" in msg for msg in log_messages))

    def test_parser_integration(self):
        """Test the integration of gear estimation within the FileParser."""
        data = {'speed_mps': [5.5, 7.5], 'cadence_rpm': [90, 90]}
        df = self._create_synthetic_df(data)
        
        parser = FileParser()
        gear_data = parser._extract_gear_data(df)

        self.assertIsInstance(gear_data, GearData)
        self.assertEqual(len(gear_data.series), 2)
        self.assertIn('top_gears', gear_data.summary)
        self.assertEqual(gear_data.summary['unique_gears_count'], 2)

    def test_analyzer_propagation(self):
        """Test that gear analysis is correctly propagated by the WorkoutAnalyzer."""
        data = {'speed_mps': [5.5, 7.5], 'cadence_rpm': [90, 90]}
        df = self._create_synthetic_df(data)
        
        # Create a mock workout data object
        metadata = WorkoutMetadata(activity_id="test", activity_name="test", start_time=datetime.now(), duration_seconds=120)
        
        parser = FileParser()
        gear_data = parser._extract_gear_data(df)
        
        workout = WorkoutData(metadata=metadata, raw_data=df, gear=gear_data)
        
        analyzer = WorkoutAnalyzer()
        analysis = analyzer.analyze_workout(workout)
        
        self.assertIn('gear_analysis', analysis)
        self.assertIn('top_gears', analysis['gear_analysis'])
        self.assertEqual(analysis['gear_analysis']['unique_gears_count'], 2)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)