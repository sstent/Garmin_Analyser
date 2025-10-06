import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock

from analyzers.workout_analyzer import WorkoutAnalyzer
from config.settings import BikeConfig
from models.workout import WorkoutData, WorkoutMetadata

class TestPowerEstimation(unittest.TestCase):

    def setUp(self):
        # Patch BikeConfig settings for deterministic tests
        self.patcher_bike_mass = patch.object(BikeConfig, 'BIKE_MASS_KG', 8.0)
        self.patcher_bike_crr = patch.object(BikeConfig, 'BIKE_CRR', 0.004)
        self.patcher_bike_cda = patch.object(BikeConfig, 'BIKE_CDA', 0.3)
        self.patcher_air_density = patch.object(BikeConfig, 'AIR_DENSITY', 1.225)
        self.patcher_drive_efficiency = patch.object(BikeConfig, 'DRIVE_EFFICIENCY', 0.97)
        self.patcher_indoor_aero_disabled = patch.object(BikeConfig, 'INDOOR_AERO_DISABLED', True)
        self.patcher_indoor_baseline = patch.object(BikeConfig, 'INDOOR_BASELINE_WATTS', 10.0)
        self.patcher_smoothing_window = patch.object(BikeConfig, 'POWER_ESTIMATE_SMOOTHING_WINDOW_SAMPLES', 3)
        self.patcher_max_power = patch.object(BikeConfig, 'MAX_POWER_WATTS', 1500)

        # Start all patches
        self.patcher_bike_mass.start()
        self.patcher_bike_crr.start()
        self.patcher_bike_cda.start()
        self.patcher_air_density.start()
        self.patcher_drive_efficiency.start()
        self.patcher_indoor_aero_disabled.start()
        self.patcher_indoor_baseline.start()
        self.patcher_smoothing_window.start()
        self.patcher_max_power.start()

        # Setup logger capture
        self.logger = logging.getLogger('analyzers.workout_analyzer')
        self.logger.setLevel(logging.DEBUG)
        self.log_capture = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.log_capture.append(record.getMessage())
        self.logger.addHandler(self.handler)

        # Create analyzer
        self.analyzer = WorkoutAnalyzer()

    def tearDown(self):
        # Stop all patches
        self.patcher_bike_mass.stop()
        self.patcher_bike_crr.stop()
        self.patcher_bike_cda.stop()
        self.patcher_air_density.stop()
        self.patcher_drive_efficiency.stop()
        self.patcher_indoor_aero_disabled.stop()
        self.patcher_indoor_baseline.stop()
        self.patcher_smoothing_window.stop()
        self.patcher_max_power.stop()

        # Restore logger
        self.logger.removeHandler(self.handler)

    def _create_mock_workout(self, df_data, metadata_attrs=None):
        """Create a mock WorkoutData object."""
        workout = MagicMock(spec=WorkoutData)
        workout.raw_data = pd.DataFrame(df_data)
        workout.metadata = MagicMock(spec=WorkoutMetadata)
        # Set default attributes
        workout.metadata.is_indoor = False
        workout.metadata.activity_name = "Outdoor Cycling"
        workout.metadata.duration_seconds = 240  # 4 minutes
        workout.metadata.distance_meters = 1000  # 1 km
        workout.metadata.avg_heart_rate = 150
        workout.metadata.max_heart_rate = 180
        workout.metadata.elevation_gain = 50
        workout.metadata.calories = 200
        # Override with provided attrs
        if metadata_attrs:
            for key, value in metadata_attrs.items():
                setattr(workout.metadata, key, value)
        workout.power = None
        workout.gear = None
        workout.heart_rate = MagicMock()
        workout.heart_rate.heart_rate_values = [150, 160, 170, 180]  # Mock HR values
        workout.speed = MagicMock()
        workout.speed.speed_values = [5.0, 10.0, 15.0, 20.0]  # Mock speed values
        workout.elevation = MagicMock()
        workout.elevation.elevation_values = [0.0, 10.0, 20.0, 30.0]  # Mock elevation values
        return workout

    def test_outdoor_physics_basics(self):
        """Test outdoor physics basics: non-negative, aero effect, no NaNs, cap."""
        # Create DataFrame with monotonic speed and positive gradient
        df_data = {
            'speed': [5.0, 10.0, 15.0, 20.0],  # Increasing speed
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],  # Constant positive gradient
            'distance': [0.0, 5.0, 10.0, 15.0],  # Cumulative distance
            'elevation': [0.0, 10.0, 20.0, 30.0]  # Increasing elevation
        }
        workout = self._create_mock_workout(df_data)

        result = self.analyzer._estimate_power(workout, 16)

        # Assertions
        self.assertEqual(len(result), 4)
        self.assertTrue(all(p >= 0 for p in result))  # Non-negative
        self.assertTrue(result[3] > result[0])  # Higher power at higher speed (aero v^3 effect)
        self.assertTrue(all(not np.isnan(p) for p in result))  # No NaNs
        self.assertTrue(all(p <= BikeConfig.MAX_POWER_WATTS for p in result))  # Capped

        # Check series name
        self.assertIsInstance(result, list)

    def test_indoor_handling(self):
        """Test indoor handling: aero disabled, baseline added, gradient clamped."""
        df_data = {
            'speed': [5.0, 10.0, 15.0, 20.0],
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout = self._create_mock_workout(df_data, {'is_indoor': True, 'activity_name': 'indoor_cycling'})

        indoor_result = self.analyzer._estimate_power(workout, 16)

        # Reset for outdoor comparison
        workout.metadata.is_indoor = False
        workout.metadata.activity_name = "Outdoor Cycling"
        outdoor_result = self.analyzer._estimate_power(workout, 16)

        # Indoor should have lower power due to disabled aero
        self.assertTrue(indoor_result[3] < outdoor_result[3])

        # Check baseline effect at low speed
        self.assertTrue(indoor_result[0] >= BikeConfig.INDOOR_BASELINE_WATTS)

        # Check unrealistic gradients clamped
        df_data_unrealistic = {
            'speed': [5.0, 10.0, 15.0, 20.0],
            'gradient_percent': [15.0, 15.0, 15.0, 15.0],  # Unrealistic for indoor
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout_unrealistic = self._create_mock_workout(df_data_unrealistic, {'is_indoor': True})
        result_clamped = self.analyzer._estimate_power(workout_unrealistic, 16)
        # Gradients should be clamped to reasonable range
        self.assertTrue(all(p >= 0 for p in result_clamped))

    def test_inputs_and_fallbacks(self):
        """Test input fallbacks: speed from distance, gradient from elevation, missing data."""
        # Speed from distance
        df_data_speed_fallback = {
            'distance': [0.0, 5.0, 10.0, 15.0],  # 5 m/s average speed
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout_speed_fallback = self._create_mock_workout(df_data_speed_fallback)
        result_speed = self.analyzer._estimate_power(workout_speed_fallback, 16)
        self.assertEqual(len(result_speed), 4)
        self.assertTrue(all(not np.isnan(p) for p in result_speed))
        self.assertTrue(all(p >= 0 for p in result_speed))

        # Gradient from elevation
        df_data_gradient_fallback = {
            'speed': [5.0, 10.0, 15.0, 20.0],
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]  # 2% gradient
        }
        workout_gradient_fallback = self._create_mock_workout(df_data_gradient_fallback)
        result_gradient = self.analyzer._estimate_power(workout_gradient_fallback, 16)
        self.assertEqual(len(result_gradient), 4)
        self.assertTrue(all(not np.isnan(p) for p in result_gradient))

        # No speed or distance - should return zeros
        df_data_no_speed = {
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout_no_speed = self._create_mock_workout(df_data_no_speed)
        result_no_speed = self.analyzer._estimate_power(workout_no_speed, 16)
        self.assertEqual(result_no_speed, [0.0] * 4)

        # Check warning logged for missing speed
        self.assertTrue(any("No speed or distance data" in msg for msg in self.log_capture))

    def test_nan_safety(self):
        """Test NaN safety: isolated NaNs handled, long runs remain NaN/zero."""
        df_data_with_nans = {
            'speed': [5.0, np.nan, 15.0, 20.0],  # Isolated NaN
            'gradient_percent': [2.0, 2.0, np.nan, 2.0],  # Another isolated NaN
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout = self._create_mock_workout(df_data_with_nans)

        result = self.analyzer._estimate_power(workout, 16)

        # Should handle NaNs gracefully
        self.assertEqual(len(result), 4)
        self.assertTrue(all(not np.isnan(p) for p in result))  # No NaNs in final result
        self.assertTrue(all(p >= 0 for p in result))

    def test_clamping_and_smoothing(self):
        """Test clamping and smoothing: spikes capped, smoothing reduces jitter."""
        # Create data with a spike
        df_data_spike = {
            'speed': [5.0, 10.0, 50.0, 20.0],  # Spike at index 2
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout = self._create_mock_workout(df_data_spike)

        result = self.analyzer._estimate_power(workout, 16)

        # Check clamping
        self.assertTrue(all(p <= BikeConfig.MAX_POWER_WATTS for p in result))

        # Check smoothing reduces variation
        # With smoothing window of 3, the spike should be attenuated
        self.assertTrue(result[2] < (BikeConfig.MAX_POWER_WATTS * 0.9))  # Not at max

    def test_integration_via_analyze_workout(self):
        """Test integration via analyze_workout: power_estimate added when real power missing."""
        df_data = {
            'speed': [5.0, 10.0, 15.0, 20.0],
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }
        workout = self._create_mock_workout(df_data)

        analysis = self.analyzer.analyze_workout(workout, 16)

        # Should have power_estimate when no real power
        self.assertIn('power_estimate', analysis)
        self.assertIn('avg_power', analysis['power_estimate'])
        self.assertIn('max_power', analysis['power_estimate'])
        self.assertTrue(analysis['power_estimate']['avg_power'] > 0)
        self.assertTrue(analysis['power_estimate']['max_power'] > 0)

        # Should have estimated_power in analysis
        self.assertIn('estimated_power', analysis)
        self.assertEqual(len(analysis['estimated_power']), 4)

        # Now test with real power present
        workout.power = MagicMock()
        workout.power.power_values = [100, 200, 300, 400]
        analysis_with_real = self.analyzer.analyze_workout(workout, 16)

        # Should not have power_estimate when real power exists
        self.assertNotIn('power_estimate', analysis_with_real)

        # Should still have estimated_power (for internal use)
        self.assertIn('estimated_power', analysis_with_real)

    def test_logging(self):
        """Test logging: info for indoor/outdoor, warnings for missing data."""
        df_data = {
            'speed': [5.0, 10.0, 15.0, 20.0],
            'gradient_percent': [2.0, 2.0, 2.0, 2.0],
            'distance': [0.0, 5.0, 10.0, 15.0],
            'elevation': [0.0, 10.0, 20.0, 30.0]
        }

        # Test indoor logging
        workout_indoor = self._create_mock_workout(df_data, {'is_indoor': True})
        self.analyzer._estimate_power(workout_indoor, 16)
        self.assertTrue(any("indoor" in msg.lower() for msg in self.log_capture))

        # Clear log
        self.log_capture.clear()

        # Test outdoor logging
        workout_outdoor = self._create_mock_workout(df_data, {'is_indoor': False})
        self.analyzer._estimate_power(workout_outdoor, 16)
        self.assertTrue(any("outdoor" in msg.lower() for msg in self.log_capture))

        # Clear log
        self.log_capture.clear()

        # Test warning for missing speed
        df_data_no_speed = {'gradient_percent': [2.0, 2.0, 2.0, 2.0]}
        workout_no_speed = self._create_mock_workout(df_data_no_speed)
        self.analyzer._estimate_power(workout_no_speed, 16)
        self.assertTrue(any("No speed or distance data" in msg for msg in self.log_capture))

if __name__ == '__main__':
    unittest.main()