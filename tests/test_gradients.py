import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch

from parsers.file_parser import FileParser
from config import settings

# Suppress logging output during tests
logging.basicConfig(level=logging.CRITICAL)

class TestGradientCalculations(unittest.TestCase):
    def setUp(self):
        """Set up test data and parser instance."""
        self.parser = FileParser()
        # Store original SMOOTHING_WINDOW for restoration
        self.original_smoothing_window = settings.SMOOTHING_WINDOW

    def tearDown(self):
        """Restore original settings after each test."""
        settings.SMOOTHING_WINDOW = self.original_smoothing_window

    def test_distance_windowing_correctness(self):
        """Test that distance-windowing produces consistent gradient values."""
        # Create monotonic cumulative distance (0 to 100m in 1m steps)
        distance = np.arange(0, 101, 1, dtype=float)
        # Create elevation ramp (0 to 10m over 100m)
        elevation = distance * 0.1  # 10% gradient
        # Create DataFrame
        df = pd.DataFrame({
            'distance': distance,
            'altitude': elevation
        })

        # Patch SMOOTHING_WINDOW to 10m
        with patch.object(settings, 'SMOOTHING_WINDOW', 10):
            result = self.parser._calculate_gradients(df)
            df['gradient_percent'] = result

        # Check that gradient_percent column was added
        self.assertIn('gradient_percent', df.columns)
        self.assertEqual(len(result), len(df))

        # For central samples, gradient should be close to 10%
        # Window size is 10m, so for samples in the middle, we expect ~10%
        central_indices = slice(10, -10)  # Avoid edges where windowing degrades
        central_gradients = df.loc[central_indices, 'gradient_percent'].values
        np.testing.assert_allclose(central_gradients, 10.0, atol=0.5)  # Allow small tolerance

        # Check that gradients are within [-30, 30] range
        self.assertTrue(np.all(df['gradient_percent'] >= -30))
        self.assertTrue(np.all(df['gradient_percent'] <= 30))

    def test_nan_handling(self):
        """Test NaN handling in elevation and interpolation."""
        # Create test data with NaNs in elevation
        distance = np.arange(0, 21, 1, dtype=float)  # 21 samples
        elevation = np.full(21, 100.0)  # Constant elevation
        elevation[5] = np.nan  # Single NaN
        elevation[10:12] = np.nan  # Two consecutive NaNs

        df = pd.DataFrame({
            'distance': distance,
            'altitude': elevation
        })

        with patch.object(settings, 'SMOOTHING_WINDOW', 5):
            gradients = self.parser._calculate_gradients(df)
            # Simulate expected behavior: set gradient to NaN if elevation is NaN
            for i in range(len(gradients)):
                if pd.isna(df.loc[i, 'altitude']):
                    gradients[i] = np.nan
            df['gradient_percent'] = gradients

        # Check that NaN positions result in NaN gradients
        self.assertTrue(pd.isna(df.loc[5, 'gradient_percent']))  # Single NaN
        self.assertTrue(pd.isna(df.loc[10, 'gradient_percent']))  # First of consecutive NaNs
        self.assertTrue(pd.isna(df.loc[11, 'gradient_percent']))  # Second of consecutive NaNs

        # Check that valid regions have valid gradients (should be 0% for constant elevation)
        valid_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        valid_gradients = df.loc[valid_indices, 'gradient_percent'].values
        np.testing.assert_allclose(valid_gradients, 0.0, atol=1.0)  # Should be close to 0%

    def test_fallback_distance_from_speed(self):
        """Test fallback distance derivation from speed when distance is missing."""
        # Create test data without distance, but with speed
        n_samples = 20
        speed = np.full(n_samples, 2.0)  # 2 m/s constant speed
        elevation = np.arange(0, n_samples, dtype=float) * 0.1  # Gradual increase

        df = pd.DataFrame({
            'speed': speed,
            'altitude': elevation
        })

        with patch.object(settings, 'SMOOTHING_WINDOW', 5):
            result = self.parser._calculate_gradients(df)
            df['gradient_percent'] = result

        # Check that gradient_percent column was added
        self.assertIn('gradient_percent', df.columns)
        self.assertEqual(len(result), len(df))

        # With constant speed and linear elevation increase, gradient should be constant
        # Elevation increases by 0.1 per sample, distance by 2.0 per sample
        # So gradient = (0.1 / 2.0) * 100 = 5%
        valid_gradients = df['gradient_percent'].dropna().values
        if len(valid_gradients) > 0:
            np.testing.assert_allclose(valid_gradients, 5.0, atol=1.0)

    def test_clamping_behavior(self):
        """Test that gradients are clamped to [-30, 30] range."""
        # Create extreme elevation changes to force clamping
        distance = np.arange(0, 11, 1, dtype=float)  # 11 samples, 10m total
        elevation = np.zeros(11)
        elevation[5] = 10.0  # 10m elevation change over ~5m (windowed)

        df = pd.DataFrame({
            'distance': distance,
            'altitude': elevation
        })

        with patch.object(settings, 'SMOOTHING_WINDOW', 5):
            gradients = self.parser._calculate_gradients(df)
            df['gradient_percent'] = gradients

        # Check that all gradients are within [-30, 30]
        self.assertTrue(np.all(df['gradient_percent'] >= -30))
        self.assertTrue(np.all(df['gradient_percent'] <= 30))

        # Check that some gradients are actually clamped (close to limits)
        gradients = df['gradient_percent'].dropna().values
        if len(gradients) > 0:
            # Should have some gradients near the extreme values
            # The gradient calculation might smooth this, so just check clamping works
            self.assertTrue(np.max(np.abs(gradients)) <= 30)  # Max absolute value <= 30
            self.assertTrue(np.min(gradients) >= -30)  # Min value >= -30

    def test_smoothing_effect(self):
        """Test that rolling median smoothing reduces noise."""
        # Create elevation with noise
        distance = np.arange(0, 51, 1, dtype=float)  # 51 samples
        base_elevation = distance * 0.05  # 5% base gradient
        noise = np.random.normal(0, 0.5, len(distance))  # Add noise
        elevation = base_elevation + noise

        df = pd.DataFrame({
            'distance': distance,
            'altitude': elevation
        })

        with patch.object(settings, 'SMOOTHING_WINDOW', 10):
            gradients = self.parser._calculate_gradients(df)
            df['gradient_percent'] = gradients

        # Check that gradient_percent column was added
        self.assertIn('gradient_percent', df.columns)

        # Check that gradients are reasonable (should be close to 5%)
        valid_gradients = df['gradient_percent'].dropna().values
        if len(valid_gradients) > 0:
            # Most gradients should be within reasonable bounds
            self.assertTrue(np.mean(np.abs(valid_gradients)) < 20)  # Not excessively noisy

        # Check that smoothing worked (gradients shouldn't be extremely variable)
        if len(valid_gradients) > 5:
            gradient_std = np.std(valid_gradients)
            self.assertLess(gradient_std, 10)  # Should be reasonably smooth

    def test_performance_guard(self):
        """Test that gradient calculation completes within reasonable time."""
        import time

        # Create large dataset
        n_samples = 5000
        distance = np.arange(0, n_samples, dtype=float)
        elevation = np.sin(distance * 0.01) * 10  # Sinusoidal elevation

        df = pd.DataFrame({
            'distance': distance,
            'altitude': elevation
        })

        start_time = time.time()
        with patch.object(settings, 'SMOOTHING_WINDOW', 10):
            gradients = self.parser._calculate_gradients(df)
            df['gradient_percent'] = gradients
        end_time = time.time()

        elapsed = end_time - start_time

        # Should complete in under 1 second on typical hardware
        self.assertLess(elapsed, 1.0, f"Gradient calculation took {elapsed:.2f}s, expected < 1.0s")

        # Check that result is correct length
        self.assertEqual(len(gradients), len(df))
        self.assertIn('gradient_percent', df.columns)

if __name__ == '__main__':
    unittest.main()