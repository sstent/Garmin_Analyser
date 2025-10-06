import os
import unittest
import logging
import io
import sys

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import settings as config_settings
from clients.garmin_client import GarminClient

class CredentialsSmokeTest(unittest.TestCase):

    def setUp(self):
        """Set up test environment for each test."""
        self.original_environ = dict(os.environ)
        # Reset the warning flag before each test
        if hasattr(config_settings, '_username_deprecation_warned'):
            delattr(config_settings, '_username_deprecation_warned')

        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger("config.settings")
        self.original_level = self.logger.level
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        """Clean up test environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_environ)
        
        self.logger.removeHandler(self.log_handler)
        self.logger.setLevel(self.original_level)
        if hasattr(config_settings, '_username_deprecation_warned'):
            delattr(config_settings, '_username_deprecation_warned')

    def test_case_A_email_and_password(self):
        """Case A: With GARMIN_EMAIL and GARMIN_PASSWORD set."""
        os.environ["GARMIN_EMAIL"] = "test@example.com"
        os.environ["GARMIN_PASSWORD"] = "password123"
        if "GARMIN_USERNAME" in os.environ:
            del os.environ["GARMIN_USERNAME"]

        email, password = config_settings.get_garmin_credentials()
        self.assertEqual(email, "test@example.com")
        self.assertEqual(password, "password123")
        
        log_output = self.log_stream.getvalue()
        self.assertNotIn("DeprecationWarning", log_output)

    def test_case_B_username_fallback_and_one_time_warning(self):
        """Case B: With only GARMIN_USERNAME and GARMIN_PASSWORD set."""
        os.environ["GARMIN_USERNAME"] = "testuser"
        os.environ["GARMIN_PASSWORD"] = "password456"
        if "GARMIN_EMAIL" in os.environ:
            del os.environ["GARMIN_EMAIL"]

        # First call
        email, password = config_settings.get_garmin_credentials()
        self.assertEqual(email, "testuser")
        self.assertEqual(password, "password456")

        # Second call
        config_settings.get_garmin_credentials()

        log_output = self.log_stream.getvalue()
        self.assertIn("GARMIN_USERNAME is deprecated", log_output)
        # Check that the warning appears only once
        self.assertEqual(log_output.count("GARMIN_USERNAME is deprecated"), 1)

    def test_case_C_garmin_client_credential_sourcing(self):
        """Case C: GarminClient uses accessor-sourced credentials."""
        from unittest.mock import patch, MagicMock

        with patch('clients.garmin_client.get_garmin_credentials', return_value=("test@example.com", "secret")) as mock_get_creds:
            with patch('clients.garmin_client.Garmin') as mock_garmin_connect:
                mock_client_instance = MagicMock()
                mock_garmin_connect.return_value = mock_client_instance

                client = GarminClient()
                client.authenticate()

                mock_get_creds.assert_called_once()
                mock_garmin_connect.assert_called_once_with("test@example.com", "secret")
                mock_client_instance.login.assert_called_once()

if __name__ == '__main__':
    unittest.main()