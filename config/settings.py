"""Configuration settings for Garmin Analyser."""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger for this module
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Garmin Connect credentials
GARMIN_EMAIL = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

# Flag to ensure deprecation warning is logged only once per process
_deprecation_warned = False

def get_garmin_credentials() -> Tuple[str, str]:
    """Get Garmin Connect credentials from environment variables.

    Prefers GARMIN_EMAIL and GARMIN_PASSWORD. If GARMIN_EMAIL is not set
    but GARMIN_USERNAME is present, uses GARMIN_USERNAME as email with a
    one-time deprecation warning.

    Returns:
        Tuple of (email, password)

    Raises:
        ValueError: If required credentials are not found
    """
    global _deprecation_warned

    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")

    if email and password:
        return email, password

    # Fallback to GARMIN_USERNAME
    username = os.getenv("GARMIN_USERNAME")
    if username and password:
        if not _deprecation_warned:
            logger.warning(
                "GARMIN_USERNAME is deprecated. Please use GARMIN_EMAIL instead. "
                "GARMIN_USERNAME will be removed in a future version."
            )
            _deprecation_warned = True
        return username, password

    raise ValueError(
        "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD "
        "environment variables."
    )

# Bike specifications
class BikeConfig:
    """Bike configuration constants."""
    
    # Valid gear configurations
    VALID_CONFIGURATIONS: Dict[int, list] = {
        38: [14, 16, 18, 20],
        46: [16]
    }
    
    # Default bike specifications
    DEFAULT_CHAINRING_TEETH = 38
    BIKE_WEIGHT_LBS = 22
    BIKE_WEIGHT_KG = BIKE_WEIGHT_LBS * 0.453592
    
    # Wheel specifications (700x25c)
    WHEEL_CIRCUMFERENCE_MM = 2111  # 700x25c wheel circumference
    WHEEL_CIRCUMFERENCE_M = WHEEL_CIRCUMFERENCE_MM / 1000
    TIRE_CIRCUMFERENCE_M = WHEEL_CIRCUMFERENCE_M  # Alias for gear estimation

    # Physics-based power estimation constants
    BIKE_MASS_KG = 75.0  # Total bike + rider mass in kg
    BIKE_CRR = 0.004  # Rolling resistance coefficient
    BIKE_CDA = 0.3  # Aerodynamic drag coefficient * frontal area (m²)
    AIR_DENSITY = 1.225  # Air density in kg/m³
    DRIVE_EFFICIENCY = 0.97  # Drive train efficiency

    # Analysis toggles and caps
    INDOOR_AERO_DISABLED = True  # Disable aerodynamic term for indoor workouts
    INDOOR_BASELINE_WATTS = 10.0  # Baseline power for indoor when stationary
    POWER_ESTIMATE_SMOOTHING_WINDOW_SAMPLES = 3  # Smoothing window for power estimates
    MAX_POWER_WATTS = 1500  # Maximum allowed power estimate to cap spikes

    # Legacy constants (kept for compatibility)
    AERO_CDA_BASE = 0.324  # Base aerodynamic drag coefficient * frontal area (m²)
    ROLLING_RESISTANCE_BASE = 0.0063  # Base rolling resistance coefficient
    EFFICIENCY = 0.97  # Drive train efficiency
    MECHANICAL_LOSS_COEFF = 5.0  # Mechanical losses in watts
    INDOOR_BASE_RESISTANCE = 0.02  # Base grade equivalent for indoor bikes
    INDOOR_CADENCE_THRESHOLD = 80  # RPM threshold for increased indoor resistance
    
    # Gear ratios
    GEAR_RATIOS = {
        38: {
            14: 38/14,
            16: 38/16,
            18: 38/18,
            20: 38/20
        },
        46: {
            16: 46/16
        }
    }

# Indoor activity detection
INDOOR_KEYWORDS = [
    'indoor_cycling', 'indoor cycling', 'indoor bike', 
    'trainer', 'zwift', 'virtual'
]

# File type detection
SUPPORTED_FORMATS = ['.fit', '.tcx', '.gpx']

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Report generation
REPORT_TEMPLATE_DIR = BASE_DIR / "reports" / "templates"
DEFAULT_REPORT_FORMAT = "markdown"
CHART_DPI = 300
CHART_FORMAT = "png"

# Data processing
SMOOTHING_WINDOW = 10  # meters for gradient smoothing
MIN_WORKOUT_DURATION = 300  # seconds (5 minutes)
MAX_POWER_ESTIMATE = 1000  # watts

# User-specific settings (can be overridden via CLI or environment)
FTP = int(os.getenv("FTP", "250"))  # Functional Threshold Power in watts
MAX_HEART_RATE = int(os.getenv("MAX_HEART_RATE", "185"))  # Maximum heart rate in bpm
COG_SIZE = int(os.getenv("COG_SIZE", str(BikeConfig.DEFAULT_CHAINRING_TEETH)))  # Chainring teeth

# Zones configuration
ZONES_FILE = BASE_DIR / "config" / "zones.json"