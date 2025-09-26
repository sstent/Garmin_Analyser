"""Configuration settings for Garmin Analyser."""

import os
from pathlib import Path
from typing import Dict, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
SMOOTHING_WINDOW = 5  # seconds for gradient smoothing
MIN_WORKOUT_DURATION = 300  # seconds (5 minutes)
MAX_POWER_ESTIMATE = 1000  # watts

# User-specific settings (can be overridden via CLI or environment)
FTP = int(os.getenv("FTP", "250"))  # Functional Threshold Power in watts
MAX_HEART_RATE = int(os.getenv("MAX_HEART_RATE", "185"))  # Maximum heart rate in bpm
COG_SIZE = int(os.getenv("COG_SIZE", str(BikeConfig.DEFAULT_CHAINRING_TEETH)))  # Chainring teeth

# Zones configuration
ZONES_FILE = BASE_DIR / "config" / "zones.json"