"""
Garmin Cycling Analyzer - A comprehensive tool for analyzing cycling workouts from Garmin devices.

This package provides functionality to:
- Parse workout files in FIT, TCX, and GPX formats
- Analyze cycling performance metrics including power, heart rate, and zones
- Generate detailed reports and visualizations
- Connect to Garmin Connect for downloading workouts
- Provide both CLI and programmatic interfaces
"""

__version__ = "1.0.0"
__author__ = "Garmin Cycling Analyzer Team"
__email__ = ""

from .parsers.file_parser import FileParser
from .analyzers.workout_analyzer import WorkoutAnalyzer
from .clients.garmin_client import GarminClient
from .visualizers.chart_generator import ChartGenerator
from .visualizers.report_generator import ReportGenerator

__all__ = [
    'FileParser',
    'WorkoutAnalyzer', 
    'GarminClient',
    'ChartGenerator',
    'ReportGenerator'
]