#!/usr/bin/env python3
"""Test script to verify Garmin Analyser installation and basic functionality."""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        from config.settings import Settings
        print("✓ Settings imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Settings: {e}")
        return False
    
    try:
        from models.workout import WorkoutData, WorkoutMetadata, WorkoutSample
        print("✓ Workout models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import workout models: {e}")
        return False
    
    try:
        from models.zones import Zones, Zone
        print("✓ Zones models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import zones models: {e}")
        return False
    
    try:
        from analyzers.workout_analyzer import WorkoutAnalyzer
        print("✓ WorkoutAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WorkoutAnalyzer: {e}")
        return False
    
    try:
        from visualizers.chart_generator import ChartGenerator
        print("✓ ChartGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ChartGenerator: {e}")
        return False
    
    try:
        from visualizers.report_generator import ReportGenerator
        print("✓ ReportGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ReportGenerator: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config.settings import Settings
        
        settings = Settings()
        print("✓ Settings loaded successfully")
        
        # Test zones configuration
        zones = settings.zones
        print(f"✓ Zones loaded: {len(zones.power_zones)} power zones, {len(zones.heart_rate_zones)} HR zones")
        
        # Test FTP value
        ftp = zones.ftp
        print(f"✓ FTP configured: {ftp} W")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality with mock data."""
    print("\nTesting basic functionality...")
    
    try:
        from models.workout import WorkoutData, WorkoutMetadata, WorkoutSample
        from models.zones import Zones, Zone
        from analyzers.workout_analyzer import WorkoutAnalyzer
        
        # Create mock zones
        zones = Zones(
            ftp=250,
            max_heart_rate=180,
            power_zones=[
                Zone("Recovery", 0, 125, True),
                Zone("Endurance", 126, 175, True),
                Zone("Tempo", 176, 212, True),
                Zone("Threshold", 213, 262, True),
                Zone("VO2 Max", 263, 300, True),
            ],
            heart_rate_zones=[
                Zone("Zone 1", 0, 108, True),
                Zone("Zone 2", 109, 126, True),
                Zone("Zone 3", 127, 144, True),
                Zone("Zone 4", 145, 162, True),
                Zone("Zone 5", 163, 180, True),
            ]
        )
        
        # Create mock workout data
        metadata = WorkoutMetadata(
            sport="cycling",
            start_time="2024-01-01T10:00:00Z",
            duration=3600.0,
            distance=30.0,
            calories=800
        )
        
        # Create mock samples
        samples = []
        for i in range(60):  # 1 sample per minute
            sample = WorkoutSample(
                timestamp=f"2024-01-01T10:{i:02d}:00Z",
                power=200 + (i % 50),  # Varying power
                heart_rate=140 + (i % 20),  # Varying HR
                speed=30.0 + (i % 5),  # Varying speed
                elevation=100 + (i % 10),  # Varying elevation
                cadence=85 + (i % 10),  # Varying cadence
                temperature=20.0  # Constant temperature
            )
            samples.append(sample)
        
        workout = WorkoutData(
            metadata=metadata,
            samples=samples
        )
        
        # Test analysis
        analyzer = WorkoutAnalyzer(zones)
        analysis = analyzer.analyze_workout(workout)
        
        print("✓ Basic analysis completed successfully")
        print(f"  - Summary: {len(analysis['summary'])} metrics")
        print(f"  - Power zones: {len(analysis['power_zones'])} zones")
        print(f"  - HR zones: {len(analysis['heart_rate_zones'])} zones")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'jinja2',
        'pyyaml',
        'fitparse',
        'lxml',
        'python-dateutil'
    ]
    
    failed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\nMissing packages: {', '.join(failed_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests."""
    print("=== Garmin Analyser Installation Test ===\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        if test_func():
            passed += 1
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Garmin Analyser is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())