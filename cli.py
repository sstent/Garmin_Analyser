#!/usr/bin/env python3
"""
Command-line interface for Garmin Cycling Analyzer.

This module provides CLI tools for analyzing cycling workouts from Garmin devices.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Import from the new structure
from Garmin_Analyser.parsers.file_parser import FileParser
from Garmin_Analyser.analyzers.workout_analyzer import WorkoutAnalyzer
from Garmin_Analyser.config import settings

# Import for Garmin Connect functionality
try:
    from Garmin_Analyser.clients.garmin_client import GarminClient
    GARMIN_CLIENT_AVAILABLE = True
except ImportError:
    GARMIN_CLIENT_AVAILABLE = False
    print("Warning: Garmin Connect client not available. Install garminconnect package for download functionality.")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI commands."""
    parser = argparse.ArgumentParser(
        description='Analyze cycling workouts from Garmin devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze file.fit --output results.json
  %(prog)s batch --input-dir ./workouts --output-dir ./results
  %(prog)s config --show
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single workout file')
    analyze_parser.add_argument('file', help='Path to the workout file (.fit, .tcx, or .gpx)')
    analyze_parser.add_argument('--output', '-o', help='Output file for results (JSON format)')
    analyze_parser.add_argument('--cog-size', type=int, help='Chainring cog size (teeth)')
    analyze_parser.add_argument('--format', choices=['json', 'summary'], default='json', 
                               help='Output format (default: json)')
    analyze_parser.add_argument('--ftp', type=int, help='Functional Threshold Power (W)')
    analyze_parser.add_argument('--max-hr', type=int, help='Maximum heart rate (bpm)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple workout files')
    batch_parser.add_argument('--input-dir', '-i', required=True, help='Directory containing workout files')
    batch_parser.add_argument('--output-dir', '-o', required=True, help='Directory for output files')
    batch_parser.add_argument('--cog-size', type=int, help='Chainring cog size (teeth)')
    batch_parser.add_argument('--pattern', default='*.fit', help='File pattern to match (default: *.fit)')
    batch_parser.add_argument('--ftp', type=int, help='Functional Threshold Power (W)')
    batch_parser.add_argument('--max-hr', type=int, help='Maximum heart rate (bpm)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    
    # Download command (from original garmin_cycling_analyzer.py)
    download_parser = subparsers.add_parser('download', help='Download workouts from Garmin Connect')
    download_parser.add_argument('--workout-id', '-w', type=int, help='Download specific workout by ID')
    download_parser.add_argument('--download-all', action='store_true', help='Download all cycling activities')
    download_parser.add_argument('--limit', type=int, default=50, help='Maximum number of activities to download')
    
    # Reanalyze command (from original garmin_cycling_analyzer.py)
    reanalyze_parser = subparsers.add_parser('reanalyze', help='Re-analyze downloaded workouts')
    reanalyze_parser.add_argument('--input-dir', '-i', default='data', help='Directory containing downloaded workouts (default: data)')
    reanalyze_parser.add_argument('--output-dir', '-o', default='reports', help='Directory for analysis reports (default: reports)')
    
    return parser


def analyze_file(file_path: str, cog_size: Optional[int] = None, 
                ftp: Optional[int] = None, max_hr: Optional[int] = None,
                output_format: str = 'json') -> dict:
    """
    Analyze a single workout file.
    
    Args:
        file_path: Path to the workout file
        cog_size: Chainring cog size for power estimation
        ftp: Functional Threshold Power
        max_hr: Maximum heart rate
        output_format: Output format ('json' or 'summary')
        
    Returns:
        Analysis results as dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Override settings with provided parameters
    if ftp:
        settings.FTP = ftp
    if max_hr:
        settings.MAX_HEART_RATE = max_hr
    if cog_size:
        settings.COG_SIZE = cog_size
    
    # Parse the file
    parser = FileParser()
    workout = parser.parse_file(Path(file_path))
    
    if not workout:
        raise ValueError(f"Failed to parse file: {file_path}")
    
    # Analyze the workout
    analyzer = WorkoutAnalyzer()
    results = analyzer.analyze_workout(workout)
    
    return results


def batch_analyze(input_dir: str, output_dir: str, cog_size: Optional[int] = None,
                 ftp: Optional[int] = None, max_hr: Optional[int] = None,
                 pattern: str = '*.fit') -> List[str]:
    """
    Analyze multiple workout files in a directory.
    
    Args:
        input_dir: Directory containing workout files
        output_dir: Directory for output files
        cog_size: Chainring cog size for power estimation
        ftp: Functional Threshold Power
        max_hr: Maximum heart rate
        pattern: File pattern to match
        
    Returns:
        List of processed file paths
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Override settings with provided parameters
    if ftp:
        settings.FTP = ftp
    if max_hr:
        settings.MAX_HEART_RATE = max_hr
    if cog_size:
        settings.COG_SIZE = cog_size
    
    # Find matching files
    files = list(input_path.glob(pattern))
    processed_files = []
    
    for file_path in files:
        try:
            print(f"Analyzing {file_path.name}...")
            results = analyze_file(str(file_path))
            
            # Save results
            output_file = output_path / f"{file_path.stem}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            processed_files.append(str(file_path))
            print(f"  ✓ Results saved to {output_file.name}")
            
        except Exception as e:
            print(f"  ✗ Error analyzing {file_path.name}: {e}")
    
    return processed_files


def show_config():
    """Display current configuration."""
    print("Current Configuration:")
    print("-" * 30)
    config_dict = {
        'FTP': settings.FTP,
        'MAX_HEART_RATE': settings.MAX_HEART_RATE,
        'COG_SIZE': getattr(settings, 'COG_SIZE', None),
        'ZONES_FILE': getattr(settings, 'ZONES_FILE', None),
        'REPORTS_DIR': settings.REPORTS_DIR,
        'DATA_DIR': settings.DATA_DIR,
    }
    
    for key, value in config_dict.items():
        print(f"{key}: {value}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'analyze':
            results = analyze_file(
                args.file,
                cog_size=getattr(args, 'cog_size', None),
                ftp=getattr(args, 'ftp', None),
                max_hr=getattr(args, 'max_hr', None),
                output_format=args.format
            )
            
            if args.format == 'json':
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    print(f"Analysis complete. Results saved to {args.output}")
                else:
                    print(json.dumps(results, indent=2, default=str))
            
            elif args.format == 'summary':
                print_summary(results)
        
        elif args.command == 'batch':
            processed = batch_analyze(
                args.input_dir,
                args.output_dir,
                cog_size=getattr(args, 'cog_size', None),
                ftp=getattr(args, 'ftp', None),
                max_hr=getattr(args, 'max_hr', None),
                pattern=args.pattern
            )
            print(f"\nBatch analysis complete. Processed {len(processed)} files.")
        
        elif args.command == 'config':
            if args.show:
                show_config()
            else:
                show_config()
        
        elif args.command == 'download':
            download_workouts(
                workout_id=getattr(args, 'workout_id', None),
                download_all=args.download_all,
                limit=getattr(args, 'limit', 50)
            )
        
        elif args.command == 'reanalyze':
            reanalyze_workouts(
                input_dir=getattr(args, 'input_dir', 'data'),
                output_dir=getattr(args, 'output_dir', 'reports')
            )
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def print_summary(results: dict):
    """Print a human-readable summary of the analysis."""
    metadata = results.get('metadata', {})
    summary = results.get('summary', {})
    
    print("\n" + "="*50)
    print("WORKOUT SUMMARY")
    print("="*50)
    
    if metadata:
        print(f"Activity: {metadata.get('activity_type', 'Unknown')}")
        print(f"Date: {metadata.get('start_time', 'Unknown')}")
        print(f"Duration: {summary.get('duration_minutes', 0):.1f} minutes")
    
    if summary:
        print(f"\nDistance: {summary.get('distance_km', 0):.1f} km")
        print(f"Average Speed: {summary.get('avg_speed_kmh', 0):.1f} km/h")
        
        if 'avg_power' in summary:
            print(f"Average Power: {summary['avg_power']:.0f} W")
        if 'max_power' in summary:
            print(f"Max Power: {summary['max_power']:.0f} W")
        
        print(f"Average Heart Rate: {summary.get('avg_heart_rate', 0):.0f} bpm")
        print(f"Max Heart Rate: {summary.get('max_heart_rate', 0):.0f} bpm")
        
        elevation = results.get('elevation_analysis', {})
        if elevation:
            print(f"Elevation Gain: {elevation.get('total_elevation_gain', 0):.0f} m")
    
    zones = results.get('zones', {})
    if zones and 'power' in zones:
        print("\nPower Zone Distribution:")
        for zone, data in zones['power'].items():
            print(f"  {zone}: {data['percentage']:.1f}% ({data['time_minutes']:.1f} min)")
    
    print("="*50)


def download_workouts(workout_id: Optional[int] = None, download_all: bool = False, limit: int = 50):
    """
    Download workouts from Garmin Connect.
    
    Args:
        workout_id: Specific workout ID to download
        download_all: Download all cycling activities
        limit: Maximum number of activities to download
    """
    if not GARMIN_CLIENT_AVAILABLE:
        print("Error: Garmin Connect client not available. Install garminconnect package:")
        print("  pip install garminconnect")
        return
    
    try:
        client = GarminClient()
        
        if workout_id:
            print(f"Downloading workout {workout_id}...")
            success = client.download_workout(workout_id)
            if success:
                print(f"✓ Workout {workout_id} downloaded successfully")
            else:
                print(f"✗ Failed to download workout {workout_id}")
        
        elif download_all:
            print(f"Downloading up to {limit} cycling activities...")
            downloaded = client.download_all_workouts(limit=limit)
            print(f"✓ Downloaded {len(downloaded)} activities")
        
        else:
            print("Downloading latest cycling activity...")
            success = client.download_latest_workout()
            if success:
                print("✓ Latest activity downloaded successfully")
            else:
                print("✗ Failed to download latest activity")
                
    except Exception as e:
        print(f"Error downloading workouts: {e}")


def reanalyze_workouts(input_dir: str = 'data', output_dir: str = 'reports'):
    """
    Re-analyze all downloaded workouts.
    
    Args:
        input_dir: Directory containing downloaded workouts
        output_dir: Directory for analysis reports
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all workout files
    patterns = ['*.fit', '*.tcx', '*.gpx']
    files = []
    for pattern in patterns:
        files.extend(input_path.glob(pattern))
    
    if not files:
        print(f"No workout files found in {input_dir}")
        return
    
    print(f"Found {len(files)} workout files to re-analyze")
    
    processed = batch_analyze(
        str(input_path),
        str(output_path),
        pattern='*.*'  # Process all files
    )
    
    print(f"\nRe-analysis complete. Processed {len(processed)} files.")


if __name__ == '__main__':
    main()