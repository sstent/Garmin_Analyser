#!/usr/bin/env python3
"""Main entry point for Garmin Analyser application."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from config import settings
from clients.garmin_client import GarminClient
from parsers.file_parser import FileParser
from analyzers.workout_analyzer import WorkoutAnalyzer
from visualizers.chart_generator import ChartGenerator
from visualizers.report_generator import ReportGenerator


def setup_logging(verbose: bool = False):
    """Set up logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('garmin_analyser.log')
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze Garmin workout data from files or Garmin Connect',
        epilog=(
            'Examples:\n'
            '  Analyze latest workout from Garmin Connect: python main.py --garmin-connect\n'
            '  Analyze specific workout by ID: python main.py --workout-id 123456789\n'
            '  Download all cycling workouts: python main.py --download-all\n'
            '  Re-analyze all downloaded workouts: python main.py --reanalyze-all\n'
            '  Analyze local FIT file: python main.py --file path/to/workout.fit\n'
            '  Analyze directory of workouts: python main.py --directory data/\n\n'
            'Configuration:\n'
            '  Set Garmin credentials in .env file: GARMIN_EMAIL and GARMIN_PASSWORD\n'
            '  Configure zones in config/config.yaml or use --zones flag\n'
            '  Override FTP with --ftp flag, max HR with --max-hr flag\n\n'
            'Output:\n'
            '  Reports saved to output/ directory by default\n'
            '  Charts saved to output/charts/ when --charts is used'
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to workout file (FIT, TCX, or GPX)'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory containing workout files'
    )
    input_group.add_argument(
        '--garmin-connect',
        action='store_true',
        help='Download from Garmin Connect'
    )
    input_group.add_argument(
        '--workout-id',
        type=int,
        help='Analyze specific workout by ID from Garmin Connect'
    )
    input_group.add_argument(
        '--download-all',
        action='store_true',
        help='Download all cycling activities from Garmin Connect (no analysis)'
    )
    input_group.add_argument(
        '--reanalyze-all',
        action='store_true',
        help='Re-analyze all downloaded activities and generate reports'
    )
    
    # Analysis options
    parser.add_argument(
        '--ftp',
        type=int,
        help='Functional Threshold Power (W)'
    )
    
    parser.add_argument(
        '--max-hr',
        type=int,
        help='Maximum heart rate (bpm)'
    )
    
    parser.add_argument(
        '--zones',
        type=str,
        help='Path to zones configuration file'
    )
    
    parser.add_argument(
        '--cog',
        type=int,
        help='Cog size (teeth) for power calculations. Auto-detected if not provided'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for reports and charts'
    )
    
    parser.add_argument(
        '--format',
        choices=['html', 'pdf', 'markdown'],
        default='html',
        help='Report format'
    )
    
    parser.add_argument(
        '--charts',
        action='store_true',
        help='Generate charts'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate comprehensive report'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate summary report for multiple workouts'
    )
    
    return parser.parse_args()


class GarminAnalyser:
    """Main application class."""
    
    def __init__(self):
        """Initialize the analyser."""
        self.settings = settings
        self.file_parser = FileParser()
        self.workout_analyzer = WorkoutAnalyzer()
        self.chart_generator = ChartGenerator(Path(settings.REPORTS_DIR) / 'charts')
        self.report_generator = ReportGenerator()
        
        # Create report templates
        self.report_generator.create_report_templates()
    
    def analyze_file(self, file_path: Path) -> dict:
        """Analyze a single workout file.
        
        Args:
            file_path: Path to workout file
            
        Returns:
            Analysis results
        """
        logging.info(f"Analyzing file: {file_path}")
        
        # Parse workout file
        workout = self.file_parser.parse_file(file_path)
        if not workout:
            raise ValueError(f"Failed to parse file: {file_path}")
        
        # Analyze workout
        analysis = self.workout_analyzer.analyze_workout(workout)
        
        return {
            'workout': workout,
            'analysis': analysis,
            'file_path': file_path
        }
    
    def analyze_directory(self, directory: Path) -> List[dict]:
        """Analyze all workout files in a directory.
        
        Args:
            directory: Directory containing workout files
            
        Returns:
            List of analysis results
        """
        logging.info(f"Analyzing directory: {directory}")
        
        results = []
        supported_extensions = {'.fit', '.tcx', '.gpx'}
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    result = self.analyze_file(file_path)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error analyzing {file_path}: {e}")
        
        return results
    
    def download_from_garmin(self, days: int = 30) -> List[dict]:
        """Download and analyze workouts from Garmin Connect.
        
        Args:
            days: Number of days to download
            
        Returns:
            List of analysis results
        """
        logging.info(f"Downloading workouts from Garmin Connect (last {days} days)")
        
        client = GarminClient(
            email=settings.GARMIN_EMAIL,
            password=settings.GARMIN_PASSWORD
        )
        
        # Download workouts
        workouts = client.get_all_cycling_workouts()
        
        # Analyze each workout
        results = []
        for workout_summary in workouts:
            try:
                activity_id = workout_summary.get('activityId')
                if not activity_id:
                    logging.warning("Skipping workout with no activity ID.")
                    continue

                logging.info(f"Downloading workout file for activity ID: {activity_id}")
                workout_file_path = client.download_activity_original(str(activity_id))

                if workout_file_path and workout_file_path.exists():
                    workout = self.file_parser.parse_file(workout_file_path)
                    if workout:
                        analysis = self.workout_analyzer.analyze_workout(workout)
                        results.append({
                            'workout': workout,
                            'analysis': analysis,
                            'file_path': workout_file_path
                        })
                else:
                    logging.error(f"Failed to download workout file for activity ID: {activity_id}")

            except Exception as e:
                logging.error(f"Error analyzing workout: {e}")
        
        return results
    
    def download_all_workouts(self) -> List[dict]:
        """Download all cycling activities from Garmin Connect without analysis.
        
        Returns:
            List of downloaded workouts
        """
        client = GarminClient(
            email=settings.GARMIN_EMAIL,
            password=settings.GARMIN_PASSWORD
        )
        
        # Download all cycling workouts
        workouts = client.get_all_cycling_workouts()
        
        # Save workouts to files
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        downloaded_workouts = []
        for workout in workouts:
            try:
                # Generate filename
                date_str = workout.metadata.start_time.strftime('%Y-%m-%d')
                filename = f"{date_str}_{workout.metadata.activity_name.replace(' ', '_')}.fit"
                file_path = data_dir / filename
                
                # Save workout data
                client.download_workout_file(workout.id, file_path)
                
                downloaded_workouts.append({
                    'workout': workout,
                    'file_path': file_path
                })
                
                logging.info(f"Downloaded: {filename}")
                
            except Exception as e:
                logging.error(f"Error downloading workout {workout.id}: {e}")
        
        logging.info(f"Downloaded {len(downloaded_workouts)} workouts")
        return downloaded_workouts
    
    def reanalyze_all_workouts(self) -> List[dict]:
        """Re-analyze all downloaded workout files.
        
        Returns:
            List of analysis results
        """
        logging.info("Re-analyzing all downloaded workouts")
        
        data_dir = Path('data')
        if not data_dir.exists():
            logging.error("No data directory found. Use --download-all first.")
            return []
        
        results = []
        supported_extensions = {'.fit', '.tcx', '.gpx'}
        
        for file_path in data_dir.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    result = self.analyze_file(file_path)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error re-analyzing {file_path}: {e}")
        
        logging.info(f"Re-analyzed {len(results)} workouts")
        return results
    
    def analyze_workout_by_id(self, workout_id: int) -> dict:
        """Analyze a specific workout by ID from Garmin Connect.
        
        Args:
            workout_id: Garmin Connect workout ID
            
        Returns:
            Analysis result
        """
        logging.info(f"Analyzing workout ID: {workout_id}")
        
        client = GarminClient(
            email=settings.GARMIN_EMAIL,
            password=settings.GARMIN_PASSWORD
        )
        
        # Download specific workout
        workout = client.get_workout_by_id(workout_id)
        if not workout:
            raise ValueError(f"Workout not found: {workout_id}")
        
        # Analyze workout
        analysis = self.workout_analyzer.analyze_workout(workout)
        
        return {
            'workout': workout,
            'analysis': analysis,
            'file_path': None
        }
    
    def generate_outputs(self, results: List[dict], args: argparse.Namespace):
        """Generate charts and reports based on results.
        
        Args:
            results: Analysis results
            args: Command line arguments
        """
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if args.charts:
            logging.info("Generating charts...")
            for result in results:
                charts = self.chart_generator.generate_workout_charts(
                    result['workout'], result['analysis']
                )
                logging.info(f"Charts saved to: {output_dir / 'charts'}")
        
        if args.report:
            logging.info("Generating reports...")
            for result in results:
                report_path = self.report_generator.generate_workout_report(
                    result['workout'], result['analysis'], args.format
                )
                logging.info(f"Report saved to: {report_path}")
        
        if args.summary and len(results) > 1:
            logging.info("Generating summary report...")
            workouts = [r['workout'] for r in results]
            analyses = [r['analysis'] for r in results]
            summary_path = self.report_generator.generate_summary_report(
                workouts, analyses
            )
            logging.info(f"Summary report saved to: {summary_path}")


def main():
    """Main application entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Override settings with command line arguments
        if args.ftp:
            settings.FTP = args.ftp
        if args.max_hr:
            settings.MAX_HEART_RATE = args.max_hr
        
        # Initialize analyser
        analyser = GarminAnalyser()
        
        # Analyze workouts
        results = []
        
        if args.file:
            file_path = Path(args.file)
            if not file_path.exists():
                logging.error(f"File not found: {file_path}")
                sys.exit(1)
            results = [analyser.analyze_file(file_path)]
        
        elif args.directory:
            directory = Path(args.directory)
            if not directory.exists():
                logging.error(f"Directory not found: {directory}")
                sys.exit(1)
            results = analyser.analyze_directory(directory)
        
        elif args.garmin_connect:
            results = analyser.download_from_garmin()
        
        elif args.workout_id:
            try:
                results = [analyser.analyze_workout_by_id(args.workout_id)]
            except ValueError as e:
                logging.error(f"Error: {e}")
                sys.exit(1)
        
        elif args.download_all:
            analyser.download_all_workouts()
            logging.info("Download complete! Use --reanalyze-all to analyze downloaded workouts.")
            return
        
        elif args.reanalyze_all:
            results = analyser.reanalyze_all_workouts()
        
        # Generate outputs
        if results:
            analyser.generate_outputs(results, args)
        
        # Print summary
        if results:
            logging.info(f"\nAnalysis complete! Processed {len(results)} workout(s)")
            for result in results:
                workout = result['workout']
                analysis = result['analysis']
                logging.info(
                    f"\n{workout.metadata.activity_name} - "
                    f"{analysis.get('summary', {}).get('duration_minutes', 0):.1f} min, "
                    f"{analysis.get('summary', {}).get('distance_km', 0):.1f} km, "
                    f"{analysis.get('summary', {}).get('avg_power', 0):.0f} W avg power"
                )
        
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()