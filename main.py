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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Garmin workout data from files or Garmin Connect',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Examples:\n'
            '  %(prog)s analyze --file path/to/workout.fit\n'
            '  %(prog)s batch --directory data/ --output-dir reports/\n'
            '  %(prog)s download --all\n'
            '  %(prog)s reanalyze --input-dir data/\n'
            '  %(prog)s config --show'
        )
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single workout or download from Garmin Connect')
    analyze_parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to workout file (FIT, TCX, or GPX)'
    )
    analyze_parser.add_argument(
        '--garmin-connect',
        action='store_true',
        help='Download and analyze latest workout from Garmin Connect'
    )
    analyze_parser.add_argument(
        '--workout-id',
        type=int,
        help='Analyze specific workout by ID from Garmin Connect'
    )
    analyze_parser.add_argument(
        '--ftp', type=int, help='Functional Threshold Power (W)'
    )
    analyze_parser.add_argument(
        '--max-hr', type=int, help='Maximum heart rate (bpm)'
    )
    analyze_parser.add_argument(
        '--zones', type=str, help='Path to zones configuration file'
    )
    analyze_parser.add_argument(
        '--cog', type=int, help='Cog size (teeth) for power calculations. Auto-detected if not provided'
    )
    analyze_parser.add_argument(
        '--output-dir', type=str, default='output', help='Output directory for reports and charts'
    )
    analyze_parser.add_argument(
        '--format', choices=['html', 'pdf', 'markdown'], default='html', help='Report format'
    )
    analyze_parser.add_argument(
        '--charts', action='store_true', help='Generate charts'
    )
    analyze_parser.add_argument(
        '--report', action='store_true', help='Generate comprehensive report'
    )

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple workout files in a directory')
    batch_parser.add_argument(
        '--directory', '-d', required=True, type=str, help='Directory containing workout files'
    )
    batch_parser.add_argument(
        '--output-dir', type=str, default='output', help='Output directory for reports and charts'
    )
    batch_parser.add_argument(
        '--format', choices=['html', 'pdf', 'markdown'], default='html', help='Report format'
    )
    batch_parser.add_argument(
        '--charts', action='store_true', help='Generate charts'
    )
    batch_parser.add_argument(
        '--report', action='store_true', help='Generate comprehensive report'
    )
    batch_parser.add_argument(
        '--summary', action='store_true', help='Generate summary report for multiple workouts'
    )
    batch_parser.add_argument(
        '--ftp', type=int, help='Functional Threshold Power (W)'
    )
    batch_parser.add_argument(
        '--max-hr', type=int, help='Maximum heart rate (bpm)'
    )
    batch_parser.add_argument(
        '--zones', type=str, help='Path to zones configuration file'
    )
    batch_parser.add_argument(
        '--cog', type=int, help='Cog size (teeth) for power calculations. Auto-detected if not provided'
    )

    # Download command
    download_parser = subparsers.add_parser('download', help='Download activities from Garmin Connect')
    download_parser.add_argument(
        '--all', action='store_true', help='Download all cycling activities'
    )
    download_parser.add_argument(
        '--workout-id', type=int, help='Download specific workout by ID'
    )
    download_parser.add_argument(
        '--limit', type=int, default=50, help='Maximum number of activities to download (with --all)'
    )
    download_parser.add_argument(
        '--output-dir', type=str, default='data', help='Directory to save downloaded files'
    )
    
    # Reanalyze command
    reanalyze_parser = subparsers.add_parser('reanalyze', help='Re-analyze all downloaded activities')
    reanalyze_parser.add_argument(
        '--input-dir', type=str, default='data', help='Directory containing downloaded workouts'
    )
    reanalyze_parser.add_argument(
        '--output-dir', type=str, default='output', help='Output directory for reports and charts'
    )
    reanalyze_parser.add_argument(
        '--format', choices=['html', 'pdf', 'markdown'], default='html', help='Report format'
    )
    reanalyze_parser.add_argument(
        '--charts', action='store_true', help='Generate charts'
    )
    reanalyze_parser.add_argument(
        '--report', action='store_true', help='Generate comprehensive report'
    )
    reanalyze_parser.add_argument(
        '--summary', action='store_true', help='Generate summary report for multiple workouts'
    )
    reanalyze_parser.add_argument(
        '--ftp', type=int, help='Functional Threshold Power (W)'
    )
    reanalyze_parser.add_argument(
        '--max-hr', type=int, help='Maximum heart rate (bpm)'
    )
    reanalyze_parser.add_argument(
        '--zones', type=str, help='Path to zones configuration file'
    )
    reanalyze_parser.add_argument(
        '--cog', type=int, help='Cog size (teeth) for power calculations. Auto-detected if not provided'
    )

    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument(
        '--show', action='store_true', help='Show current configuration'
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
    
    def _apply_analysis_overrides(self, args: argparse.Namespace):
        """Apply FTP, Max HR, and zones overrides from arguments."""
        if hasattr(args, 'ftp') and args.ftp:
            self.settings.FTP = args.ftp
        if hasattr(args, 'max_hr') and args.max_hr:
            self.settings.MAX_HEART_RATE = args.max_hr
        if hasattr(args, 'zones') and args.zones:
            self.settings.ZONES_FILE = args.zones
            # Reload zones if the file path is updated
            self.settings.load_zones(Path(args.zones))

    def analyze_file(self, file_path: Path, args: argparse.Namespace) -> dict:
        """Analyze a single workout file.

        Args:
            file_path: Path to workout file
            args: Command line arguments including analysis overrides

        Returns:
            Analysis results
        """
        logging.info(f"Analyzing file: {file_path}")
        self._apply_analysis_overrides(args)

        workout = self.file_parser.parse_file(file_path)
        if not workout:
            raise ValueError(f"Failed to parse file: {file_path}")

        # Determine cog size from args or auto-detect
        cog_size = None
        if hasattr(args, 'cog') and args.cog:
            cog_size = args.cog
        elif hasattr(args, 'auto_detect_cog') and args.auto_detect_cog:
            # Implement auto-detection logic if needed, or rely on analyzer's default
            pass

        analysis = self.workout_analyzer.analyze_workout(workout, cog_size=cog_size)
        return {'workout': workout, 'analysis': analysis, 'file_path': file_path}

    def batch_analyze_directory(self, directory: Path, args: argparse.Namespace) -> List[dict]:
        """Analyze multiple workout files in a directory.

        Args:
            directory: Directory containing workout files
            args: Command line arguments including analysis overrides

        Returns:
            List of analysis results
        """
        logging.info(f"Analyzing directory: {directory}")
        self._apply_analysis_overrides(args)

        results = []
        supported_extensions = {'.fit', '.tcx', '.gpx'}

        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    result = self.analyze_file(file_path, args)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error analyzing {file_path}: {e}")
        return results

    def download_workouts(self, args: argparse.Namespace) -> List[dict]:
        """Download workouts from Garmin Connect.

        Args:
            args: Command line arguments for download options

        Returns:
            List of downloaded workout data or analysis results
        """
        email, password = self.settings.get_garmin_credentials()
        client = GarminClient(email=email, password=password)
        
        download_output_dir = Path(getattr(args, 'output_dir', 'data'))
        download_output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_activities = []
        if getattr(args, 'all', False):
            logging.info(f"Downloading up to {getattr(args, 'limit', 50)} cycling activities...")
            downloaded_activities = client.download_all_workouts(limit=getattr(args, 'limit', 50), output_dir=download_output_dir)
        elif getattr(args, 'workout_id', None):
            logging.info(f"Downloading workout {args.workout_id}...")
            activity_path = client.download_activity_original(str(args.workout_id), output_dir=download_output_dir)
            if activity_path:
                downloaded_activities.append({'file_path': activity_path})
        else:
            logging.info("Downloading latest cycling activity...")
            activity_path = client.download_latest_workout(output_dir=download_output_dir)
            if activity_path:
                downloaded_activities.append({'file_path': activity_path})

        results = []
        # Check if any analysis-related flags are set
        if (getattr(args, 'charts', False)) or \
           (getattr(args, 'report', False)) or \
           (getattr(args, 'summary', False)) or \
           (getattr(args, 'ftp', None)) or \
           (getattr(args, 'max_hr', None)) or \
           (getattr(args, 'zones', None)) or \
           (getattr(args, 'cog', None)):
            logging.info("Analyzing downloaded workouts...")
            for activity_data in downloaded_activities:
                file_path = activity_data['file_path']
                try:
                    result = self.analyze_file(file_path, args)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error analyzing downloaded file {file_path}: {e}")
        return results if results else downloaded_activities # Return analysis results if analysis was requested, else just downloaded file paths

    def reanalyze_workouts(self, args: argparse.Namespace) -> List[dict]:
        """Re-analyze all downloaded workout files.

        Args:
            args: Command line arguments including input/output directories and analysis overrides

        Returns:
            List of analysis results
        """
        logging.info("Re-analyzing all downloaded workouts")
        self._apply_analysis_overrides(args)

        input_dir = Path(getattr(args, 'input_dir', 'data'))
        if not input_dir.exists():
            logging.error(f"Input directory not found: {input_dir}. Please download workouts first.")
            return []

        results = []
        supported_extensions = {'.fit', '.tcx', '.gpx'}

        for file_path in input_dir.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    result = self.analyze_file(file_path, args)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error re-analyzing {file_path}: {e}")
        logging.info(f"Re-analyzed {len(results)} workouts")
        return results
    
    def show_config(self):
        """Display current configuration."""
        logging.info("Current Configuration:")
        logging.info("-" * 30)
        config_dict = {
            'FTP': self.settings.FTP,
            'MAX_HEART_RATE': self.settings.MAX_HEART_RATE,
            'ZONES_FILE': getattr(self.settings, 'ZONES_FILE', 'N/A'),
            'REPORTS_DIR': self.settings.REPORTS_DIR,
            'DATA_DIR': self.settings.DATA_DIR,
        }
        for key, value in config_dict.items():
            logging.info(f"{key}: {value}")

    def generate_outputs(self, results: List[dict], args: argparse.Namespace):
        """Generate charts and reports based on results.
        
        Args:
            results: Analysis results
            args: Command line arguments
        """
        output_dir = Path(getattr(args, 'output_dir', 'output'))
        output_dir.mkdir(exist_ok=True)
        
        if getattr(args, 'charts', False):
            logging.info("Generating charts...")
            for result in results:
                self.chart_generator.generate_workout_charts(
                    result['workout'], result['analysis']
                )
            logging.info(f"Charts saved to: {output_dir / 'charts'}")
        
        if getattr(args, 'report', False):
            logging.info("Generating reports...")
            for result in results:
                report_path = self.report_generator.generate_workout_report(
                    result['workout'], result['analysis'], getattr(args, 'format', 'html')
                )
                logging.info(f"Report saved to: {report_path}")
        
        if getattr(args, 'summary', False) and len(results) > 1:
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
        analyser = GarminAnalyser()
        results = []

        if args.command == 'analyze':
            if args.file:
                file_path = Path(args.file)
                if not file_path.exists():
                    logging.error(f"File not found: {file_path}")
                    sys.exit(1)
                results = [analyser.analyze_file(file_path, args)]
            elif args.garmin_connect or args.workout_id:
                results = analyser.download_workouts(args)
            else:
                logging.error("Please specify a file, --garmin-connect, or --workout-id for the analyze command.")
                sys.exit(1)
            
            if results: # Only generate outputs if there are results
                analyser.generate_outputs(results, args)

        elif args.command == 'batch':
            directory = Path(args.directory)
            if not directory.exists():
                logging.error(f"Directory not found: {directory}")
                sys.exit(1)
            results = analyser.batch_analyze_directory(directory, args)
            
            if results: # Only generate outputs if there are results
                analyser.generate_outputs(results, args)

        elif args.command == 'download':
            # Download workouts and potentially analyze them if analysis flags are present
            results = analyser.download_workouts(args)
            if results:
                # If analysis was part of download, generate outputs
                if (getattr(args, 'charts', False) or getattr(args, 'report', False) or getattr(args, 'summary', False)):
                    analyser.generate_outputs(results, args)
                else:
                    logging.info(f"Downloaded {len(results)} activities to {getattr(args, 'output_dir', 'data')}")
            logging.info("Download command complete!")

        elif args.command == 'reanalyze':
            results = analyser.reanalyze_workouts(args)
            if results: # Only generate outputs if there are results
                analyser.generate_outputs(results, args)

        elif args.command == 'config':
            if getattr(args, 'show', False):
                analyser.show_config()
        
        # Print summary for analyze, batch, reanalyze commands if results are available
        if args.command in ['analyze', 'batch', 'reanalyze'] and results:
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
        logging.error(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == '__main__':
    main()