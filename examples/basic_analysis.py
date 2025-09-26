#!/usr/bin/env python3
"""Basic example of using Garmin Analyser to process workout files."""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from parsers.file_parser import FileParser
from analyzers.workout_analyzer import WorkoutAnalyzer
from visualizers.chart_generator import ChartGenerator
from visualizers.report_generator import ReportGenerator


def analyze_workout(file_path: str, output_dir: str = "output"):
    """Analyze a single workout file and generate reports."""
    
    # Initialize components
    settings = Settings()
    parser = FileParser()
    analyzer = WorkoutAnalyzer(settings.zones)
    chart_gen = ChartGenerator()
    report_gen = ReportGenerator(settings)
    
    # Parse the workout file
    print(f"Parsing workout file: {file_path}")
    workout = parser.parse_file(Path(file_path))
    
    if workout is None:
        print("Failed to parse workout file")
        return
    
    print(f"Workout type: {workout.metadata.sport}")
    print(f"Duration: {workout.metadata.duration}")
    print(f"Start time: {workout.metadata.start_time}")
    
    # Analyze the workout
    print("Analyzing workout data...")
    analysis = analyzer.analyze_workout(workout)
    
    # Print basic summary
    summary = analysis['summary']
    print("\n=== WORKOUT SUMMARY ===")
    print(f"Average Power: {summary.get('avg_power', 'N/A')} W")
    print(f"Average Heart Rate: {summary.get('avg_heart_rate', 'N/A')} bpm")
    print(f"Average Speed: {summary.get('avg_speed', 'N/A')} km/h")
    print(f"Distance: {summary.get('distance', 'N/A')} km")
    print(f"Elevation Gain: {summary.get('elevation_gain', 'N/A')} m")
    print(f"Training Stress Score: {summary.get('training_stress_score', 'N/A')}")
    
    # Generate charts
    print("\nGenerating charts...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Power curve
    if 'power_curve' in analysis:
        chart_gen.create_power_curve_chart(
            analysis['power_curve'],
            output_path / "power_curve.png"
        )
        print("Power curve saved to power_curve.png")
    
    # Heart rate zones
    if 'heart_rate_zones' in analysis:
        chart_gen.create_heart_rate_zones_chart(
            analysis['heart_rate_zones'],
            output_path / "hr_zones.png"
        )
        print("Heart rate zones saved to hr_zones.png")
    
    # Elevation profile
    if workout.samples and any(s.elevation for s in workout.samples):
        chart_gen.create_elevation_profile(
            workout.samples,
            output_path / "elevation_profile.png"
        )
        print("Elevation profile saved to elevation_profile.png")
    
    # Generate report
    print("\nGenerating report...")
    report_gen.generate_report(
        workout,
        analysis,
        output_path / "workout_report.html"
    )
    print("Report saved to workout_report.html")
    
    return analysis


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python basic_analysis.py <workout_file> [output_dir]")
        print("Example: python basic_analysis.py workout.fit")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    try:
        analyze_workout(file_path, output_dir)
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()