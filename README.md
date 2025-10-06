# Garmin Analyser

A comprehensive Python application for analyzing Garmin workout data from FIT, TCX, and GPX files, as well as direct integration with Garmin Connect. Provides detailed power, heart rate, and performance analysis with beautiful visualizations and comprehensive reports.

## Features

- **Multi-format Support**: Parse FIT files. TCX and GPX parsing is not yet implemented and is planned for a future enhancement.
- **Garmin Connect Integration**: Direct download from Garmin Connect
- **Comprehensive Analysis**: Power, heart rate, speed, elevation, and zone analysis
- **Advanced Metrics**: Normalized Power, Intensity Factor, Training Stress Score
- **Interactive Charts**: Power curves, heart rate zones, elevation profiles
- **Detailed Reports**: HTML, PDF, and Markdown reports with customizable templates
- **Interval Detection**: Automatic detection and analysis of workout intervals
- **Performance Tracking**: Long-term performance trends and summaries

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For PDF report generation:
```bash
pip install weasyprint
```

## Quick Start

### Basic Usage

Analyze a single workout file:
```bash
python main.py --file path/to/workout.fit --report --charts
```

Analyze all workouts in a directory:
```bash
python main.py --directory path/to/workouts --summary --format html
```

Download from Garmin Connect:
```bash
python main.py --garmin-connect --report --charts --summary
```

### Command Line Options

```
usage: main.py [-h] [--config CONFIG] [--verbose]
               (--file FILE | --directory DIRECTORY | --garmin-connect | --workout-id WORKOUT_ID | --download-all | --reanalyze-all)
               [--ftp FTP] [--max-hr MAX_HR] [--zones ZONES] [--cog COG]
               [--output-dir OUTPUT_DIR] [--format {html,pdf,markdown}]
               [--charts] [--report] [--summary]

Analyze Garmin workout data from files or Garmin Connect

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Configuration file path
  --verbose, -v         Enable verbose logging

Input options:
  --file FILE, -f FILE  Path to workout file (FIT, TCX, or GPX)
  --directory DIRECTORY, -d DIRECTORY
                        Directory containing workout files
  --garmin-connect      Download from Garmin Connect
  --workout-id WORKOUT_ID
                        Analyze specific workout by ID from Garmin Connect
  --download-all        Download all cycling activities from Garmin Connect (no analysis)
  --reanalyze-all       Re-analyze all downloaded activities and generate reports

Analysis options:
  --ftp FTP             Functional Threshold Power (W)
  --max-hr MAX_HR       Maximum heart rate (bpm)
  --zones ZONES         Path to zones configuration file
  --cog COG             Cog size (teeth) for power calculations. Auto-detected if not provided

Output options:
  --output-dir OUTPUT_DIR
                        Output directory for reports and charts
  --format {html,pdf,markdown}
                        Report format
  --charts              Generate charts
  --report              Generate comprehensive report
  --summary             Generate summary report for multiple workouts

Examples:
  Analyze latest workout from Garmin Connect: python main.py --garmin-connect
  Analyze specific workout by ID: python main.py --workout-id 123456789
  Download all cycling workouts: python main.py --download-all
  Re-analyze all downloaded workouts: python main.py --reanalyze-all
  Analyze local FIT file: python main.py --file path/to/workout.fit
  Analyze directory of workouts: python main.py --directory data/

Configuration:
  Set Garmin credentials in .env file: GARMIN_EMAIL and GARMIN_PASSWORD
  Configure zones in config/config.yaml or use --zones flag
  Override FTP with --ftp flag, max HR with --max-hr flag

Output:
  Reports saved to output/ directory by default
  Charts saved to output/charts/ when --charts is used
```

## Setup credentials

Canonical environment variables:
- GARMIN_EMAIL
- GARMIN_PASSWORD

Single source of truth:
- Credentials are centrally accessed via [get_garmin_credentials()](config/settings.py:31). If GARMIN_EMAIL is not set but GARMIN_USERNAME is present, the username value is used as email and a one-time deprecation warning is logged. GARMIN_USERNAME is deprecated and will be removed in a future version.

Linux/macOS (bash/zsh):
```bash
export GARMIN_EMAIL="you@example.com"
export GARMIN_PASSWORD="your-app-password"
```

Windows PowerShell:
```powershell
$env:GARMIN_EMAIL = "you@example.com"
$env:GARMIN_PASSWORD = "your-app-password"
```

.env sample:
```dotenv
GARMIN_EMAIL=you@example.com
GARMIN_PASSWORD=your-app-password
```

Note on app passwords:
- If your Garmin account uses two-factor authentication or app-specific passwords, create an app password in your Garmin account settings and use it for GARMIN_PASSWORD.

TUI with dotenv:
- When using the TUI with dotenv, prefer GARMIN_EMAIL and GARMIN_PASSWORD in your .env file. GARMIN_USERNAME continues to work via fallback with a one-time deprecation warning, but it is deprecated; switch to GARMIN_EMAIL.

Parity and unaffected behavior:
- Authentication and download parity is maintained. Original ZIP downloads and FIT extraction workflows are unchanged in [clients/garmin_client.py](clients/garmin_client.py).
- Alternate format downloads (FIT, TCX, GPX) are unaffected by this credentials change.
## Configuration

### Basic Configuration

Create a `config/config.yaml` file:

```yaml
# Garmin Connect credentials
# Credentials are provided via environment variables (GARMIN_EMAIL, GARMIN_PASSWORD).
# Do not store credentials in config.yaml. See "Setup credentials" in README.

# Output settings
output_dir: output
log_level: INFO

# Training zones
zones:
  ftp: 250  # Functional Threshold Power (W)
  max_heart_rate: 185  # Maximum heart rate (bpm)
  
  power_zones:
    - name: Active Recovery
      min: 0
      max: 55
      percentage: true
    - name: Endurance
      min: 56
      max: 75
      percentage: true
    - name: Tempo
      min: 76
      max: 90
      percentage: true
    - name: Threshold
      min: 91
      max: 105
      percentage: true
    - name: VO2 Max
      min: 106
      max: 120
      percentage: true
    - name: Anaerobic
      min: 121
      max: 150
      percentage: true
  
  heart_rate_zones:
    - name: Zone 1
      min: 0
      max: 60
      percentage: true
    - name: Zone 2
      min: 60
      max: 70
      percentage: true
    - name: Zone 3
      min: 70
      max: 80
      percentage: true
    - name: Zone 4
      min: 80
      max: 90
      percentage: true
    - name: Zone 5
      min: 90
      max: 100
      percentage: true
```

### Advanced Configuration

You can also specify zones configuration in a separate file:

```yaml
# zones.yaml
ftp: 275
max_heart_rate: 190

power_zones:
  - name: Recovery
    min: 0
    max: 50
    percentage: true
  - name: Endurance
    min: 51
    max: 70
    percentage: true
  # ... additional zones
```

## Usage Examples

### Single Workout Analysis

```bash
# Analyze a single FIT file with custom FTP
python main.py --file workouts/2024-01-15-ride.fit --ftp 275 --report --charts

# Generate PDF report
python main.py --file workouts/workout.tcx --format pdf --report

# Quick analysis with verbose output
python main.py --file workout.gpx --verbose --report
```

### Batch Analysis

```bash
# Analyze all files in a directory
python main.py --directory data/workouts/ --summary --charts --format html

# Analyze with custom zones
python main.py --directory data/workouts/ --zones config/zones.yaml --summary
```

### Reports: normalized variables example

Reports consume normalized speed and heart rate keys in templates. Example (HTML template):

```jinja2
{# See workout_report.html #}
<p>Sport: {{ metadata.sport }} ({{ metadata.sub_sport }})</p>
<p>Speed: {{ summary.avg_speed_kmh|default(0) }} km/h; HR: {{ summary.avg_hr|default(0) }} bpm</p>
```

- Template references: [workout_report.html](visualizers/templates/workout_report.html:1), [workout_report.md](visualizers/templates/workout_report.md:1)

### Garmin Connect Integration

```bash
# Download and analyze last 30 days
python main.py --garmin-connect --report --charts --summary

# Download specific period
python main.py --garmin-connect --report --output-dir reports/january/
```

## Output Structure

The application creates the following output structure:

```
output/
├── charts/
│   ├── workout_20240115_143022_power_curve.png
│   ├── workout_20240115_143022_heart_rate_zones.png
│   └── ...
├── reports/
│   ├── workout_report_20240115_143022.html
│   ├── workout_report_20240115_143022.pdf
│   └── summary_report_20240115_143022.html
└── logs/
    └── garmin_analyser.log
```

## Analysis Features

### Power Analysis
- **Average Power**: Mean power output
- **Normalized Power**: Adjusted power accounting for variability
- **Maximum Power**: Peak power output
- **Power Zones**: Time spent in each power zone
- **Power Curve**: Maximum power for different durations

### Heart Rate Analysis
- **Average Heart Rate**: Mean heart rate
- **Maximum Heart Rate**: Peak heart rate
- **Heart Rate Zones**: Time spent in each heart rate zone
- **Heart Rate Variability**: Analysis of heart rate patterns

### Performance Metrics
- **Intensity Factor (IF)**: Ratio of Normalized Power to FTP
- **Training Stress Score (TSS)**: Overall training load
- **Variability Index**: Measure of power consistency
- **Efficiency Factor**: Ratio of Normalized Power to Average Heart Rate

### Interval Detection
- Automatic detection of high-intensity intervals
- Analysis of interval duration, power, and recovery
- Summary of interval performance

## Analysis outputs and normalized naming

The analyzer and report pipeline now provide normalized keys for speed and heart rate to ensure consistent units and naming across code and templates. See [WorkoutAnalyzer.analyze_workout()](analyzers/workout_analyzer.py:1) and [ReportGenerator._prepare_report_data()](visualizers/report_generator.py:1) for implementation details.

- Summary keys:
  - summary.avg_speed_kmh — Average speed in km/h (derived from speed_mps)
  - summary.avg_hr — Average heart rate in beats per minute (bpm)
- Speed analysis keys:
  - speed_analysis.avg_speed_kmh — Average speed in km/h
  - speed_analysis.max_speed_kmh — Maximum speed in km/h
- Heart rate analysis keys:
  - heart_rate_analysis.avg_hr — Average heart rate (bpm)
  - heart_rate_analysis.max_hr — Maximum heart rate (bpm)
- Backward-compatibility aliases maintained in code:
  - summary.avg_speed — Alias of avg_speed_kmh
  - summary.avg_heart_rate — Alias of avg_hr

Guidance: templates should use the normalized names going forward.

## Templates: variables and metadata

Templates should reference normalized variables and the workout metadata fields:
- Use metadata.sport and metadata.sub_sport instead of activity_type.
- Example snippet referencing normalized keys:
  - speed: {{ summary.avg_speed_kmh }} km/h; HR: {{ summary.avg_hr }} bpm
- For defensive rendering, Jinja defaults may be used (e.g., {{ summary.avg_speed_kmh|default(0) }}), though normalized keys are expected to be present.

Reference templates:
- [workout_report.html](visualizers/templates/workout_report.html:1)
- [workout_report.md](visualizers/templates/workout_report.md:1)

## Migration note

- Legacy template fields avg_speed and avg_heart_rate are deprecated; the code provides aliases (summary.avg_speed → avg_speed_kmh, summary.avg_heart_rate → avg_hr) to prevent breakage temporarily.
- Users should update custom templates to use avg_speed_kmh and avg_hr.
- metadata.activity_type is replaced by metadata.sport and metadata.sub_sport.

## Customization

### Custom Report Templates

You can customize report templates by modifying the files in `visualizers/templates/`:

- `workout_report.html`: HTML report template
- `workout_report.md`: Markdown report template
- `summary_report.html`: Summary report template

### Adding New Analysis Metrics

Extend the `WorkoutAnalyzer` class in `analyzers/workout_analyzer.py`:

```python
def analyze_custom_metric(self, workout: WorkoutData) -> dict:
    """Analyze custom metric."""
    # Your custom analysis logic here
    return {'custom_metric': value}
```

### Custom Chart Types

Add new chart types in `visualizers/chart_generator.py`:

```python
def generate_custom_chart(self, workout: WorkoutData, analysis: dict) -> str:
    """Generate custom chart."""
    # Your custom chart logic here
    return chart_path
```

## Troubleshooting

### Common Issues

**File Not Found Errors**
- Ensure file paths are correct and files exist
- Check file permissions

**Garmin Connect Authentication**
- Verify GARMIN_EMAIL and GARMIN_PASSWORD environment variables (or entries in your .env) are set; fallback from GARMIN_USERNAME logs a one-time deprecation warning via [get_garmin_credentials()](config/settings.py:31)
- Check internet connection
- Ensure Garmin Connect account is active

**Missing Dependencies**
- Run `pip install -r requirements.txt`
- For PDF support: `pip install weasyprint`

**Performance Issues**
- For large datasets, use batch processing
- Consider using `--summary` flag for multiple files

### Debug Mode

Enable verbose logging for troubleshooting:
```bash
python main.py --verbose --file workout.fit --report
```

## API Reference

### Core Classes

- `WorkoutData`: Main workout data structure
- `WorkoutAnalyzer`: Performs workout analysis
- `ChartGenerator`: Creates visualizations
- `ReportGenerator`: Generates reports
- `GarminClient`: Handles Garmin Connect integration

### Example API Usage

```python
from pathlib import Path
from config.settings import Settings
from parsers.file_parser import FileParser
from analyzers.workout_analyzer import WorkoutAnalyzer

# Initialize components
settings = Settings('config/config.yaml')
parser = FileParser()
analyzer = WorkoutAnalyzer(settings.zones)

# Parse and analyze workout
workout = parser.parse_file(Path('workout.fit'))
analysis = analyzer.analyze_workout(workout)

# Access results
print(f"Average Power: {analysis['summary']['avg_power']} W")
print(f"Training Stress Score: {analysis['summary']['training_stress_score']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review log files in `output/logs/`
- Open an issue on GitHub