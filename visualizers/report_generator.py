"""Report generator for creating comprehensive workout reports."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import jinja2
import pandas as pd
from markdown import markdown
from weasyprint import HTML, CSS
import json

from models.workout import WorkoutData

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive workout reports in various formats."""
    
    def __init__(self, template_dir: Path = None):
        """Initialize report generator.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = template_dir or Path(__file__).parent / 'templates'
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Add custom filters
        self.jinja_env.filters['format_duration'] = self._format_duration
        self.jinja_env.filters['format_distance'] = self._format_distance
        self.jinja_env.filters['format_speed'] = self._format_speed
        self.jinja_env.filters['format_power'] = self._format_power
        self.jinja_env.filters['format_heart_rate'] = self._format_heart_rate
    
    def generate_workout_report(self, workout: WorkoutData, analysis: Dict[str, Any],
                               format: str = 'html') -> str:
        """Generate comprehensive workout report.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results from WorkoutAnalyzer
            format: Report format ('html', 'pdf', 'markdown')
            
        Returns:
            Rendered report content as a string (for html/markdown) or path to PDF file.
        """
        # Prepare report data
        report_data = self._prepare_report_data(workout, analysis)
        
        # Generate report based on format
        if format == 'html':
            return self._generate_html_report(report_data)
        elif format == 'pdf':
            return self._generate_pdf_report(report_data, workout.metadata.activity_name)
        elif format == 'markdown':
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _prepare_report_data(self, workout: WorkoutData, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for report generation.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results
            
        Returns:
            Dictionary with report data
        """
        # Normalize and alias data for template compatibility
        summary = analysis.get('summary', {})
        summary['avg_speed'] = summary.get('avg_speed_kmh')
        summary['avg_heart_rate'] = summary.get('avg_hr')

        power_analysis = analysis.get('power_analysis', {})
        if 'avg_power' not in power_analysis and 'avg_power' in summary:
            power_analysis['avg_power'] = summary['avg_power']
        if 'max_power' not in power_analysis and 'max_power' in summary:
            power_analysis['max_power'] = summary['max_power']

        heart_rate_analysis = analysis.get('heart_rate_analysis', {})
        if 'avg_hr' not in heart_rate_analysis and 'avg_hr' in summary:
            heart_rate_analysis['avg_hr'] = summary['avg_hr']
        if 'max_hr' not in heart_rate_analysis and 'max_hr' in summary:
            heart_rate_analysis['max_hr'] = summary['max_hr']
        # For templates using avg_heart_rate
        heart_rate_analysis['avg_heart_rate'] = heart_rate_analysis.get('avg_hr')
        heart_rate_analysis['max_heart_rate'] = heart_rate_analysis.get('max_hr')


        speed_analysis = analysis.get('speed_analysis', {})
        speed_analysis['avg_speed'] = speed_analysis.get('avg_speed_kmh')
        speed_analysis['max_speed'] = speed_analysis.get('max_speed_kmh')


        report_context = {
            "workout": {
                "metadata": workout.metadata,
                "summary": summary,
                "power_analysis": power_analysis,
                "heart_rate_analysis": heart_rate_analysis,
                "speed_analysis": speed_analysis,
                "elevation_analysis": analysis.get("elevation_analysis", {}),
                "intervals": analysis.get("intervals", []),
                "zones": analysis.get("zones", {}),
                "efficiency": analysis.get("efficiency", {}),
            },
            "report": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "tool": "Garmin Analyser",
            },
        }

        # Add minute-by-minute aggregation if data is available
        if workout.df is not None and not workout.df.empty:
            report_context["minute_by_minute"] = self._aggregate_minute_by_minute(
                workout.df, analysis
            )
        return report_context
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report.
        
        Args:
            report_data: Report data
            
        Returns:
            Rendered HTML content as a string.
        """
        template = self.jinja_env.get_template('workout_report.html')
        html_content = template.render(**report_data)
        
        # In a real application, you might save this to a file or return it directly
        # For testing, we return the content directly
        return html_content
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], activity_name: str) -> str:
        """Generate PDF report.
        
        Args:
            report_data: Report data
            activity_name: Name of the activity for the filename.
            
        Returns:
            Path to generated PDF report.
        """
        html_content = self._generate_html_report(report_data)
        
        output_dir = Path('reports')
        output_dir.mkdir(exist_ok=True)
        
        # Sanitize activity_name for filename
        sanitized_activity_name = "".join(
            [c if c.isalnum() or c in (' ', '-', '_') else '_' for c in activity_name]
        ).replace(' ', '_')
        
        pdf_path = output_dir / f"{sanitized_activity_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        HTML(string=html_content).write_pdf(str(pdf_path))
        
        return str(pdf_path)
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown report.
        
        Args:
            report_data: Report data
            
        Returns:
            Rendered Markdown content as a string.
        """
        template = self.jinja_env.get_template('workout_report.md')
        markdown_content = template.render(**report_data)
        
        # In a real application, you might save this to a file or return it directly
        # For testing, we return the content directly
        return markdown_content
    
    def generate_summary_report(self, workouts: List[WorkoutData],
                               analyses: List[Dict[str, Any]],
                               format: str = 'html') -> str:
        """Generate summary report for multiple workouts.
        
        Args:
            workouts: List of WorkoutData objects
            analyses: List of analysis results
            format: Report format ('html', 'pdf', 'markdown')
            
        Returns:
            Rendered summary report content as a string (for html/markdown) or path to PDF file.
        """
        # Aggregate data
        summary_data = self._aggregate_workout_data(workouts, analyses)
        
        # Generate report based on format
        if format == 'html':
            template = self.jinja_env.get_template("summary_report.html")
            return template.render(**summary_data)
        elif format == 'pdf':
            html_content = self._generate_summary_html_report(summary_data)
            output_dir = Path('reports')
            output_dir.mkdir(exist_ok=True)
            pdf_path = output_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            HTML(string=html_content).write_pdf(str(pdf_path))
            return str(pdf_path)
        elif format == 'markdown':
            template = self.jinja_env.get_template('summary_report.md')
            return template.render(**summary_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_summary_html_report(self, report_data: Dict[str, Any]) -> str:
        """Helper to generate HTML for summary report, used by PDF generation."""
        template = self.jinja_env.get_template('summary_report.html')
        return template.render(**report_data)
    
    def _aggregate_workout_data(self, workouts: List[WorkoutData], 
                              analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data from multiple workouts.
        
        Args:
            workouts: List of WorkoutData objects
            analyses: List of analysis results
            
        Returns:
            Dictionary with aggregated data
        """
        # Create DataFrame for analysis
        workout_data = []
        
        for workout, analysis in zip(workouts, analyses):
            data = {
                'date': workout.metadata.start_time,
                'activity_type': workout.metadata.sport or workout.metadata.activity_type,
                'duration_minutes': analysis.get('summary', {}).get('duration_minutes', 0),
                'distance_km': analysis.get('summary', {}).get('distance_km', 0),
                'avg_power': analysis.get('summary', {}).get('avg_power', 0),
                'avg_heart_rate': analysis.get('summary', {}).get('avg_hr', 0),
                'avg_speed': analysis.get('summary', {}).get('avg_speed_kmh', 0),
                'elevation_gain': analysis.get('summary', {}).get('elevation_gain_m', 0),
                'calories': analysis.get('summary', {}).get('calories', 0),
                'tss': analysis.get('summary', {}).get('training_stress_score', 0)
            }
            workout_data.append(data)
        
        df = pd.DataFrame(workout_data)
        
        # Calculate aggregations
        aggregations = {
            'total_workouts': len(workouts),
            'total_duration_hours': df['duration_minutes'].sum() / 60,
            'total_distance_km': df['distance_km'].sum(),
            'total_elevation_m': df['elevation_gain'].sum(),
            'total_calories': df['calories'].sum(),
            'avg_workout_duration': df['duration_minutes'].mean(),
            'avg_power': df['avg_power'].mean(),
            'avg_heart_rate': df['avg_heart_rate'].mean(),
            'avg_speed': df['avg_speed'].mean(),
            'total_tss': df['tss'].sum(),
            'weekly_tss': df['tss'].sum() / 4,  # Assuming 4 weeks
            'workouts_by_type': df['activity_type'].value_counts().to_dict(),
            'weekly_volume': df.groupby(pd.Grouper(key='date', freq='W'))['duration_minutes'].sum().to_dict()
        }
        
        return {
            'workouts': workouts,
            'analyses': analyses,
            'aggregations': aggregations,
            'report': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'tool': 'Garmin Analyser'
            }
        }
    
    def _aggregate_minute_by_minute(
        self, df: pd.DataFrame, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Aggregate workout data into minute-by-minute summaries.

        Args:
            df: Workout DataFrame.
            analysis: Analysis results.

        Returns:
            A list of dictionaries, each representing one minute of the workout.
        """
        if "timestamp" not in df.columns:
            return []

        df = df.copy()
        df["elapsed_time"] = (
            df["timestamp"] - df["timestamp"].iloc[0]
        ).dt.total_seconds()
        df["minute_index"] = (df["elapsed_time"] // 60).astype(int)

        agg_rules = {}
        if "speed" in df.columns:
            agg_rules["avg_speed_kmh"] = ("speed", "mean")
        if "cadence" in df.columns:
            agg_rules["avg_cadence"] = ("cadence", "mean")
        if "heart_rate" in df.columns:
            agg_rules["avg_hr"] = ("heart_rate", "mean")
            agg_rules["max_hr"] = ("heart_rate", "max")
        if "power" in df.columns:
            agg_rules["avg_real_power"] = ("power", "mean")
        elif "estimated_power" in df.columns:
            agg_rules["avg_power_estimate"] = ("estimated_power", "mean")

        if not agg_rules:
            return []

        minute_stats = df.groupby("minute_index").agg(**agg_rules).reset_index()

        # Distance and elevation require special handling
        if "distance" in df.columns:
            minute_stats["distance_km"] = (
                df.groupby("minute_index")["distance"]
                .apply(lambda x: (x.max() - x.min()) / 1000.0)
                .values
            )
        if "altitude" in df.columns:
            minute_stats["elevation_change"] = (
                df.groupby("minute_index")["altitude"]
                .apply(lambda x: x.iloc[-1] - x.iloc[0] if not x.empty else 0)
                .values
            )
        if "gradient" in df.columns:
            minute_stats["avg_gradient"] = (
                df.groupby("minute_index")["gradient"].mean().values
            )

        # Convert to km/h if speed is in m/s
        if "avg_speed_kmh" in minute_stats.columns:
            minute_stats["avg_speed_kmh"] *= 3.6

        # Round and format
        for col in minute_stats.columns:
            if minute_stats[col].dtype == "float64":
                minute_stats[col] = minute_stats[col].round(2)

        return minute_stats.to_dict("records")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if pd.isna(seconds):
            return ""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _format_distance(self, meters: float) -> str:
        """Format distance in meters to human-readable format.
        
        Args:
            meters: Distance in meters
            
        Returns:
            Formatted distance string
        """
        if meters >= 1000:
            return f"{meters/1000:.2f} km"
        else:
            return f"{meters:.0f} m"
    
    def _format_speed(self, kmh: float) -> str:
        """Format speed in km/h to human-readable format.
        
        Args:
            kmh: Speed in km/h
            
        Returns:
            Formatted speed string
        """
        return f"{kmh:.1f} km/h"
    
    def _format_power(self, watts: float) -> str:
        """Format power in watts to human-readable format.
        
        Args:
            watts: Power in watts
            
        Returns:
            Formatted power string
        """
        return f"{watts:.0f} W"
    
    def _format_heart_rate(self, bpm: float) -> str:
        """Format heart rate in bpm to human-readable format.
        
        Args:
            bpm: Heart rate in bpm
            
        Returns:
            Formatted heart rate string
        """
        return f"{bpm:.0f} bpm"
    
    def create_report_templates(self):
        """Create default report templates."""
        self.template_dir.mkdir(exist_ok=True)
        
        # HTML template
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workout Report - {{ workout.metadata.activity_name }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Workout Report: {{ workout.metadata.activity_name }}</h1>
        <p><strong>Date:</strong> {{ workout.metadata.start_time }}</p>
        <p><strong>Activity Type:</strong> {{ workout.metadata.activity_type }}</p>
        
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Duration</h3>
                <div class="value">{{ workout.summary.duration_minutes|format_duration }}</div>
            </div>
            <div class="summary-card">
                <h3>Distance</h3>
                <div class="value">{{ workout.summary.distance_km|format_distance }}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Power</h3>
                <div class="value">{{ workout.summary.avg_power|format_power }}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Heart Rate</h3>
                <div class="value">{{ workout.summary.avg_heart_rate|format_heart_rate }}</div>
            </div>
            <div class="summary-card">
                <h3>Avg Speed</h3>
                <div class="value">{{ workout.summary.avg_speed_kmh|format_speed }}</div>
            </div>
            <div class="summary-card">
                <h3>Calories</h3>
                <div class="value">{{ workout.summary.calories|int }}</div>
            </div>
        </div>
        
        <h2>Detailed Analysis</h2>
        
        <h3>Power Analysis</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Power</td>
                <td>{{ workout.power_analysis.avg_power|format_power }}</td>
            </tr>
            <tr>
                <td>Maximum Power</td>
                <td>{{ workout.power_analysis.max_power|format_power }}</td>
            </tr>
            <tr>
                <td>Normalized Power</td>
                <td>{{ workout.summary.normalized_power|format_power }}</td>
            </tr>
            <tr>
                <td>Intensity Factor</td>
                <td>{{ "%.2f"|format(workout.summary.intensity_factor) }}</td>
            </tr>
        </table>
        
        <h3>Heart Rate Analysis</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Heart Rate</td>
                <td>{{ workout.heart_rate_analysis.avg_heart_rate|format_heart_rate }}</td>
            </tr>
            <tr>
                <td>Maximum Heart Rate</td>
                <td>{{ workout.heart_rate_analysis.max_heart_rate|format_heart_rate }}</td>
            </tr>
        </table>
        
        <h3>Speed Analysis</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Speed</td>
                <td>{{ workout.speed_analysis.avg_speed|format_speed }}</td>
            </tr>
            <tr>
                <td>Maximum Speed</td>
                <td>{{ workout.speed_analysis.max_speed|format_speed }}</td>
            </tr>
        </table>
        
        {% if minute_by_minute %}
        <h2>Minute-by-Minute Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Minute</th>
                    <th>Distance (km)</th>
                    <th>Avg Speed (km/h)</th>
                    <th>Avg Cadence</th>
                    <th>Avg HR</th>
                    <th>Max HR</th>
                    <th>Avg Gradient (%)</th>
                    <th>Elevation Change (m)</th>
                    <th>Avg Power (W)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in minute_by_minute %}
                <tr>
                    <td>{{ row.minute_index }}</td>
                    <td>{{ "%.2f"|format(row.distance_km) if row.distance_km is not none }}</td>
                    <td>{{ "%.1f"|format(row.avg_speed_kmh) if row.avg_speed_kmh is not none }}</td>
                    <td>{{ "%.0f"|format(row.avg_cadence) if row.avg_cadence is not none }}</td>
                    <td>{{ "%.0f"|format(row.avg_hr) if row.avg_hr is not none }}</td>
                    <td>{{ "%.0f"|format(row.max_hr) if row.max_hr is not none }}</td>
                    <td>{{ "%.1f"|format(row.avg_gradient) if row.avg_gradient is not none }}</td>
                    <td>{{ "%.1f"|format(row.elevation_change) if row.elevation_change is not none }}</td>
                    <td>{{ "%.0f"|format(row.avg_real_power or row.avg_power_estimate) if (row.avg_real_power or row.avg_power_estimate) is not none }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        <div class="footer">
            <p>Report generated on {{ report.generated_at }} using {{ report.tool }} v{{ report.version }}</p>
        </div>
    </div>
</body>
</html>"""
        
        with open(self.template_dir / 'workout_report.html', 'w') as f:
            f.write(html_template)
        
        # Markdown template
        md_template = """# Workout Report: {{ workout.metadata.activity_name }}

**Date:** {{ workout.metadata.start_time }}  
**Activity Type:** {{ workout.metadata.activity_type }}

## Summary

| Metric | Value |
|--------|--------|
| Duration | {{ workout.summary.duration_minutes|format_duration }} |
| Distance | {{ workout.summary.distance_km|format_distance }} |
| Average Power | {{ workout.summary.avg_power|format_power }} |
| Average Heart Rate | {{ workout.summary.avg_heart_rate|format_heart_rate }} |
| Average Speed | {{ workout.summary.avg_speed_kmh|format_speed }} |
| Calories | {{ workout.summary.calories|int }} |

## Detailed Analysis

### Power Analysis

- **Average Power:** {{ workout.power_analysis.avg_power|format_power }}
- **Maximum Power:** {{ workout.power_analysis.max_power|format_power }}
- **Normalized Power:** {{ workout.summary.normalized_power|format_power }}
- **Intensity Factor:** {{ "%.2f"|format(workout.summary.intensity_factor) }}

### Heart Rate Analysis

- **Average Heart Rate:** {{ workout.heart_rate_analysis.avg_heart_rate|format_heart_rate }}
- **Maximum Heart Rate:** {{ workout.heart_rate_analysis.max_heart_rate|format_heart_rate }}

### Speed Analysis

- **Average Speed:** {{ workout.speed_analysis.avg_speed|format_speed }}
- **Maximum Speed:** {{ workout.speed_analysis.max_speed|format_speed }}

{% if minute_by_minute %}
### Minute-by-Minute Analysis

| Minute | Dist (km) | Speed (km/h) | Cadence | HR | Max HR | Grad (%) | Elev (m) | Power (W) |
|--------|-----------|--------------|---------|----|--------|----------|----------|-----------|
{% for row in minute_by_minute -%}
| {{ row.minute_index }} | {{ "%.2f"|format(row.distance_km) if row.distance_km is not none }} | {{ "%.1f"|format(row.avg_speed_kmh) if row.avg_speed_kmh is not none }} | {{ "%.0f"|format(row.avg_cadence) if row.avg_cadence is not none }} | {{ "%.0f"|format(row.avg_hr) if row.avg_hr is not none }} | {{ "%.0f"|format(row.max_hr) if row.max_hr is not none }} | {{ "%.1f"|format(row.avg_gradient) if row.avg_gradient is not none }} | {{ "%.1f"|format(row.elevation_change) if row.elevation_change is not none }} | {{ "%.0f"|format(row.avg_real_power or row.avg_power_estimate) if (row.avg_real_power or row.avg_power_estimate) is not none }} |
{% endfor %}
{% endif %}

---

*Report generated on {{ report.generated_at }} using {{ report.tool }} v{{ report.version }}*"""
        
        with open(self.template_dir / 'workout_report.md', 'w') as f:
            f.write(md_template)
        
        logger.info("Report templates created successfully")