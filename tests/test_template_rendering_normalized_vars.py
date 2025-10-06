"""
Tests for template rendering with normalized variables.

Validates that [ReportGenerator](visualizers/report_generator.py) can render
HTML and Markdown templates using normalized keys from analysis and metadata.
"""

import pytest
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

from analyzers.workout_analyzer import WorkoutAnalyzer
from models.workout import WorkoutData, WorkoutMetadata, SpeedData, HeartRateData
from visualizers.report_generator import ReportGenerator
from tests.test_analyzer_speed_and_normalized_naming import synthetic_workout_data


@pytest.fixture
def analysis_result(synthetic_workout_data):
    """Get analysis result from synthetic workout data."""
    analyzer = WorkoutAnalyzer()
    return analyzer.analyze_workout(synthetic_workout_data)


def test_template_rendering_with_normalized_variables(synthetic_workout_data, analysis_result):
    """
    Test that HTML and Markdown templates render successfully with normalized
    and sport/sub_sport variables.

    Validates that templates can access:
    - metadata.sport and metadata.sub_sport
    - summary.avg_speed_kmh and summary.avg_hr
    """
    report_gen = ReportGenerator()

    # Test HTML template rendering
    try:
        html_output = report_gen.generate_workout_report(synthetic_workout_data, analysis_result, format='html')
        assert isinstance(html_output, str)
        assert len(html_output) > 0
        # Check that sport and sub_sport appear in rendered output
        assert synthetic_workout_data.metadata.sport in html_output
        assert synthetic_workout_data.metadata.sub_sport in html_output
        # Check that normalized keys appear (as numeric values)
        # Check that normalized keys appear (as plausible numeric values)
        assert "Average Speed</td>\n                <td>7.4 km/h" in html_output
        assert "Average Heart Rate</td>\n                <td>133 bpm" in html_output
    except Exception as e:
        pytest.fail(f"HTML template rendering failed: {e}")

    # Test Markdown template rendering
    try:
        md_output = report_gen.generate_workout_report(synthetic_workout_data, analysis_result, format='markdown')
        assert isinstance(md_output, str)
        assert len(md_output) > 0
        # Check that sport and sub_sport appear in rendered output
        assert synthetic_workout_data.metadata.sport in md_output
        assert synthetic_workout_data.metadata.sub_sport in md_output
        # Check that normalized keys appear (as numeric values)
        # Check that normalized keys appear (as plausible numeric values)
        assert "Average Speed | 7.4 km/h" in md_output
        assert "Average Heart Rate | 133 bpm" in md_output
    except Exception as e:
        pytest.fail(f"Markdown template rendering failed: {e}")