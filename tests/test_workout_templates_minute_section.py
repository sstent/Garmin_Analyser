import pytest
from visualizers.report_generator import ReportGenerator

@pytest.fixture
def report_generator():
    return ReportGenerator()

def _get_base_context():
    """Provides a minimal, valid context for rendering."""
    return {
        "workout": {
            "metadata": {
                "sport": "Cycling",
                "sub_sport": "Road",
                "start_time": "2024-01-01 10:00:00",
                "total_duration": 120,
                "total_distance_km": 5.0,
                "avg_speed_kmh": 25.0,
                "avg_hr": 150,
                "avg_power": 200,
            },
            "summary": {
                "np": 210,
                "if": 0.8,
                "tss": 30,
            },
            "zones": {},
            "charts": {},
        },
        "report": {
            "generated_at": "2024-01-01T12:00:00",
            "version": "1.0.0",
        },
    }

def test_workout_report_renders_minute_section_when_present(report_generator):
    context = _get_base_context()
    context["minute_by_minute"] = [
        {
            "minute_index": 0,
            "distance_km": 0.5,
            "avg_speed_kmh": 30.0,
            "avg_cadence": 90,
            "avg_hr": 140,
            "max_hr": 145,
            "avg_gradient": 1.0,
            "elevation_change": 5,
            "avg_real_power": 210,
            "avg_power_estimate": None,
        }
    ]

    # Test HTML
    html_output = report_generator.generate_workout_report(context, None, "html")
    assert "<h3>Minute-by-Minute Breakdown</h3>" in html_output
    assert "<th>Minute</th>" in html_output
    assert "<td>0.50</td>" in html_output  # distance_km
    assert "<td>30.0</td>" in html_output  # avg_speed_kmh
    assert "<td>140</td>" in html_output  # avg_hr
    assert "<td>210</td>" in html_output  # avg_real_power

    # Test Markdown
    md_output = report_generator.generate_workout_report(context, None, "md")
    assert "### Minute-by-Minute Breakdown" in md_output
    assert "| Minute |" in md_output
    assert "| 0.50 |" in md_output
    assert "| 30.0 |" in md_output
    assert "| 140 |" in md_output
    assert "| 210 |" in md_output


def test_workout_report_omits_minute_section_when_absent(report_generator):
    context = _get_base_context()
    # Case 1: key is absent
    context_absent = context.copy()

    html_output_absent = report_generator.generate_workout_report(
        context_absent, None, "html"
    )
    assert "<h3>Minute-by-Minute Breakdown</h3>" not in html_output_absent

    md_output_absent = report_generator.generate_workout_report(
        context_absent, None, "md"
    )
    assert "### Minute-by-Minute Breakdown" not in md_output_absent

    # Case 2: key is present but empty
    context_empty = context.copy()
    context_empty["minute_by_minute"] = []

    html_output_empty = report_generator.generate_workout_report(
        context_empty, None, "html"
    )
    assert "<h3>Minute-by-Minute Breakdown</h3>" not in html_output_empty

    md_output_empty = report_generator.generate_workout_report(
        context_empty, None, "md"
    )
    assert "### Minute-by-Minute Breakdown" not in md_output_empty