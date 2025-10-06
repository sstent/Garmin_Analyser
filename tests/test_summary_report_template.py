import pytest
from visualizers.report_generator import ReportGenerator


class MockWorkoutData:
    def __init__(self, summary_dict):
        self.metadata = summary_dict.get("metadata", {})
        self.summary = summary_dict.get("summary", {})


@pytest.fixture
def report_generator():
    return ReportGenerator()


def _get_full_summary(date="2024-01-01"):
    return {
        "metadata": {
            "start_time": f"{date} 10:00:00",
            "sport": "Cycling",
            "sub_sport": "Road",
            "total_duration": 3600,
            "total_distance_km": 30.0,
            "avg_speed_kmh": 30.0,
            "avg_hr": 150,
        },
        "summary": {"np": 220, "if": 0.85, "tss": 60},
    }


def _get_partial_summary(date="2024-01-02"):
    """Summary missing NP, IF, and TSS."""
    return {
        "metadata": {
            "start_time": f"{date} 09:00:00",
            "sport": "Cycling",
            "sub_sport": "Indoor",
            "total_duration": 1800,
            "total_distance_km": 15.0,
            "avg_speed_kmh": 30.0,
            "avg_hr": 145,
        },
        "summary": {},  # Missing optional keys
    }


def test_summary_report_generation_with_full_data(report_generator, tmp_path):
    workouts = [MockWorkoutData(_get_full_summary())]
    analyses = [_get_full_summary()]
    output_file = tmp_path / "summary.html"

    html_output = report_generator.generate_summary_report(
        workouts, analyses, format="html"
    )
    output_file.write_text(html_output)

    assert output_file.exists()
    content = output_file.read_text()
    
    assert "<h2>Workout Summary</h2>" in content
    assert "<th>Date</th>" in content
    assert "<th>Sport</th>" in content
    assert "<th>Duration</th>" in content
    assert "<th>Distance (km)</th>" in content
    assert "<th>Avg Speed (km/h)</th>" in content
    assert "<th>Avg HR</th>" in content
    assert "<th>NP</th>" in content
    assert "<th>IF</th>" in content
    assert "<th>TSS</th>" in content
    
    assert "<td>2024-01-01 10:00:00</td>" in content
    assert "<td>Cycling (Road)</td>" in content
    assert "<td>01:00:00</td>" in content
    assert "<td>30.0</td>" in content
    assert "<td>150</td>" in content
    assert "<td>220</td>" in content
    assert "<td>0.85</td>" in content
    assert "<td>60</td>" in content

def test_summary_report_gracefully_handles_missing_data(report_generator, tmp_path):
    workouts = [
        MockWorkoutData(_get_full_summary()),
        MockWorkoutData(_get_partial_summary()),
    ]
    analyses = [_get_full_summary(), _get_partial_summary()]
    output_file = tmp_path / "summary_mixed.html"

    html_output = report_generator.generate_summary_report(
        workouts, analyses, format="html"
    )
    output_file.write_text(html_output)

    assert output_file.exists()
    content = output_file.read_text()

    # Check that the table structure is there
    assert content.count("<tr>") == 3  # Header + 2 data rows
    
    # Check full data row
    assert "<td>220</td>" in content
    assert "<td>0.85</td>" in content
    assert "<td>60</td>" in content
    
    # Check partial data row - should have empty cells for missing data
    assert "<td>2024-01-02 09:00:00</td>" in content
    assert "<td>Cycling (Indoor)</td>" in content
    
    # Locate the row for the partial summary to check for empty cells
    # A bit brittle, but good enough for this test
    rows = content.split("<tr>")
    partial_row = [r for r in rows if "2024-01-02" in r][0]
    cells = partial_row.split("<td>")
    
    # NP, IF, TSS are the last 3 cells. They should be empty or just contain whitespace.
    assert "</td>" * 3 in partial_row.replace(" ", "").replace("\n", "")
    assert "<td></td>" * 3 in partial_row.replace(" ", "").replace("\n", "")