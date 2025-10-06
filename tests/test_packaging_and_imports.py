import subprocess
import sys
import zipfile
import tempfile
import shutil
import pytest
from pathlib import Path

# Since we are running this from the tests directory, we need to add the project root to the path
# to import the parser.
sys.path.insert(0, str(Path(__file__).parent.parent))

from parsers.file_parser import FileParser


PROJECT_ROOT = Path(__file__).parent.parent
DIST_DIR = PROJECT_ROOT / "dist"


def run_command(command, cwd=PROJECT_ROOT, venv_python=None):
    """Helper to run a command and check for success."""
    env = None
    if venv_python:
        env = {"PATH": f"{Path(venv_python).parent}:{subprocess.os.environ['PATH']}"}

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        shell=isinstance(command, str),
    )
    assert result.returncode == 0, f"Command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
    return result


@pytest.fixture(scope="module")
def wheel_path():
    """Builds the wheel and yields its path."""
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    
    # Build the wheel
    run_command([sys.executable, "setup.py", "sdist", "bdist_wheel"])
    
    wheel_files = list(DIST_DIR.glob("*.whl"))
    assert len(wheel_files) > 0, "Wheel file not found in dist/ directory."
    
    return wheel_files[0]


def test_editable_install_validation():
    """Validates that an editable install is successful and the CLI script works."""
    # Use the current python executable for pip
    pip_executable = Path(sys.executable).parent / "pip"
    run_command([str(pip_executable), "install", "-e", "."])
    
    # Check if the CLI script runs
    cli_executable = Path(sys.executable).parent / "garmin-analyzer-cli"
    run_command([str(cli_executable), "--help"])


def test_wheel_distribution_validation(wheel_path):
    """Validates the wheel build and a clean installation."""
    # 1. Inspect wheel contents for templates
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        namelist = zf.namelist()
        template_paths = [
            "garmin_analyser/visualizers/templates/workout_report.html",
            "garmin_analyser/visualizers/templates/workout_report.md",
            "garmin_analyser/visualizers/templates/summary_report.html",
        ]
        for path in template_paths:
            assert any(p.endswith(path) for p in namelist), f"Template '{path}' not found in wheel."

    # 2. Create a clean environment and install the wheel
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create venv
        run_command([sys.executable, "-m", "venv", str(temp_path / "venv")])
        
        venv_python = temp_path / "venv" / "bin" / "python"
        venv_pip = temp_path / "venv" / "bin" / "pip"

        # Install wheel into venv
        run_command([str(venv_pip), "install", str(wheel_path)])
        
        # 3. Execute console scripts from the new venv
        run_command("garmin-analyzer-cli --help", venv_python=venv_python)
        run_command("garmin-analyzer --help", venv_python=venv_python)


def test_unsupported_file_types_raise_not_implemented_error():
    """Tests that parsing .tcx and .gpx files raises NotImplementedError."""
    parser = FileParser()
    
    with pytest.raises(NotImplementedError):
        parser.parse_file(PROJECT_ROOT / "tests" / "dummy.tcx")
        
    with pytest.raises(NotImplementedError):
        parser.parse_file(PROJECT_ROOT / "tests" / "dummy.gpx")
