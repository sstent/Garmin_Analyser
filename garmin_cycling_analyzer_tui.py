#!/usr/bin/env python3
"""
Garmin Cycling Analyzer TUI
A modern terminal user interface for the Garmin workout analyzer.

Requirements:
pip install textual rich
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
import subprocess

try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.widgets import (
        Header, Footer, Button, Static, DataTable, Input, Select, 
        ProgressBar, Log, Tabs, TabPane, ListView, ListItem, Label,
        Collapsible, Tree, Markdown, SelectionList
    )
    from textual.screen import Screen, ModalScreen
    from textual.binding import Binding
    from textual.reactive import reactive
    from textual.message import Message
    from textual import work
    from rich.text import Text
    from rich.table import Table
    from rich.console import Console
    from rich.markdown import Markdown as RichMarkdown
except ImportError:
    print("Missing required packages. Install with:")
    print("pip install textual rich")
    sys.exit(1)

# Import the analyzer (assuming it's in the same directory)
try:
    from garmin_cycling_analyzer import GarminWorkoutAnalyzer
except ImportError:
    print("Error: Could not import GarminWorkoutAnalyzer")
    print("Make sure garmin_cycling_analyzer.py is in the same directory")
    sys.exit(1)


class ActivityListScreen(Screen):
    """Screen for displaying and selecting activities."""
    
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("r", "refresh_activities", "Refresh"),
    ]
    
    def __init__(self, analyzer: GarminWorkoutAnalyzer):
        super().__init__()
        self.analyzer = analyzer
        self.activities = []
        self.selected_activity = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("🚴 Select Activity to Analyze", classes="title"),
            Container(
                Button("Refresh Activities", id="refresh_btn"),
                Button("Download Latest", id="download_latest_btn"),
                Button("Download All", id="download_all_btn"),
                classes="button_row"
            ),
            DataTable(id="activity_table", classes="activity_table"),
            Container(
                Button("Analyze Selected", id="analyze_btn", variant="primary"),
                Button("View Report", id="view_report_btn"),
                Button("Back", id="back_btn"),
                classes="button_row"
            ),
            classes="main_container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        table = self.query_one("#activity_table", DataTable)
        table.add_columns("ID", "Name", "Type", "Date", "Distance", "Duration", "Status")
        self.refresh_activity_list()
    
    @work(exclusive=True)
    async def refresh_activity_list(self):
        """Refresh the list of activities from Garmin Connect."""
        table = self.query_one("#activity_table", DataTable)
        table.clear()
        
        # Show loading message
        table.add_row("Loading...", "", "", "", "", "", "")
        
        try:
            # Connect to Garmin if not already connected
            if not hasattr(self.analyzer, 'garmin_client'):
                success = await asyncio.get_event_loop().run_in_executor(
                    None, self.analyzer.connect_to_garmin
                )
                if not success:
                    table.clear()
                    table.add_row("Error", "Failed to connect to Garmin", "", "", "", "", "")
                    return
            
            # Get activities
            activities = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.garmin_client.get_activities, 0, 50
            )
            
            table.clear()
            
            # Filter for cycling activities
            cycling_keywords = ['cycling', 'bike', 'road_biking', 'mountain_biking', 'indoor_cycling', 'biking']
            cycling_activities = []
            
            for activity in activities:
                activity_type = activity.get('activityType', {})
                type_key = activity_type.get('typeKey', '').lower()
                type_name = str(activity_type.get('typeId', '')).lower()
                activity_name = activity.get('activityName', '').lower()
                
                if any(keyword in type_key or keyword in type_name or keyword in activity_name
                       for keyword in cycling_keywords):
                    cycling_activities.append(activity)
            
            self.activities = cycling_activities
            
            # Populate table
            for activity in cycling_activities:
                activity_id = str(activity['activityId'])
                name = activity.get('activityName', 'Unnamed')
                activity_type = activity.get('activityType', {}).get('typeKey', 'unknown')
                start_time = activity.get('startTimeLocal', 'unknown')
                distance = activity.get('distance', 0)
                distance_km = f"{distance / 1000:.1f} km" if distance else "0.0 km"
                duration = activity.get('duration', 0)
                duration_str = str(timedelta(seconds=duration)) if duration else "0:00:00"
                
                # Check if already downloaded
                data_dir = Path("data")
                existing_files = []
                if data_dir.exists():
                    existing_files = [f for f in data_dir.glob(f"{activity_id}_*")]
                
                # Check if report exists
                report_files = []
                reports_dir = Path("reports")
                if reports_dir.exists():
                    report_files = list(reports_dir.glob(f"**/*{activity_id}*.md"))
                
                status = "📊 Report" if report_files else "💾 Downloaded" if existing_files else "🌐 Online"
                
                table.add_row(
                    activity_id, name, activity_type, start_time,
                    distance_km, duration_str, status
                )
        
        except Exception as e:
            table.clear()
            table.add_row("Error", f"Failed to load activities: {str(e)}", "", "", "", "", "")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "refresh_btn":
            self.refresh_activity_list()
        elif event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "analyze_btn":
            self.analyze_selected_activity()
        elif event.button.id == "view_report_btn":
            self.view_selected_report()
        elif event.button.id == "download_latest_btn":
            self.download_latest_workout()
        elif event.button.id == "download_all_btn":
            self.download_all_workouts()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the activity table."""
        table = event.data_table
        
        # Get the cursor row (currently selected row index)
        try:
            cursor_row = table.cursor_row
            if 0 <= cursor_row < len(self.activities_list):
                self.selected_activity = self.activities_list[cursor_row]
                activity_name = self.selected_activity.get('activityName', 'Unnamed')
                self.notify(f"Selected: {activity_name}", severity="information")
            else:
                self.selected_activity = None
        except (IndexError, AttributeError):
            # Fallback: try to get activity ID from the row data and find it
            row_data = table.get_row(event.row_key)
            if len(row_data) > 0 and row_data[0] not in ["Loading...", "Error"]:
                activity_id = row_data[0]
                # Find the activity in our list
                for activity in self.activities:
                    if str(activity['activityId']) == activity_id:
                        self.selected_activity = activity
                        activity_name = activity.get('activityName', 'Unnamed')
                        self.notify(f"Selected: {activity_name}", severity="information")
                        break
                else:
                    self.selected_activity = None
    
    @work(exclusive=True)
    async def analyze_selected_activity(self):
        """Analyze the selected activity."""
        if not self.selected_activity:
            self.notify("Please select an activity first", severity="warning")
            return
        
        activity_id = self.selected_activity['activityId']
        
        # Show progress screen
        progress_screen = ProgressScreen(f"Analyzing Activity {activity_id}")
        self.app.push_screen(progress_screen)
        
        try:
            # Download workout
            progress_screen.update_status("Downloading workout data...")
            fit_file_path = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.download_specific_workout, activity_id
            )
            
            if not fit_file_path:
                progress_screen.update_status("Failed to download workout", error=True)
                await asyncio.sleep(2)
                self.app.pop_screen()
                return
            
            progress_screen.update_status("Estimating gear configuration...")
            estimated_cog = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.estimate_cog_from_cadence, fit_file_path
            )
            
            # Use default cog for indoor, or estimated for outdoor
            confirmed_cog = 14 if self.analyzer.is_indoor else estimated_cog
            
            progress_screen.update_status("Analyzing workout data...")
            analysis_data = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.analyze_fit_file, fit_file_path, confirmed_cog
            )
            
            if not analysis_data:
                progress_screen.update_status("Failed to analyze workout data", error=True)
                await asyncio.sleep(2)
                self.app.pop_screen()
                return
            
            progress_screen.update_status("Generating report...")
            report_file = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.generate_markdown_report, analysis_data, activity_id
            )
            
            progress_screen.update_status(f"Analysis complete! Report saved: {report_file}", success=True)
            await asyncio.sleep(2)
            self.app.pop_screen()
            
            # Refresh the activity list to update status
            self.refresh_activity_list()
            
            # Open the report viewer
            self.app.push_screen(ReportViewerScreen(report_file))
            
        except Exception as e:
            progress_screen.update_status(f"Error: {str(e)}", error=True)
            await asyncio.sleep(3)
            self.app.pop_screen()
    
    def view_selected_report(self):
        """View the report for the selected activity."""
        if not self.selected_activity:
            self.notify("Please select an activity first", severity="warning")
            return
        
        activity_id = self.selected_activity['activityId']
        
        # Look for existing report
        reports_dir = Path("reports")
        if not reports_dir.exists():
            self.notify("No reports directory found", severity="warning")
            return
        
        report_files = list(reports_dir.glob(f"**/*{activity_id}*.md"))
        
        if not report_files:
            self.notify(f"No report found for activity {activity_id}", severity="warning")
            return
        
        # Use the first report file found
        report_file = report_files[0]
        self.app.push_screen(ReportViewerScreen(str(report_file)))
    
    @work(exclusive=True)
    async def download_latest_workout(self):
        """Download the latest cycling workout."""
        progress_screen = ProgressScreen("Downloading Latest Workout")
        self.app.push_screen(progress_screen)
        
        try:
            progress_screen.update_status("Fetching latest cycling workout...")
            fit_file_path = await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.download_latest_workout
            )
            
            if fit_file_path:
                progress_screen.update_status(f"Downloaded: {fit_file_path}", success=True)
            else:
                progress_screen.update_status("Failed to download latest workout", error=True)
                
            await asyncio.sleep(2)
            self.app.pop_screen()
            self.refresh_activity_list()
            
        except Exception as e:
            progress_screen.update_status(f"Error: {str(e)}", error=True)
            await asyncio.sleep(3)
            self.app.pop_screen()
    
    @work(exclusive=True) 
    async def download_all_workouts(self):
        """Download all cycling workouts."""
        progress_screen = ProgressScreen("Downloading All Workouts")
        self.app.push_screen(progress_screen)
        
        try:
            progress_screen.update_status("Downloading all cycling activities...")
            await asyncio.get_event_loop().run_in_executor(
                None, self.analyzer.download_all_workouts
            )
            
            progress_screen.update_status("All workouts downloaded!", success=True)
            await asyncio.sleep(2)
            self.app.pop_screen()
            self.refresh_activity_list()
            
        except Exception as e:
            progress_screen.update_status(f"Error: {str(e)}", error=True)
            await asyncio.sleep(3)
            self.app.pop_screen()
    
    def action_refresh_activities(self) -> None:
        """Refresh activities action."""
        self.refresh_activity_list()


class ReportViewerScreen(Screen):
    """Screen for viewing workout reports."""
    
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
    ]
    
    def __init__(self, report_file: str):
        super().__init__()
        self.report_file = Path(report_file)
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(f"📊 Report: {self.report_file.name}", classes="title"),
            ScrollableContainer(
                Markdown(id="report_content"),
                classes="report_container"
            ),
            Container(
                Button("Open in Editor", id="open_editor_btn"),
                Button("Open Report Folder", id="open_folder_btn"),
                Button("Back", id="back_btn"),
                classes="button_row"
            ),
            classes="main_container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Load and display the report when mounted."""
        self.load_report()
    
    def load_report(self):
        """Load and display the report content."""
        try:
            if self.report_file.exists():
                content = self.report_file.read_text(encoding='utf-8')
                markdown_widget = self.query_one("#report_content", Markdown)
                markdown_widget.update(content)
            else:
                self.query_one("#report_content", Markdown).update(
                    f"# Error\n\nReport file not found: {self.report_file}"
                )
        except Exception as e:
            self.query_one("#report_content", Markdown).update(
                f"# Error\n\nFailed to load report: {str(e)}"
            )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "open_editor_btn":
            self.open_in_editor()
        elif event.button.id == "open_folder_btn":
            self.open_report_folder()
    
    def open_in_editor(self):
        """Open the report file in the default editor."""
        try:
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', str(self.report_file)])
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', str(self.report_file)])
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(str(self.report_file))
            else:
                self.notify("Unsupported platform for opening files", severity="warning")
        except Exception as e:
            self.notify(f"Failed to open file: {str(e)}", severity="error")
    
    def open_report_folder(self):
        """Open the report folder in the file manager."""
        try:
            folder = self.report_file.parent
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', str(folder)])
            elif sys.platform.startswith('linux'):  # Linux
                subprocess.run(['xdg-open', str(folder)])
            elif sys.platform.startswith('win'):  # Windows
                os.startfile(str(folder))
            else:
                self.notify("Unsupported platform for opening folders", severity="warning")
        except Exception as e:
            self.notify(f"Failed to open folder: {str(e)}", severity="error")


class LocalReportsScreen(Screen):
    """Screen for viewing local report files."""
    
    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("r", "refresh_reports", "Refresh"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("📊 Local Reports", classes="title"),
            Container(
                Button("Refresh", id="refresh_btn"),
                Button("Re-analyze All", id="reanalyze_btn"),
                classes="button_row"
            ),
            DataTable(id="reports_table", classes="reports_table"),
            Container(
                Button("View Selected", id="view_btn", variant="primary"),
                Button("Delete Selected", id="delete_btn", variant="error"),
                Button("Back", id="back_btn"),
                classes="button_row"
            ),
            classes="main_container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        table = self.query_one("#reports_table", DataTable)
        table.add_columns("Activity ID", "Date", "Name", "Report File", "Size")
        self.refresh_reports()
    
    def refresh_reports(self):
        """Refresh the list of local reports."""
        table = self.query_one("#reports_table", DataTable)
        table.clear()
        
        reports_dir = Path("reports")
        if not reports_dir.exists():
            table.add_row("No reports directory found", "", "", "", "")
            return
        
        # Find all markdown report files
        report_files = list(reports_dir.glob("**/*.md"))
        
        if not report_files:
            table.add_row("No reports found", "", "", "", "")
            return
        
        for report_file in sorted(report_files, key=lambda x: x.stat().st_mtime, reverse=True):
            # Extract info from filename and path
            filename = report_file.name
            
            # Try to extract activity ID from filename
            activity_id = "Unknown"
            parts = filename.split('_')
            for part in parts:
                if part.isdigit() and len(part) > 8:  # Garmin activity IDs are long
                    activity_id = part
                    break
            
            # Get file stats
            stat = report_file.stat()
            size = f"{stat.st_size / 1024:.1f} KB"
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            date_str = modified_time.strftime("%Y-%m-%d %H:%M")
            
            # Try to extract workout name from parent directory
            parent_name = report_file.parent.name
            if parent_name != "reports":
                name = parent_name
            else:
                name = filename.replace('.md', '').replace('_workout_analysis', '')
            
            table.add_row(activity_id, date_str, name, str(report_file), size)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back_btn":
            self.app.pop_screen()
        elif event.button.id == "refresh_btn":
            self.refresh_reports()
        elif event.button.id == "view_btn":
            self.view_selected_report()
        elif event.button.id == "delete_btn":
            self.delete_selected_report()
        elif event.button.id == "reanalyze_btn":
            self.reanalyze_all_workouts()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection in the reports table."""
        table = event.data_table
        row_key = event.row_key
        row_data = table.get_row(row_key)
        
        if len(row_data) > 3:
            self.selected_report_file = row_data[3]  # Report file path
    
    def view_selected_report(self):
        """View the selected report."""
        if not hasattr(self, 'selected_report_file'):
            self.notify("Please select a report first", severity="warning")
            return
        
        self.app.push_screen(ReportViewerScreen(self.selected_report_file))
    
    def delete_selected_report(self):
        """Delete the selected report."""
        if not hasattr(self, 'selected_report_file'):
            self.notify("Please select a report first", severity="warning")
            return
        
        # Show confirmation dialog
        self.app.push_screen(ConfirmDialog(
            f"Delete report?\n\n{self.selected_report_file}",
            self.confirm_delete_report
        ))
    
    def confirm_delete_report(self):
        """Confirm and delete the report."""
        try:
            report_path = Path(self.selected_report_file)
            if report_path.exists():
                report_path.unlink()
                self.notify(f"Deleted: {report_path.name}", severity="information")
                self.refresh_reports()
            else:
                self.notify("Report file not found", severity="warning")
        except Exception as e:
            self.notify(f"Failed to delete report: {str(e)}", severity="error")
    
    @work(exclusive=True)
    async def reanalyze_all_workouts(self):
        """Re-analyze all downloaded workouts."""
        progress_screen = ProgressScreen("Re-analyzing All Workouts")
        self.app.push_screen(progress_screen)
        
        try:
            analyzer = GarminWorkoutAnalyzer()
            progress_screen.update_status("Re-analyzing all downloaded activities...")
            
            await asyncio.get_event_loop().run_in_executor(
                None, analyzer.reanalyze_all_workouts
            )
            
            progress_screen.update_status("All workouts re-analyzed!", success=True)
            await asyncio.sleep(2)
            self.app.pop_screen()
            self.refresh_reports()
            
        except Exception as e:
            progress_screen.update_status(f"Error: {str(e)}", error=True)
            await asyncio.sleep(3)
            self.app.pop_screen()
    
    def action_refresh_reports(self) -> None:
        """Refresh reports action."""
        self.refresh_reports()


class ProgressScreen(ModalScreen):
    """Modal screen for showing progress of long-running operations."""
    
    def __init__(self, title: str):
        super().__init__()
        self.title = title
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static(self.title, classes="progress_title"),
            Static("Starting...", id="status_text", classes="status_text"),
            ProgressBar(id="progress_bar"),
            classes="progress_container"
        )
    
    def update_status(self, message: str, error: bool = False, success: bool = False):
        """Update the status message."""
        status_text = self.query_one("#status_text", Static)
        if error:
            status_text.update(f"❌ {message}")
        elif success:
            status_text.update(f"✅ {message}")
        else:
            status_text.update(f"⏳ {message}")


class ConfirmDialog(ModalScreen):
    """Modal dialog for confirmation."""
    
    def __init__(self, message: str, callback):
        super().__init__()
        self.message = message
        self.callback = callback
    
    def compose(self) -> ComposeResult:
        yield Container(
            Static(self.message, classes="dialog_message"),
            Container(
                Button("Yes", id="yes_btn", variant="error"),
                Button("No", id="no_btn", variant="primary"),
                classes="dialog_buttons"
            ),
            classes="dialog_container"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "yes_btn":
            self.app.pop_screen()
            if self.callback:
                self.callback()
        else:
            self.app.pop_screen()


class MainMenuScreen(Screen):
    """Main menu screen."""
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("1", "activities", "Activities"),
        ("2", "reports", "Reports"),
    ]
    
    def __init__(self, analyzer: GarminWorkoutAnalyzer):
        super().__init__()
        self.analyzer = analyzer
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("🚴 Garmin Cycling Analyzer TUI", classes="main_title"),
            Static("Select an option:", classes="subtitle"),
            Container(
                Button("1. Browse & Analyze Activities", id="activities_btn", variant="primary"),
                Button("2. View Local Reports", id="reports_btn"),
                Button("3. Settings", id="settings_btn"),
                Button("4. Quit", id="quit_btn", variant="error"),
                classes="menu_buttons"
            ),
            Static("\nKeyboard shortcuts: 1=Activities, 2=Reports, Q=Quit", classes="help_text"),
            classes="main_menu_container"
        )
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "activities_btn":
            self.action_activities()
        elif event.button.id == "reports_btn":
            self.action_reports()
        elif event.button.id == "settings_btn":
            self.action_settings()
        elif event.button.id == "quit_btn":
            self.action_quit()
    
    def action_activities(self) -> None:
        """Open activities screen."""
        self.app.push_screen(ActivityListScreen(self.analyzer))
    
    def action_reports(self) -> None:
        """Open reports screen."""
        self.app.push_screen(LocalReportsScreen())
    
    def action_settings(self) -> None:
        """Open settings screen."""
        self.notify("Settings not implemented yet", severity="information")
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class GarminTUIApp(App):
    """Main TUI application."""
    
    CSS = """
    .main_title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1;
    }
    
    .subtitle {
        text-align: center;
        margin: 1;
    }
    
    .title {
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    
    .main_container {
        margin: 1;
        height: 100%;
    }
    
    .main_menu_container {
        height: 100%;
        align: center middle;
    }
    
    .menu_buttons {
        align: center middle;
        width: 60%;
    }
    
    .menu_buttons Button {
        width: 100%;
        margin: 0 0 1 0;
    }
    
    .button_row {
        height: auto;
        margin: 1 0;
    }
    
    .button_row Button {
        margin: 0 1 0 0;
    }
    
    .activity_table, .reports_table {
        height: 70%;
        margin: 1 0;
    }
    
    .report_container {
        height: 80%;
        border: solid $primary;
        margin: 1 0;
    }
    
    .help_text {
        text-align: center;
        color: $text-muted;
        margin: 2 0;
    }
    
    .progress_container {
        width: 60;
        height: 15;
        background: $surface;
        border: solid $primary;
        align: center middle;
    }
    
    .progress_title {
        text-align: center;
        text-style: bold;
        margin: 1;
    }
    
    .status_text {
        text-align: center;
        margin: 1;
    }
    
    .dialog_container {
        width: 50;
        height: 15;
        background: $surface;
        border: solid $primary;
        align: center middle;
    }
    
    .dialog_message {
        text-align: center;
        margin: 1;
        width: 100%;
    }
    
    .dialog_buttons {
        align: center middle;
        width: 100%;
    }
    
    .dialog_buttons Button {
        margin: 0 1;
    }
    """
    
    TITLE = "Garmin Cycling Analyzer TUI"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]
    
    def on_mount(self) -> None:
        """Initialize the application."""
        # Check for .env file
        env_file = Path('.env')
        if not env_file.exists():
            self.notify("Creating .env file template. Please add your Garmin credentials.", severity="warning")
            with open('.env', 'w') as f:
                f.write("# Garmin Connect Credentials\n")
                f.write("GARMIN_USERNAME=your_username_here\n")
                f.write("GARMIN_PASSWORD=your_password_here\n")
            self.exit(message="Please edit .env file with your Garmin credentials")
            return
        
        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # Initialize analyzer
        self.analyzer = GarminWorkoutAnalyzer()
        
        # Push main menu screen
        self.push_screen(MainMenuScreen(self.analyzer))
    
    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def main():
    """Main entry point for the TUI application."""
    try:
        app = GarminTUIApp()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error running TUI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if required dependencies are available
    missing_deps = []
    
    try:
        import textual
    except ImportError:
        missing_deps.append("textual")
    
    try:
        import rich
    except ImportError:
        missing_deps.append("rich")
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    main()
