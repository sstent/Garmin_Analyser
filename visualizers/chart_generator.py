"""Chart generator for workout data visualization."""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..models.workout import WorkoutData

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate various charts and visualizations for workout data."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize chart generator.
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = output_dir or Path('charts')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_workout_charts(self, workout: WorkoutData, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate all workout charts.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results from WorkoutAnalyzer
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        charts = {}
        
        # Time series charts
        charts['power_time_series'] = self._create_power_time_series(workout)
        charts['heart_rate_time_series'] = self._create_heart_rate_time_series(workout)
        charts['speed_time_series'] = self._create_speed_time_series(workout)
        charts['elevation_time_series'] = self._create_elevation_time_series(workout)
        
        # Distribution charts
        charts['power_distribution'] = self._create_power_distribution(workout, analysis)
        charts['heart_rate_distribution'] = self._create_heart_rate_distribution(workout, analysis)
        charts['speed_distribution'] = self._create_speed_distribution(workout, analysis)
        
        # Zone charts
        charts['power_zones'] = self._create_power_zones_chart(analysis)
        charts['heart_rate_zones'] = self._create_heart_rate_zones_chart(analysis)
        
        # Correlation charts
        charts['power_vs_heart_rate'] = self._create_power_vs_heart_rate(workout)
        charts['power_vs_speed'] = self._create_power_vs_speed(workout)
        
        # Summary dashboard
        charts['workout_dashboard'] = self._create_workout_dashboard(workout, analysis)
        
        return charts
    
    def _create_power_time_series(self, workout: WorkoutData) -> str:
        """Create power vs time chart.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Path to saved chart
        """
        if not workout.power or not workout.power.power_values:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        power_values = workout.power.power_values
        time_minutes = np.arange(len(power_values)) / 60
        
        ax.plot(time_minutes, power_values, linewidth=0.5, alpha=0.8)
        ax.axhline(y=workout.power.avg_power, color='r', linestyle='--', 
                   label=f'Avg: {workout.power.avg_power:.0f}W')
        ax.axhline(y=workout.power.max_power, color='g', linestyle='--', 
                   label=f'Max: {workout.power.max_power:.0f}W')
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'power_time_series.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_heart_rate_time_series(self, workout: WorkoutData) -> str:
        """Create heart rate vs time chart.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Path to saved chart
        """
        if not workout.heart_rate or not workout.heart_rate.heart_rate_values:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        hr_values = workout.heart_rate.heart_rate_values
        time_minutes = np.arange(len(hr_values)) / 60
        
        ax.plot(time_minutes, hr_values, linewidth=0.5, alpha=0.8, color='red')
        ax.axhline(y=workout.heart_rate.avg_hr, color='darkred', linestyle='--', 
                   label=f'Avg: {workout.heart_rate.avg_hr:.0f} bpm')
        ax.axhline(y=workout.heart_rate.max_hr, color='darkgreen', linestyle='--', 
                   label=f'Max: {workout.heart_rate.max_hr:.0f} bpm')
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Heart Rate (bpm)')
        ax.set_title('Heart Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'heart_rate_time_series.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_speed_time_series(self, workout: WorkoutData) -> str:
        """Create speed vs time chart.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Path to saved chart
        """
        if not workout.speed or not workout.speed.speed_values:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        speed_values = workout.speed.speed_values
        time_minutes = np.arange(len(speed_values)) / 60
        
        ax.plot(time_minutes, speed_values, linewidth=0.5, alpha=0.8, color='blue')
        ax.axhline(y=workout.speed.avg_speed, color='darkblue', linestyle='--', 
                   label=f'Avg: {workout.speed.avg_speed:.1f} km/h')
        ax.axhline(y=workout.speed.max_speed, color='darkgreen', linestyle='--', 
                   label=f'Max: {workout.speed.max_speed:.1f} km/h')
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'speed_time_series.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_elevation_time_series(self, workout: WorkoutData) -> str:
        """Create elevation vs time chart.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Path to saved chart
        """
        if not workout.elevation or not workout.elevation.elevation_values:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        elevation_values = workout.elevation.elevation_values
        time_minutes = np.arange(len(elevation_values)) / 60
        
        ax.plot(time_minutes, elevation_values, linewidth=1, alpha=0.8, color='brown')
        ax.fill_between(time_minutes, elevation_values, alpha=0.3, color='brown')
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Elevation Profile')
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'elevation_time_series.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_power_distribution(self, workout: WorkoutData, analysis: Dict[str, Any]) -> str:
        """Create power distribution histogram.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results
            
        Returns:
            Path to saved chart
        """
        if not workout.power or not workout.power.power_values:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        power_values = workout.power.power_values
        
        ax.hist(power_values, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.axvline(x=workout.power.avg_power, color='red', linestyle='--', 
                   label=f'Avg: {workout.power.avg_power:.0f}W')
        
        ax.set_xlabel('Power (W)')
        ax.set_ylabel('Frequency')
        ax.set_title('Power Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'power_distribution.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_heart_rate_distribution(self, workout: WorkoutData, analysis: Dict[str, Any]) -> str:
        """Create heart rate distribution histogram.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results
            
        Returns:
            Path to saved chart
        """
        if not workout.heart_rate or not workout.heart_rate.heart_rate_values:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        hr_values = workout.heart_rate.heart_rate_values
        
        ax.hist(hr_values, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(x=workout.heart_rate.avg_hr, color='darkred', linestyle='--', 
                   label=f'Avg: {workout.heart_rate.avg_hr:.0f} bpm')
        
        ax.set_xlabel('Heart Rate (bpm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Heart Rate Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'heart_rate_distribution.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_speed_distribution(self, workout: WorkoutData, analysis: Dict[str, Any]) -> str:
        """Create speed distribution histogram.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results
            
        Returns:
            Path to saved chart
        """
        if not workout.speed or not workout.speed.speed_values:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        speed_values = workout.speed.speed_values
        
        ax.hist(speed_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=workout.speed.avg_speed, color='darkblue', linestyle='--', 
                   label=f'Avg: {workout.speed.avg_speed:.1f} km/h')
        
        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('Frequency')
        ax.set_title('Speed Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'speed_distribution.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_power_zones_chart(self, analysis: Dict[str, Any]) -> str:
        """Create power zones pie chart.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Path to saved chart
        """
        if 'power_analysis' not in analysis or 'power_zones' not in analysis['power_analysis']:
            return None
        
        power_zones = analysis['power_analysis']['power_zones']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = list(power_zones.keys())
        sizes = list(power_zones.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Time in Power Zones')
        
        filepath = self.output_dir / 'power_zones.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_heart_rate_zones_chart(self, analysis: Dict[str, Any]) -> str:
        """Create heart rate zones pie chart.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Path to saved chart
        """
        if 'heart_rate_analysis' not in analysis or 'hr_zones' not in analysis['heart_rate_analysis']:
            return None
        
        hr_zones = analysis['heart_rate_analysis']['hr_zones']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = list(hr_zones.keys())
        sizes = list(hr_zones.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Time in Heart Rate Zones')
        
        filepath = self.output_dir / 'heart_rate_zones.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_power_vs_heart_rate(self, workout: WorkoutData) -> str:
        """Create power vs heart rate scatter plot.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Path to saved chart
        """
        if (not workout.power or not workout.power.power_values or
            not workout.heart_rate or not workout.heart_rate.heart_rate_values):
            return None
        
        power_values = workout.power.power_values
        hr_values = workout.heart_rate.heart_rate_values
        
        # Align arrays
        min_len = min(len(power_values), len(hr_values))
        if min_len == 0:
            return None
        
        power_values = power_values[:min_len]
        hr_values = hr_values[:min_len]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(power_values, hr_values, alpha=0.5, s=1)
        
        # Add trend line
        z = np.polyfit(power_values, hr_values, 1)
        p = np.poly1d(z)
        ax.plot(power_values, p(power_values), "r--", alpha=0.8)
        
        ax.set_xlabel('Power (W)')
        ax.set_ylabel('Heart Rate (bpm)')
        ax.set_title('Power vs Heart Rate')
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'power_vs_heart_rate.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_power_vs_speed(self, workout: WorkoutData) -> str:
        """Create power vs speed scatter plot.
        
        Args:
            workout: WorkoutData object
            
        Returns:
            Path to saved chart
        """
        if (not workout.power or not workout.power.power_values or
            not workout.speed or not workout.speed.speed_values):
            return None
        
        power_values = workout.power.power_values
        speed_values = workout.speed.speed_values
        
        # Align arrays
        min_len = min(len(power_values), len(speed_values))
        if min_len == 0:
            return None
        
        power_values = power_values[:min_len]
        speed_values = speed_values[:min_len]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(power_values, speed_values, alpha=0.5, s=1)
        
        # Add trend line
        z = np.polyfit(power_values, speed_values, 1)
        p = np.poly1d(z)
        ax.plot(power_values, p(power_values), "r--", alpha=0.8)
        
        ax.set_xlabel('Power (W)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Power vs Speed')
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / 'power_vs_speed.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _create_workout_dashboard(self, workout: WorkoutData, analysis: Dict[str, Any]) -> str:
        """Create comprehensive workout dashboard.
        
        Args:
            workout: WorkoutData object
            analysis: Analysis results
            
        Returns:
            Path to saved chart
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Power Over Time', 'Heart Rate Over Time',
                           'Speed Over Time', 'Elevation Profile',
                           'Power Distribution', 'Heart Rate Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Power time series
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
            time_minutes = np.arange(len(power_values)) / 60
            fig.add_trace(
                go.Scatter(x=time_minutes, y=power_values, name='Power', line=dict(color='orange')),
                row=1, col=1
            )
        
        # Heart rate time series
        if workout.heart_rate and workout.heart_rate.heart_rate_values:
            hr_values = workout.heart_rate.heart_rate_values
            time_minutes = np.arange(len(hr_values)) / 60
            fig.add_trace(
                go.Scatter(x=time_minutes, y=hr_values, name='Heart Rate', line=dict(color='red')),
                row=1, col=2
            )
        
        # Speed time series
        if workout.speed and workout.speed.speed_values:
            speed_values = workout.speed.speed_values
            time_minutes = np.arange(len(speed_values)) / 60
            fig.add_trace(
                go.Scatter(x=time_minutes, y=speed_values, name='Speed', line=dict(color='blue')),
                row=2, col=1
            )
        
        # Elevation profile
        if workout.elevation and workout.elevation.elevation_values:
            elevation_values = workout.elevation.elevation_values
            time_minutes = np.arange(len(elevation_values)) / 60
            fig.add_trace(
                go.Scatter(x=time_minutes, y=elevation_values, name='Elevation', line=dict(color='brown')),
                row=2, col=2
            )
        
        # Power distribution
        if workout.power and workout.power.power_values:
            power_values = workout.power.power_values
            fig.add_trace(
                go.Histogram(x=power_values, name='Power Distribution', nbinsx=50),
                row=3, col=1
            )
        
        # Heart rate distribution
        if workout.heart_rate and workout.heart_rate.heart_rate_values:
            hr_values = workout.heart_rate.heart_rate_values
            fig.add_trace(
                go.Histogram(x=hr_values, name='HR Distribution', nbinsx=30),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Workout Dashboard - {workout.metadata.activity_name}",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (minutes)", row=1, col=1)
        fig.update_yaxes(title_text="Power (W)", row=1, col=1)
        fig.update_xaxes(title_text="Time (minutes)", row=1, col=2)
        fig.update_yaxes(title_text="Heart Rate (bpm)", row=1, col=2)
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="Speed (km/h)", row=2, col=1)
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Elevation (m)", row=2, col=2)
        fig.update_xaxes(title_text="Power (W)", row=3, col=1)
        fig.update_xaxes(title_text="Heart Rate (bpm)", row=3, col=2)
        
        filepath = self.output_dir / 'workout_dashboard.html'
        fig.write_html(str(filepath))
        
        return str(filepath)