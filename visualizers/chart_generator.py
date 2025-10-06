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

from models.workout import WorkoutData
from models.zones import ZoneCalculator

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
        self.zone_calculator = ZoneCalculator()

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def _get_avg_max_values(self, analysis: Dict[str, Any], data_type: str, workout: WorkoutData) -> Tuple[float, float]:
        """Get avg and max values from analysis dict or compute from workout data.

        Args:
            analysis: Analysis results from WorkoutAnalyzer
            data_type: 'power', 'hr', or 'speed'
            workout: WorkoutData object

        Returns:
            Tuple of (avg_value, max_value)
        """
        if analysis and 'summary' in analysis:
            summary = analysis['summary']
            if data_type == 'power':
                avg_key, max_key = 'avg_power', 'max_power'
            elif data_type == 'hr':
                avg_key, max_key = 'avg_hr', 'max_hr'
            elif data_type == 'speed':
                avg_key, max_key = 'avg_speed_kmh', 'max_speed_kmh'
            else:
                raise ValueError(f"Unsupported data_type: {data_type}")

            avg_val = summary.get(avg_key)
            max_val = summary.get(max_key)

            if avg_val is not None and max_val is not None:
                return avg_val, max_val

        # Fallback: compute from workout data
        if data_type == 'power' and workout.power and workout.power.power_values:
            return np.mean(workout.power.power_values), np.max(workout.power.power_values)
        elif data_type == 'hr' and workout.heart_rate and workout.heart_rate.heart_rate_values:
            return np.mean(workout.heart_rate.heart_rate_values), np.max(workout.heart_rate.heart_rate_values)
        elif data_type == 'speed' and workout.speed and workout.speed.speed_values:
            return np.mean(workout.speed.speed_values), np.max(workout.speed.speed_values)

        # Default fallback
        return 0, 0

    def _get_avg_max_labels(self, data_type: str, analysis: Dict[str, Any], workout: WorkoutData) -> Tuple[str, str]:
        """Get formatted average and maximum labels for chart annotations.

        Args:
            data_type: 'power', 'hr', or 'speed'
            analysis: Analysis results from WorkoutAnalyzer
            workout: WorkoutData object

        Returns:
            Tuple of (avg_label, max_label)
        """
        avg_val, max_val = self._get_avg_max_values(analysis, data_type, workout)

        if data_type == 'power':
            avg_label = f'Avg: {avg_val:.0f}W'
            max_label = f'Max: {max_val:.0f}W'
        elif data_type == 'hr':
            avg_label = f'Avg: {avg_val:.0f} bpm'
            max_label = f'Max: {max_val:.0f} bpm'
        elif data_type == 'speed':
            avg_label = f'Avg: {avg_val:.1f} km/h'
            max_label = f'Max: {max_val:.1f} km/h'
        else:
            avg_label = f'Avg: {avg_val:.1f}'
            max_label = f'Max: {max_val:.1f}'

        return avg_label, max_label

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
        charts['power_time_series'] = self._create_power_time_series(workout, analysis, elevation_overlay=True, zone_shading=True)
        charts['heart_rate_time_series'] = self._create_heart_rate_time_series(workout, analysis, elevation_overlay=True)
        charts['speed_time_series'] = self._create_speed_time_series(workout, analysis, elevation_overlay=True)
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

    def _create_power_time_series(self, workout: WorkoutData, analysis: Dict[str, Any] = None, elevation_overlay: bool = True, zone_shading: bool = True) -> str:
        """Create power vs time chart.

        Args:
            workout: WorkoutData object
            analysis: Analysis results from WorkoutAnalyzer
            elevation_overlay: Whether to add an elevation overlay
            zone_shading: Whether to add power zone shading

        Returns:
            Path to saved chart
        """
        if not workout.power or not workout.power.power_values:
            return None

        fig, ax1 = plt.subplots(figsize=(12, 6))

        power_values = workout.power.power_values
        time_minutes = np.arange(len(power_values)) / 60

        # Plot power
        ax1.plot(time_minutes, power_values, linewidth=0.5, alpha=0.8, color='blue')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Power (W)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Add avg/max annotations from analysis or fallback
        avg_power_label, max_power_label = self._get_avg_max_labels('power', analysis, workout)
        ax1.axhline(y=self._get_avg_max_values(analysis, 'power', workout)[0], color='red', linestyle='--',
                    label=avg_power_label)
        ax1.axhline(y=self._get_avg_max_values(analysis, 'power', workout)[1], color='green', linestyle='--',
                    label=max_power_label)

        # Add power zone shading
        if zone_shading and analysis and 'power_analysis' in analysis:
            power_zones = self.zone_calculator.get_power_zones()
            # Try to get FTP from analysis, otherwise use a default or the zone calculator's default
            ftp = analysis.get('power_analysis', {}).get('ftp', 250) # Fallback to 250W if not in analysis

            # Recalculate zones based on FTP percentage
            power_zones_percent = {
                'Recovery': {'min': 0, 'max': 0.5}, # <50% FTP
                'Endurance': {'min': 0.5, 'max': 0.75}, # 50-75% FTP
                'Tempo': {'min': 0.75, 'max': 0.9}, # 75-90% FTP
                'Threshold': {'min': 0.9, 'max': 1.05}, # 90-105% FTP
                'VO2 Max': {'min': 1.05, 'max': 1.2}, # 105-120% FTP
                'Anaerobic': {'min': 1.2, 'max': 10} # >120% FTP (arbitrary max for shading)
            }

            for zone_name, zone_def_percent in power_zones_percent.items():
                min_power = ftp * zone_def_percent['min']
                max_power = ftp * zone_def_percent['max']

                # Find the corresponding ZoneDefinition to get the color
                zone_color = next((z.color for z_name, z in power_zones.items() if z_name == zone_name), 'grey')

                ax1.axhspan(min_power, max_power,
                            alpha=0.1, color=zone_color,
                            label=f'{zone_name} ({min_power:.0f}-{max_power:.0f}W)')

        # Add elevation overlay if available
        if elevation_overlay and workout.elevation and workout.elevation.elevation_values:
            # Create twin axis for elevation
            ax2 = ax1.twinx()
            elevation_values = workout.elevation.elevation_values

            # Apply light smoothing to elevation for visual stability
            # Using a simple rolling mean, NaN-safe
            elevation_smoothed = pd.Series(elevation_values).rolling(window=5, min_periods=1, center=True).mean().values

            # Align lengths (assume same sampling rate)
            min_len = min(len(power_values), len(elevation_smoothed))
            elevation_aligned = elevation_smoothed[:min_len]
            time_aligned = time_minutes[:min_len]

            ax2.fill_between(time_aligned, elevation_aligned, alpha=0.2, color='brown', label='Elevation')
            ax2.set_ylabel('Elevation (m)', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend()

        ax1.set_title('Power Over Time')
        ax1.grid(True, alpha=0.3)

        filepath = self.output_dir / 'power_time_series.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _create_heart_rate_time_series(self, workout: WorkoutData, analysis: Dict[str, Any] = None, elevation_overlay: bool = True) -> str:
        """Create heart rate vs time chart.

        Args:
            workout: WorkoutData object
            analysis: Analysis results from WorkoutAnalyzer
            elevation_overlay: Whether to add an elevation overlay

        Returns:
            Path to saved chart
        """
        if not workout.heart_rate or not workout.heart_rate.heart_rate_values:
            return None

        fig, ax1 = plt.subplots(figsize=(12, 6))

        hr_values = workout.heart_rate.heart_rate_values
        time_minutes = np.arange(len(hr_values)) / 60

        # Plot heart rate
        ax1.plot(time_minutes, hr_values, linewidth=0.5, alpha=0.8, color='red')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Heart Rate (bpm)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        # Add avg/max annotations from analysis or fallback
        avg_hr_label, max_hr_label = self._get_avg_max_labels('hr', analysis, workout)
        ax1.axhline(y=self._get_avg_max_values(analysis, 'hr', workout)[0], color='darkred', linestyle='--',
                    label=avg_hr_label)
        ax1.axhline(y=self._get_avg_max_values(analysis, 'hr', workout)[1], color='darkgreen', linestyle='--',
                    label=max_hr_label)

        # Add elevation overlay if available
        if elevation_overlay and workout.elevation and workout.elevation.elevation_values:
            # Create twin axis for elevation
            ax2 = ax1.twinx()
            elevation_values = workout.elevation.elevation_values

            # Apply light smoothing to elevation for visual stability
            elevation_smoothed = pd.Series(elevation_values).rolling(window=5, min_periods=1, center=True).mean().values

            # Align lengths (assume same sampling rate)
            min_len = min(len(hr_values), len(elevation_smoothed))
            elevation_aligned = elevation_smoothed[:min_len]
            time_aligned = time_minutes[:min_len]

            ax2.fill_between(time_aligned, elevation_aligned, alpha=0.2, color='brown', label='Elevation')
            ax2.set_ylabel('Elevation (m)', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend()

        ax1.set_title('Heart Rate Over Time')
        ax1.grid(True, alpha=0.3)

        filepath = self.output_dir / 'heart_rate_time_series.png'
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _create_speed_time_series(self, workout: WorkoutData, analysis: Dict[str, Any] = None, elevation_overlay: bool = True) -> str:
        """Create speed vs time chart.

        Args:
            workout: WorkoutData object
            analysis: Analysis results from WorkoutAnalyzer
            elevation_overlay: Whether to add an elevation overlay

        Returns:
            Path to saved chart
        """
        if not workout.speed or not workout.speed.speed_values:
            return None

        fig, ax1 = plt.subplots(figsize=(12, 6))

        speed_values = workout.speed.speed_values
        time_minutes = np.arange(len(speed_values)) / 60

        # Plot speed
        ax1.plot(time_minutes, speed_values, linewidth=0.5, alpha=0.8, color='blue')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Speed (km/h)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Add avg/max annotations from analysis or fallback
        avg_speed_label, max_speed_label = self._get_avg_max_labels('speed', analysis, workout)
        ax1.axhline(y=self._get_avg_max_values(analysis, 'speed', workout)[0], color='darkblue', linestyle='--',
                    label=avg_speed_label)
        ax1.axhline(y=self._get_avg_max_values(analysis, 'speed', workout)[1], color='darkgreen', linestyle='--',
                    label=max_speed_label)

        # Add elevation overlay if available
        if elevation_overlay and workout.elevation and workout.elevation.elevation_values:
            # Create twin axis for elevation
            ax2 = ax1.twinx()
            elevation_values = workout.elevation.elevation_values

            # Apply light smoothing to elevation for visual stability
            elevation_smoothed = pd.Series(elevation_values).rolling(window=5, min_periods=1, center=True).mean().values

            # Align lengths (assume same sampling rate)
            min_len = min(len(speed_values), len(elevation_smoothed))
            elevation_aligned = elevation_smoothed[:min_len]
            time_aligned = time_minutes[:min_len]

            ax2.fill_between(time_aligned, elevation_aligned, alpha=0.2, color='brown', label='Elevation')
            ax2.set_ylabel('Elevation (m)', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend()

        ax1.set_title('Speed Over Time')
        ax1.grid(True, alpha=0.3)

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