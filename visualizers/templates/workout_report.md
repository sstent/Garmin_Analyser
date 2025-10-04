# Workout Report: {{ workout.metadata.activity_name }}

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

---

*Report generated on {{ report.generated_at }} using {{ report.tool }} v{{ report.version }}*