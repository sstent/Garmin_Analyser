"""Garmin Connect client for downloading workout data."""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

try:
    from garminconnect import Garmin
except ImportError:
    raise ImportError("garminconnect package required. Install with: pip install garminconnect")

from ..config.settings import GARMIN_EMAIL, GARMIN_PASSWORD, DATA_DIR

logger = logging.getLogger(__name__)


class GarminClient:
    """Client for interacting with Garmin Connect API."""
    
    def __init__(self, email: Optional[str] = None, password: Optional[str] = None):
        """Initialize Garmin client.
        
        Args:
            email: Garmin Connect email (defaults to GARMIN_EMAIL env var)
            password: Garmin Connect password (defaults to GARMIN_PASSWORD env var)
        """
        self.email = email or GARMIN_EMAIL
        self.password = password or GARMIN_PASSWORD
        
        if not self.email or not self.password:
            raise ValueError(
                "Garmin credentials not provided. Set GARMIN_EMAIL and GARMIN_PASSWORD "
                "environment variables or pass credentials to constructor."
            )
        
        self.client = None
        self._authenticated = False
    
    def authenticate(self) -> bool:
        """Authenticate with Garmin Connect.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self.client = Garmin(self.email, self.password)
            self.client.login()
            self._authenticated = True
            logger.info("Successfully authenticated with Garmin Connect")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Garmin Connect: {e}")
            self._authenticated = False
            return False
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self._authenticated and self.client is not None
    
    def get_latest_activity(self, activity_type: str = "cycling") -> Optional[Dict[str, Any]]:
        """Get the latest activity of specified type.
        
        Args:
            activity_type: Type of activity to retrieve
            
        Returns:
            Activity data dictionary or None if not found
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        try:
            activities = self.client.get_activities(0, 10)
            
            for activity in activities:
                activity_name = activity.get("activityName", "").lower()
                activity_type_garmin = activity.get("activityType", {}).get("typeKey", "").lower()
                
                # Check if this is a cycling activity
                is_cycling = (
                    "cycling" in activity_name or 
                    "bike" in activity_name or
                    "cycling" in activity_type_garmin or
                    "bike" in activity_type_garmin
                )
                
                if is_cycling:
                    logger.info(f"Found latest cycling activity: {activity.get('activityName', 'Unknown')}")
                    return activity
            
            logger.warning("No cycling activities found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest activity: {e}")
            return None
    
    def get_activity_by_id(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Get activity by ID.
        
        Args:
            activity_id: Garmin activity ID
            
        Returns:
            Activity data dictionary or None if not found
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        try:
            activity = self.client.get_activity(activity_id)
            logger.info(f"Retrieved activity: {activity.get('activityName', 'Unknown')}")
            return activity
        except Exception as e:
            logger.error(f"Failed to get activity {activity_id}: {e}")
            return None
    
    def download_activity_file(self, activity_id: str, file_format: str = "fit") -> Optional[Path]:
        """Download activity file in specified format.
        
        Args:
            activity_id: Garmin activity ID
            file_format: File format to download (fit, tcx, gpx)
            
        Returns:
            Path to downloaded file or None if download failed
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        try:
            # Create data directory if it doesn't exist
            DATA_DIR.mkdir(exist_ok=True)
            
            # Download file
            file_data = self.client.download_activity(
                activity_id, 
                dl_fmt=file_format.upper()
            )
            
            # Save to file
            filename = f"activity_{activity_id}.{file_format}"
            file_path = DATA_DIR / filename
            
            with open(file_path, "wb") as f:
                f.write(file_data)
            
            logger.info(f"Downloaded activity file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download activity {activity_id}: {e}")
            return None
    
    def download_activity_original(self, activity_id: str) -> Optional[Path]:
        """Download original activity file (usually FIT format).
        
        Args:
            activity_id: Garmin activity ID
            
        Returns:
            Path to downloaded file or None if download failed
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        try:
            # Create data directory if it doesn't exist
            DATA_DIR.mkdir(exist_ok=True)
            
            # Download original file
            file_data = self.client.download_original_activity(activity_id)
            
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                tmp_file.write(file_data)
                tmp_path = Path(tmp_file.name)
            
            # Extract zip file
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                # Find the first FIT file in the zip
                fit_files = [f for f in zip_ref.namelist() if f.lower().endswith('.fit')]
                
                if fit_files:
                    # Extract the first FIT file
                    fit_filename = fit_files[0]
                    extracted_path = DATA_DIR / f"activity_{activity_id}.fit"
                    
                    with zip_ref.open(fit_filename) as source, open(extracted_path, 'wb') as target:
                        target.write(source.read())
                    
                    # Clean up temporary zip file
                    tmp_path.unlink()
                    
                    logger.info(f"Downloaded original activity file: {extracted_path}")
                    return extracted_path
                else:
                    logger.warning("No FIT file found in downloaded archive")
                    tmp_path.unlink()
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to download original activity {activity_id}: {e}")
            return None
    
    def get_activity_summary(self, activity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed activity summary.
        
        Args:
            activity_id: Garmin activity ID
            
        Returns:
            Activity summary dictionary or None if not found
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        try:
            activity = self.client.get_activity(activity_id)
            laps = self.client.get_activity_laps(activity_id)
            
            summary = {
                "activity": activity,
                "laps": laps,
                "activity_id": activity_id
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get activity summary for {activity_id}: {e}")
            return None
    
    def get_all_cycling_workouts(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all cycling activities from Garmin Connect.
        
        Args:
            limit: Maximum number of activities to retrieve
            
        Returns:
            List of cycling activity dictionaries
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return []
        
        try:
            activities = []
            offset = 0
            batch_size = 100
            
            while offset < limit:
                batch = self.client.get_activities(offset, min(batch_size, limit - offset))
                if not batch:
                    break
                
                for activity in batch:
                    activity_name = activity.get("activityName", "").lower()
                    activity_type_garmin = activity.get("activityType", {}).get("typeKey", "").lower()
                    
                    # Check if this is a cycling activity
                    is_cycling = (
                        "cycling" in activity_name or
                        "bike" in activity_name or
                        "cycling" in activity_type_garmin or
                        "bike" in activity_type_garmin
                    )
                    
                    if is_cycling:
                        activities.append(activity)
                
                offset += len(batch)
                
                # Stop if we got fewer activities than requested
                if len(batch) < batch_size:
                    break
            
            logger.info(f"Found {len(activities)} cycling activities")
            return activities
            
        except Exception as e:
            logger.error(f"Failed to get cycling activities: {e}")
            return []
    
    def get_workout_by_id(self, workout_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific workout by ID.
        
        Args:
            workout_id: Garmin workout ID
            
        Returns:
            Workout data dictionary or None if not found
        """
        return self.get_activity_by_id(str(workout_id))
    
    def download_workout_file(self, workout_id: int, file_path: Path) -> bool:
        """Download workout file to specified path.
        
        Args:
            workout_id: Garmin workout ID
            file_path: Path to save the file
            
        Returns:
            True if download successful, False otherwise
        """
        downloaded_path = self.download_activity_original(str(workout_id))
        if downloaded_path and downloaded_path.exists():
            # Move to requested location
            downloaded_path.rename(file_path)
            return True
        return False