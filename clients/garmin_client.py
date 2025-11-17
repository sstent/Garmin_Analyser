"""Garmin Connect client for downloading workout data."""

import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import hashlib
from datetime import datetime

import time
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from garminconnect import Garmin
except ImportError:
    raise ImportError("garminconnect package required. Install with: pip install garminconnect")

from config.settings import get_garmin_credentials, DATA_DIR, DATABASE_URL
from db.models import ActivityDownload
from db.session import SessionLocal


logger = logging.getLogger(__name__)


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def upsert_activity_download(
    activity_id: int,
    source: str,
    file_path: Path,
    file_format: str,
    status: str,
    http_status: Optional[int] = None,
    etag: Optional[str] = None,
    last_modified: Optional[datetime] = None,
    size_bytes: Optional[int] = None,
    checksum_sha256: Optional[str] = None,
    error_message: Optional[str] = None,
    db_session: Optional[Session] = None,
):
    """Upsert an activity download record in the database."""
    if db_session is not None:
        db = db_session
        close_session = False
    else:
        db = SessionLocal()
        close_session = True

    try:
        record = db.query(ActivityDownload).filter_by(activity_id=activity_id).first()
        if record:
            record.source = source
            record.file_path = str(file_path)
            record.file_format = file_format
            record.status = status
            record.http_status = http_status
            record.etag = etag
            record.last_modified = last_modified
            record.size_bytes = size_bytes
            record.checksum_sha256 = checksum_sha256
            record.updated_at = datetime.utcnow()
            record.error_message = error_message
        else:
            record = ActivityDownload(
                activity_id=activity_id,
                source=source,
                file_path=str(file_path),
                file_format=file_format,
                status=status,
                http_status=http_status,
                etag=etag,
                last_modified=last_modified,
                size_bytes=size_bytes,
                checksum_sha256=checksum_sha256,
                downloaded_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                error_message=error_message,
            )
            db.add(record)
        db.commit()
        db.refresh(record)
    finally:
        if close_session:
            db.close()
    return record


class GarminClient:
    """Client for interacting with Garmin Connect API."""

    def __init__(self, email: Optional[str] = None, password: Optional[str] = None, db_session: Optional[Session] = None):
        """Initialize Garmin client.

        Args:
            email: Garmin Connect email (defaults to standardized accessor)
            password: Garmin Connect password (defaults to standardized accessor)
        """
        if email and password:
            self.email = email
            self.password = password
        else:
            self.email, self.password = get_garmin_credentials()

        self.db_session = db_session if db_session else SessionLocal()
        
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
    
    def download_activity_file(
        self, activity_id: str, file_format: str = "fit", force_download: bool = False
    ) -> Optional[Path]:
        """Download activity file in specified format.

        Args:
            activity_id: Garmin activity ID
            file_format: File format to download (fit, tcx, gpx, csv, original)
            force_download: If True, bypasses database checks and forces a re-download.

        Returns:
            Path to downloaded file or None if download failed
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        try:
            # Create data directory if it doesn't exist
            DATA_DIR.mkdir(exist_ok=True)

            fmt_upper = (file_format or "").upper()
            logger.debug(f"download_activity_file: requested format='{file_format}' normalized='{fmt_upper}'")
            
            if fmt_upper in {"TCX", "GPX", "CSV"}:
                # Direct format downloads supported by garminconnect
                dl_fmt = getattr(self.client.ActivityDownloadFormat, fmt_upper)
                file_data = self.client.download_activity(activity_id, dl_fmt=dl_fmt)
                
                # Save to file using lowercase extension
                filename = f"activity_{activity_id}.{fmt_upper.lower()}"
                file_path = DATA_DIR / filename
                
                with open(file_path, "wb") as f:
                    f.write(file_data)
                
                logger.info(f"Downloaded activity file: {file_path}")
                return file_path

            # FIT is not a direct dl_fmt in some client versions; use ORIGINAL to obtain ZIP and extract .fit
            if fmt_upper in {"FIT", "ORIGINAL"} or file_format.lower() == "fit":
                fit_path = self.download_activity_original(
                    activity_id, force_download=force_download
                )
                return fit_path

            logger.error(f"Unsupported download format '{file_format}'. Valid: GPX, TCX, ORIGINAL, CSV")
            return None
            
        except Exception as e:
            logger.error(f"Failed to download activity {activity_id}: {e}")
            return None
    
    def download_activity_original(self, activity_id: str, force_download: bool = False, db_session: Optional[Session] = None) -> Optional[Path]:
        """Download original activity file (usually FIT format).
        
        Args:
            activity_id: Garmin activity ID
            force_download: If True, bypasses database checks and forces a re-download.
            db_session: Optional SQLAlchemy session to use for database operations.
            
        Returns:
            Path to downloaded file or None if download failed
        """
        if not self.is_authenticated():
            if not self.authenticate():
                return None
        
        db = db_session if db_session else self.db_session
        if not db:
            db = SessionLocal()
            close_session = True
        else:
            close_session = False
        try:
            # Check database for existing record unless force_download is True
            if not force_download:
                record = db.query(ActivityDownload).filter_by(activity_id=int(activity_id)).first()
                if record and record.status == "success" and Path(record.file_path).exists():
                    current_checksum = calculate_sha256(Path(record.file_path))
                    if current_checksum == record.checksum_sha256:
                        logger.info(f"Activity {activity_id} already downloaded and verified; skipping.")
                        return Path(record.file_path)
                    else:
                        logger.warning(f"Checksum mismatch for activity {activity_id}. Re-downloading.")

        finally:
            if close_session:
                db.close()

        download_status = "failed"
        error_message = None
        http_status = None
        downloaded_path = None
        
        try:
            # Create data directory if it doesn't exist
            DATA_DIR.mkdir(exist_ok=True)
            
            # Capability probe: does garminconnect client expose a native original download?
            has_native_original = hasattr(self.client, 'download_activity_original')
            logger.debug(f"garminconnect has download_activity_original: {has_native_original}")
            
            file_data = None
            attempts: List[str] = []
            
            # 1) Prefer native method when available
            if has_native_original:
                try:
                    attempts.append("self.client.download_original_activity(activity_id)")
                    logger.debug(f"Attempting native download_original_activity for activity {activity_id}")
                    file_data = self.client.download_activity_original(activity_id)
                except Exception as e:
                    logger.debug(f"Native download_original_activity failed: {e} (type={type(e).__name__})")
                    file_data = None
            
            # 2) Try download_activity with 'original' format
            if file_data is None and hasattr(self.client, 'download_activity'):
                try:
                    attempts.append("self.client.download_activity(activity_id, dl_fmt=self.client.ActivityDownloadFormat.ORIGINAL)")
                    logger.debug(f"Attempting original download via download_activity(dl_fmt=self.client.ActivityDownloadFormat.ORIGINAL) for activity {activity_id}")
                    file_data = self.client.download_activity(activity_id, dl_fmt=self.client.ActivityDownloadFormat.ORIGINAL)
                    logger.debug(f"download_activity(dl_fmt='original') succeeded, got data type: {type(file_data).__name__}, length: {len(file_data) if hasattr(file_data, '__len__') else 'N/A'}")
                    if file_data is not None and hasattr(file_data, '__len__') and len(file_data) > 0:
                        logger.debug(f"First 100 bytes: {file_data[:100]}")
                except Exception as e:
                    logger.debug(f"download_activity(dl_fmt='original') failed: {e} (type={type(e).__name__})")
                    file_data = None
            
            # 3) Try download_activity with positional token (older signatures)
            if file_data is None and hasattr(self.client, 'download_activity'):
                tokens_to_try_pos = ['ORIGINAL', 'original', 'FIT', 'fit']
                for token in tokens_to_try_pos:
                    try:
                        attempts.append(f"self.client.download_activity(activity_id, '{token}')")
                        logger.debug(f"Attempting original download via download_activity(activity_id, '{token}') for activity {activity_id}")
                        file_data = self.client.download_activity(activity_id, token)
                        logger.debug(f"download_activity(activity_id, '{token}') succeeded, got data type: {type(file_data).__name__}, length: {len(file_data) if hasattr(file_data, '__len__') else 'N/A'}")
                        if file_data is not None and hasattr(file_data, '__len__') and len(file_data) > 0:
                            logger.debug(f"First 100 bytes: {file_data[:100]}")
                        break
                    except Exception as e:
                        logger.debug(f"download_activity(activity_id, '{token}') failed: {e} (type={type(e).__name__})")
                        file_data = None
            
            # 4) Try alternate method names commonly seen in different garminconnect variants
            alt_methods_with_format = [
                ('download_activity_file', ['ORIGINAL', 'original', 'FIT', 'fit']),
            ]
            alt_methods_no_format = [
                'download_original_activity',
                'get_original_activity',
            ]
            
            if file_data is None:
                for method_name, fmts in alt_methods_with_format:
                    if hasattr(self.client, method_name):
                        method = getattr(self.client, method_name)
                        for fmt in fmts:
                            try:
                                attempts.append(f"self.client.{method_name}(activity_id, '{fmt}')")
                                logger.debug(f"Attempting {method_name}(activity_id, '{fmt}') for activity {activity_id}")
                                file_data = method(activity_id, fmt)
                                logger.debug(f"{method_name}(activity_id, '{fmt}') succeeded, got data type: {type(file_data).__name__}")
                                break
                            except Exception as e:
                                logger.debug(f"Attempting {method_name}(activity_id, '{fmt}') failed: {e} (type={type(e).__name__})")
                                file_data = None
                        if file_data is not None:
                            break
            
            if file_data is None:
                for method_name in alt_methods_no_format:
                    if hasattr(self.client, method_name):
                        method = getattr(self.client, method_name)
                        try:
                            attempts.append(f"self.client.{method_name}(activity_id)")
                            logger.debug(f"Attempting {method_name}(activity_id) for activity {activity_id}")
                            file_data = method(activity_id)
                            logger.debug(f"{method_name}(activity_id) succeeded, got data type: {type(file_data).__name__}")
                            break
                        except Exception as e:
                            logger.debug(f"Attempting {method_name}(activity_id) failed: {e} (type={type(e).__name__})")
                            file_data = None
            
            if file_data is None:
                # 5) HTTP fallback using authenticated requests session from garminconnect client
                session = None
                # Try common attributes that hold a requests.Session or similar
                for attr in ("session", "_session", "requests_session", "req_session", "http", "client"):
                    candidate = getattr(self.client, attr, None)
                    if candidate is not None and hasattr(candidate, "get"):
                        session = candidate
                        break
                    if candidate is not None and hasattr(candidate, "session") and hasattr(candidate.session, "get"):
                        session = candidate.session
                        break
                
                if session is not None:
                    http_urls = [
                        f"https://connect.garmin.com/modern/proxy/download-service/export/original/{activity_id}",
                        f"https://connect.garmin.com/modern/proxy/download-service/files/activity/{activity_id}",
                        f"https://connect.garmin.com/modern/proxy/download-service/export/zip/activity/{activity_id}",
                    ]
                    for url in http_urls:
                        try:
                            attempts.append(f"HTTP GET {url}")
                            logger.debug(f"Attempting HTTP fallback GET for original: {url}")
                            resp = session.get(url, timeout=30)
                            status = getattr(resp, "status_code", None)
                            content = getattr(resp, "content", None)
                            if status == 200 and content:
                                content_type = getattr(resp, "headers", {}).get("Content-Type", "")
                                logger.debug(f"HTTP fallback succeeded: status={status}, content-type='{content_type}', bytes={len(content)}")
                                file_data = content
                                http_status = status
                                break
                            else:
                                logger.debug(f"HTTP fallback GET {url} returned status={status} or empty content")
                                http_status = status
                        except Exception as e:
                            logger.debug(f"HTTP fallback GET {url} failed: {e} (type={type(e).__name__})")
                            error_message = str(e)
                
                if file_data is None:
                    logger.error(
                        f"Failed to obtain original/FIT data for activity {activity_id}. "
                        f"Attempts: {attempts}"
                    )
                    upsert_activity_download(
                        activity_id=int(activity_id),
                        source="garmin-connect",
                        file_path=DATA_DIR / f"activity_{activity_id}.fit", # Placeholder path
                        file_format="fit", # Assuming fit as target format
                        status="failed",
                        http_status=http_status,
                        error_message=error_message or f"All download attempts failed: {attempts}",
                        db_session=db
                    )
                    return None
            
            # Normalize to raw bytes if response-like object returned
            if hasattr(file_data, 'content'):
                try:
                    file_data = file_data.content
                except Exception:
                    pass
            elif hasattr(file_data, 'read'):
                try:
                    file_data = file_data.read()
                except Exception:
                    pass
            
            if not isinstance(file_data, (bytes, bytearray)):
                logger.error(f"Downloaded data for activity {activity_id} is not bytes (type={type(file_data).__name__}); aborting")
                logger.debug(f"Data content: {repr(file_data)[:200]}")
                upsert_activity_download(
                    activity_id=int(activity_id),
                    source="garmin-connect",
                    file_path=DATA_DIR / f"activity_{activity_id}.fit", # Placeholder path
                    file_format="fit", # Assuming fit as target format
                    status="failed",
                    http_status=http_status,
                    error_message=f"Downloaded data is not bytes: {type(file_data).__name__}",
                    db_session=db
                )
                return None
            
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file_data)
                tmp_path = Path(tmp_file.name)
            
            # Determine if the response is a ZIP archive (original) or a direct FIT file
            file_format_detected = "fit" # Default to fit
            extracted_path = DATA_DIR / f"activity_{activity_id}.fit" # Default path
            
            if zipfile.is_zipfile(tmp_path):
                # Extract zip file
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    # Find the first FIT file in the zip
                    fit_files = [f for f in zip_ref.namelist() if f.lower().endswith('.fit')]
                    
                    if fit_files:
                        # Extract the first FIT file
                        fit_filename = fit_files[0]
                        
                        with zip_ref.open(fit_filename) as source, open(extracted_path, 'wb') as target:
                            target.write(source.read())
                        
                        # Clean up temporary zip file
                        tmp_path.unlink()
                        
                        logger.info(f"Downloaded original activity file: {extracted_path}")
                        downloaded_path = extracted_path
                        download_status = "success"
                    else:
                        logger.warning("No FIT file found in downloaded archive")
                        tmp_path.unlink()
                        error_message = "No FIT file found in downloaded archive"
            else:
                # Treat data as direct FIT bytes
                try:
                    tmp_path.rename(extracted_path)
                    downloaded_path = extracted_path
                    download_status = "success" # Consider copy as success if file is there
                except Exception as move_err:
                    logger.debug(f"Rename temp FIT to destination failed ({move_err}); falling back to copy")
                    with open(extracted_path, 'wb') as target, open(tmp_path, 'rb') as source:
                        target.write(source.read())
                    tmp_path.unlink()
                    downloaded_path = extracted_path
                    download_status = "success" # Consider copy as success if file is there
                logger.info(f"Downloaded original activity file: {extracted_path}")
            
        except Exception as e:
            logger.error(f"Failed to download original activity {activity_id}: {e} (type={type(e).__name__})")
            error_message = str(e)
        finally:
            if downloaded_path:
                file_size = os.path.getsize(downloaded_path)
                file_checksum = calculate_sha256(downloaded_path)
                upsert_activity_download(
                    activity_id=int(activity_id),
                    source="garmin-connect",
                    file_path=downloaded_path,
                    file_format=file_format_detected,
                    status=download_status,
                    http_status=http_status,
                    size_bytes=file_size,
                    checksum_sha256=file_checksum,
                    error_message=error_message,
                    db_session=db
                )
            else:
                upsert_activity_download(
                    activity_id=int(activity_id),
                    source="garmin-connect",
                    file_path=DATA_DIR / f"activity_{activity_id}.fit", # Placeholder path
                    file_format="fit", # Assuming fit as target format
                    status="failed",
                    http_status=http_status,
                    error_message=error_message or "Unknown error during download",
                    db_session=db
                )
            if close_session:
                db.close()
        return downloaded_path
    
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
    
    def get_all_activities(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all activities from Garmin Connect.

        Args:
            limit: Maximum number of activities to retrieve

        Returns:
            List of activity dictionaries
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

                activities.extend(batch)

                offset += len(batch)

                # Stop if we got fewer activities than requested
                if len(batch) < batch_size:
                    break

            logger.info(f"Found {len(activities)} activities")
            return activities

        except Exception as e:
            logger.error(f"Failed to get activities: {e}")
            return []

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
    def download_all_workouts(
        self, limit: int = 50, output_dir: Path = DATA_DIR, force_download: bool = False
    ) -> List[Dict[str, Path]]:
        """Download up to 'limit' activities and save FIT files to output_dir.

        Uses get_all_activities() to list activities, then downloads each original
        activity archive and extracts the FIT file via download_activity_original().

        Args:
            limit: Maximum number of activities to download
            output_dir: Directory to save downloaded FIT files
            force_download: If True, bypasses database checks and forces a re-download.

        Returns:
            List of dicts with 'file_path' pointing to downloaded FIT paths
        """
        if not self.is_authenticated():
            if not self.authenticate():
                logger.error("Authentication failed; cannot download workouts")
                return []

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            activities = self.get_all_activities(limit=limit) # Changed from get_all_cycling_workouts
            total = min(limit, len(activities))
            logger.info(f"Preparing to download up to {total} activities into {output_dir}") # Changed from cycling activities

            results: List[Dict[str, Path]] = []
            for idx, activity in enumerate(activities[:limit], start=1):
                activity_id = (
                    activity.get("activityId")
                    or activity.get("activity_id")
                    or activity.get("id")
                )
                if not activity_id:
                    logger.warning("Skipping activity with missing ID key (activityId/activity_id/id)")
                    continue

                dest_path = output_dir / f"activity_{activity_id}.fit"
                data_dir_path = DATA_DIR / f"activity_{activity_id}.fit"

                if dest_path.exists():
                    logger.info(f"Activity {activity_id} already exists in {output_dir}; skipping download.")
                    results.append({"file_path": dest_path})
                    continue
                elif data_dir_path.exists():
                    logger.info(f"Activity {activity_id} found in {DATA_DIR}; moving to {output_dir} and skipping download.")
                    try:
                        data_dir_path.rename(dest_path)
                        results.append({"file_path": dest_path})
                        continue
                    except Exception as move_err:
                        logger.error(f"Failed to move {data_dir_path} to {dest_path}: {move_err}")
                        # Fall through to download if move fails

                logger.debug(f"Downloading activity ID {activity_id} ({idx}/{total})")
                
                # Add rate limiting
                import time
                time.sleep(1.0)

                src_path = self.download_activity_original(
                    str(activity_id), force_download=force_download, db_session=self.db_session
                )
                if src_path and src_path.exists():
                    # Check if the downloaded file is already the desired destination
                    if src_path.resolve() == dest_path.resolve():
                        logger.info(f"Saved activity {activity_id} to {dest_path}")
                        results.append({"file_path": dest_path})
                    else:
                        try:
                            # If not, move it to the desired location
                            if dest_path.exists():
                                dest_path.unlink()  # Overwrite existing destination to keep most recent download
                            src_path.rename(dest_path)
                            logger.info(f"Saved activity {activity_id} to {dest_path}")
                            results.append({"file_path": dest_path})
                        except Exception as move_err:
                            logger.error(f"Failed to move {src_path} to {dest_path}: {move_err}")
                            results.append({"file_path": src_path})  # Fall back to original location
                else:
                    logger.warning(f"Download returned no file for activity {activity_id}")

            logger.info(f"Downloaded {len(results)} activities to {output_dir}")
            return results

        except Exception as e:
            logger.error(f"Failed during batch download: {e}")
            return []

    def download_latest_workout(
        self, output_dir: Path = DATA_DIR, force_download: bool = False
    ) -> Optional[Path]:
        """Download the latest cycling workout and save FIT file to output_dir.

        Uses get_latest_activity('cycling') to find the most recent cycling activity,
        then downloads the original archive and extracts the FIT via download_activity_original().

        Args:
            output_dir: Directory to save the downloaded FIT file
            force_download: If True, bypasses database checks and forces a re-download.

        Returns:
            Path to the downloaded FIT file or None if download failed
        """
        if not self.is_authenticated():
            if not self.authenticate():
                logger.error("Authentication failed; cannot download latest workout")
                return None

        try:
            latest = self.get_latest_activity(activity_type="cycling")
            if not latest:
                logger.warning("No latest cycling activity found")
                return None

            activity_id = (
                latest.get("activityId")
                or latest.get("activity_id")
                or latest.get("id")
            )
            if not activity_id:
                logger.error("Latest activity missing ID key (activityId/activity_id/id)")
                return None

            logger.info(f"Downloading latest cycling activity ID {activity_id}")
            src_path = self.download_activity_original(
                str(activity_id), force_download=force_download, db_session=self.db_session
            )
            if src_path and src_path.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                dest_path = output_dir / src_path.name
                try:
                    if src_path.resolve() != dest_path.resolve():
                        if dest_path.exists():
                            dest_path.unlink()
                        src_path.rename(dest_path)
                except Exception as move_err:
                    logger.error(f"Failed to move {src_path} to {dest_path}: {move_err}")
                    return src_path  # Return original location if move failed

                logger.info(f"Saved latest activity {activity_id} to {dest_path}")
                return dest_path

            logger.warning(f"Download returned no file for latest activity {activity_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to download latest workout: {e}")
            return None