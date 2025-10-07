def download_activity_file(self, activity_id: str, file_format: str = "fit") -> Optional[Path]:
    """Download activity file in specified format.
    
    Args:
        activity_id: Garmin activity ID
        file_format: File format to download (fit, tcx, gpx, csv, original)
        
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
        
        # Map string format to ActivityDownloadFormat enum
        # Access the enum from the client instance
        format_mapping = {
            "GPX": self.client.ActivityDownloadFormat.GPX,
            "TCX": self.client.ActivityDownloadFormat.TCX,
            "ORIGINAL": self.client.ActivityDownloadFormat.ORIGINAL,
            "CSV": self.client.ActivityDownloadFormat.CSV,
        }
        
        if fmt_upper in format_mapping:
            # Use the enum value from the mapping
            dl_fmt = format_mapping[fmt_upper]
            file_data = self.client.download_activity(activity_id, dl_fmt=dl_fmt)
            
            # Determine file extension
            if fmt_upper == "ORIGINAL":
                extension = "zip"
            else:
                extension = fmt_upper.lower()
            
            # Save to file
            filename = f"activity_{activity_id}.{extension}"
            file_path = DATA_DIR / filename
            
            with open(file_path, "wb") as f:
                f.write(file_data)
            
            logger.info(f"Downloaded activity file: {file_path}")
            return file_path

        # For FIT format, use download_activity_original which handles the ZIP extraction
        elif fmt_upper == "FIT" or file_format.lower() == "fit":
            fit_path = self.download_activity_original(activity_id)
            return fit_path

        else:
            logger.error(f"Unsupported download format '{file_format}'. Valid: GPX, TCX, ORIGINAL, CSV, FIT")
            return None
            
    except Exception as e:
        logger.error(f"Failed to download activity {activity_id}: {e}")
        return None


def download_activity_original(self, activity_id: str) -> Optional[Path]:
    """Download original activity file (usually FIT format in a ZIP).
    
    Args:
        activity_id: Garmin activity ID
        
    Returns:
        Path to extracted FIT file or None if download failed
    """
    if not self.is_authenticated():
        if not self.authenticate():
            return None
    
    try:
        # Create data directory if it doesn't exist
        DATA_DIR.mkdir(exist_ok=True)
        
        # Use the ORIGINAL format enum to download the ZIP
        file_data = self.client.download_activity(
            activity_id, 
            dl_fmt=self.client.ActivityDownloadFormat.ORIGINAL
        )
        
        if not file_data:
            logger.error(f"No data received for activity {activity_id}")
            return None
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(file_data)
            tmp_path = Path(tmp_file.name)
        
        # Check if it's a ZIP file and extract
        if zipfile.is_zipfile(tmp_path):
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
                    
                    logger.info(f"Downloaded and extracted original activity: {extracted_path}")
                    return extracted_path
                else:
                    logger.warning("No FIT file found in downloaded ZIP archive")
                    tmp_path.unlink()
                    return None
        else:
            # If it's not a ZIP, assume it's already a FIT file
            extracted_path = DATA_DIR / f"activity_{activity_id}.fit"
            tmp_path.rename(extracted_path)
            logger.info(f"Downloaded original activity file: {extracted_path}")
            return extracted_path
                
    except Exception as e:
        logger.error(f"Failed to download original activity {activity_id}: {e}")
        return None
