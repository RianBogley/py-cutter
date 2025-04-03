"""
py-cutter

This script processes a CSV or Excel file with video/audio files and cuts them
according to specified start and end times. It supports custom output labeling.
It can also optionally transcribe the output files using OpenAI's Whisper model.

Input file must have the following columns:
- original_filepath: Path to the source file
- cut_start_time: Start time in HH:MM:SS format
- cut_stop_time: Stop time in HH:MM:SS format
- cut_label: Label to add to the end of the output filename
- language: Language of the audio (optional, defaults to English)

The script will:
1. Validate input data
2. Cut files according to specified times
3. Save cut files to specified output directory or original location
4. Add _copy-# suffix if a file already exists at the target location
5. Optionally transcribe the cut files using Whisper (when --transcribe flag is used)
"""

import os
import pandas as pd
import re
import subprocess
import sys
from pathlib import Path
import uuid
from typing import Optional, Tuple, Dict, Any, List
import mimetypes
import logging
from datetime import datetime
import argparse
import importlib
import pkg_resources
from packaging import version

# Set up logging
def setup_logging():
    """Configure logging for the script (console only)."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def parse_time(time_value: Any) -> Optional[str]:
    """
    Validate and parse time value in various formats and convert to HH:MM:SS format.
    
    Args:
        time_value: Time value (string, datetime.time, pandas Timestamp, etc.)
        
    Returns:
        Validated time string in HH:MM:SS format or None if invalid
    """
    if pd.isna(time_value) or time_value is None:
        return None
    
    # If it's already a string, check if it's properly formatted
    if isinstance(time_value, str):
        # Check if it matches the HH:MM:SS format with optional leading zeros
        time_pattern = re.compile(r'^(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.\d+)?$')
        match = time_pattern.match(time_value)
        if match:
            # Format with leading zeros if needed
            hours, minutes, seconds = match.groups()
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Try to parse as time string in case it's in a different format
        try:
            parsed_time = pd.to_datetime(time_value).time()
            return f"{parsed_time.hour:02d}:{parsed_time.minute:02d}:{parsed_time.second:02d}"
        except:
            return None
    
    # Handle datetime.time objects
    if hasattr(time_value, 'strftime') and hasattr(time_value, 'hour') and hasattr(time_value, 'minute') and hasattr(time_value, 'second'):
        try:
            return time_value.strftime('%H:%M:%S')
        except:
            pass
            
    # Handle pandas Timestamp objects
    if hasattr(time_value, 'time') and callable(getattr(time_value, 'time', None)):
        try:
            return time_value.time().strftime('%H:%M:%S')
        except:
            pass
    
    # For Excel Time objects - these are often represented as fractional days
    # A time of 0:03:00 would be represented as 0.00208... (3 minutes / 1440 minutes in a day)
    try:
        # Check if it's a number between 0 and 1 (fractional day)
        if isinstance(time_value, (int, float)) and 0 <= time_value < 1:
            # Convert to seconds, then to time
            seconds_in_day = 24 * 60 * 60
            total_seconds = round(time_value * seconds_in_day)
            
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except:
        pass
    
    # Try to convert to string and see if it works
    try:
        time_str = str(time_value)
        # Check if it looks like a time string with optional AM/PM
        time_pattern = re.compile(r'^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?(?:\s*([AP]M))?$', re.IGNORECASE)
        match = time_pattern.match(time_str)
        if match:
            hours, minutes, seconds, ampm = match.groups()
            hours = int(hours)
            minutes = int(minutes)
            seconds = int(seconds) if seconds else 0
            
            # Adjust for AM/PM if present
            if ampm and ampm.upper() == 'PM' and hours < 12:
                hours += 12
            if ampm and ampm.upper() == 'AM' and hours == 12:
                hours = 0
                
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except:
        pass
    
    # Last resort: try to convert using pandas
    try:
        # Handle various formats including Excel datetime
        dt = pd.to_datetime(time_value, errors='coerce')
        if not pd.isna(dt):
            return dt.strftime('%H:%M:%S')
    except:
        pass
        
    return None

def check_ffmpeg_installed() -> bool:
    """Check if FFmpeg is installed and available in PATH."""
    try:
        subprocess.run(
            ['ffmpeg', '-version'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_video_duration(filepath: Path) -> Optional[float]:
    """
    Get the duration of a video or audio file in seconds.
    
    Args:
        filepath: Path to the video or audio file
        
    Returns:
        Duration in seconds, or None if there was an error
    """
    try:
        # Use ffprobe to get duration information
        command = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'default=noprint_wrappers=1:nokey=1', 
            str(filepath)
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logging.error(f"Failed to get duration for {filepath}: {result.stderr}")
            return None
        
        # Parse the duration
        duration = float(result.stdout.strip())
        return duration
    
    except Exception as e:
        logging.error(f"Error getting duration for {filepath}: {str(e)}")
        return None

def get_output_filepath(source_path: Path, cut_label: str, output_dir: Optional[Path] = None) -> Path:
    """
    Determine the output filepath based on source file, cut label, and output directory.
    
    Args:
        source_path: Path to the source file
        cut_label: Label to add to the output filename
        output_dir: Optional output directory
        
    Returns:
        Path object for the output file
    """
    # Get the base filename without extension
    stem = source_path.stem
    suffix = source_path.suffix
    
    # Create new filename with cut label
    new_filename = f"{stem}_{cut_label}{suffix}"
    
    # Determine target directory
    if output_dir:
        target_dir = output_dir
    else:
        target_dir = source_path.parent
    
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Return full path to target file
    return target_dir / new_filename

def get_next_available_filename(filepath: Path) -> Path:
    """
    Get next available filename by adding _copy-N if a file with the same name exists.
    
    Args:
        filepath: Path to check
        
    Returns:
        Path with _copy-N suffix if needed
    """
    if not filepath.exists():
        return filepath
        
    suffix = filepath.suffix
    parent = filepath.parent
    base = filepath.stem
    
    # Remove existing copy tag if present
    if '_copy-' in base:
        base = base.split('_copy-')[0]
    
    counter = 1
    while True:
        test_path = parent / f"{base}_copy-{counter}{suffix}"
        if not test_path.exists():
            return test_path
        counter += 1

def cut_media_file(source_path: Path, target_path: Path, start_time: str, end_time: str) -> Tuple[bool, str]:
    """
    Cut a media file from start_time to end_time.
    
    Args:
        source_path: Path to the source file
        target_path: Path to save the cut file
        start_time: Start time in HH:MM:SS format
        end_time: End time in HH:MM:SS format
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Ensure target directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary output file path
        temp_output = target_path.with_name(f".tmp_{uuid.uuid4()}_{target_path.name}")
        
        # Check file type
        suffix = source_path.suffix.lower()
        is_audio_only = suffix in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a']
        
        # Use different ffmpeg commands for audio and video files
        if is_audio_only:
            # Command optimized for audio files
            command = [
                'ffmpeg',
                '-i', str(source_path),
                '-ss', start_time,
                '-to', end_time,
                '-acodec', 'copy',  # Copy audio codec for faster processing
                '-y',               # Overwrite output if exists
                str(temp_output)
            ]
        else:
            # Command for video files with audio
            command = [
                'ffmpeg',
                '-i', str(source_path),
                '-ss', start_time,
                '-to', end_time,
                # Use re-encoding for accurate cutting
                '-c:v', 'libx264',  # Use H.264 for video
                '-c:a', 'aac',      # Use AAC for audio
                '-strict', 'experimental',
                '-b:a', '192k',     # Audio bitrate
                '-avoid_negative_ts', '1',
                '-y',               # Overwrite output if exists
                str(temp_output)
            ]
        
        # Run the ffmpeg command
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the command failed - try alternative method for audio files
        if (not temp_output.exists() or temp_output.stat().st_size == 0) and is_audio_only:
            logging.info(f"First attempt failed for audio file, trying alternative method...")
            
            # Alternative command for audio files that uses re-encoding
            alt_command = [
                'ffmpeg',
                '-i', str(source_path),
                '-ss', start_time,
                '-to', end_time,
                '-acodec', 'libmp3lame' if suffix == '.mp3' else 'aac',
                '-ar', '44100',      # Sample rate
                '-ab', '192k',       # Bitrate
                '-y',                # Overwrite output if exists
                str(temp_output)
            ]
            
            process = subprocess.run(
                alt_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        
        # Final check if output was created successfully
        if not temp_output.exists() or temp_output.stat().st_size == 0:
            if temp_output.exists():
                temp_output.unlink()
            return False, f"Failed to create output file: {process.stderr}"
            
        # Check if the destination exists
        if target_path.exists():
            # Generate a new filename with a copy tag
            target_path = get_next_available_filename(target_path)
            logging.warning(f"Target file already exists, using: {target_path.name}")
        
        # Rename the temporary file to the target path
        os.rename(temp_output, target_path)
            
        return True, str(target_path)
    except subprocess.CalledProcessError as e:
        if 'temp_output' in locals() and temp_output.exists():
            temp_output.unlink()
        return False, f"FFmpeg error: {e.stderr}"
    except Exception as e:
        if 'temp_output' in locals() and temp_output.exists():
            temp_output.unlink()
        return False, f"Error: {str(e)}"

def validate_input_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the input DataFrame to ensure it has all required columns and values.
    Also validates that cut times are within the bounds of original files.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for required columns
    required_columns = ['original_filepath', 'cut_start_time', 'cut_stop_time', 'cut_label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return False, errors
        
    # Add 'language' column if it doesn't exist
    if 'language' not in df.columns:
        default_language = args.default_language if 'args' in locals() else 'en'
        logging.info(f"'language' column not found in input file, adding with default value ({default_language})")
        df['language'] = default_language
    
    # Check for missing values
    for col in required_columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            errors.append(f"Column '{col}' has {missing_count} missing values")
    
    # Check if source files exist and validate time bounds
    for idx, row in df.iterrows():
        filepath = row['original_filepath']
        source_path = Path(filepath)
        
        # Check if file exists
        if not source_path.exists():
            errors.append(f"Row {idx+1}: Source file does not exist: {filepath}")
            continue
        
        # Validate time formats
        start_time = parse_time(row['cut_start_time'])
        if not start_time:
            errors.append(f"Row {idx+1}: Invalid start time format: {row['cut_start_time']}")
            continue
            
        end_time = parse_time(row['cut_stop_time'])
        if not end_time:
            errors.append(f"Row {idx+1}: Invalid end time format: {row['cut_stop_time']}")
            continue
        
        # Check time bounds
        # Get video duration
        duration = get_video_duration(source_path)
        if duration is None:
            errors.append(f"Row {idx+1}: Could not determine duration of file: {filepath}")
            continue
            
        # Convert time strings to seconds for comparison
        start_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
        end_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))
        
        # Validate start and end times are within the duration
        if start_seconds >= duration:
            errors.append(f"Row {idx+1}: Start time ({start_time}) exceeds file duration ({duration:.2f} seconds)")
        
        if end_seconds > duration:
            errors.append(f"Row {idx+1}: End time ({end_time}) exceeds file duration ({duration:.2f} seconds)")
        
        # Check if start time is before end time
        if start_seconds >= end_seconds:
            errors.append(f"Row {idx+1}: Start time ({start_time}) must be before end time ({end_time})")
    
    return len(errors) == 0, errors

def check_for_duplicate_outputs(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """
    Check for potential duplicate output files in the input DataFrame.
    
    Args:
        df: Input DataFrame
        output_dir: Optional output directory
        
    Returns:
        Tuple of (has_duplicates, list_of_duplicates)
    """
    output_paths = []
    duplicate_paths = []
    
    for idx, row in df.iterrows():
        source_path = Path(row['original_filepath'])
        output_path = get_output_filepath(source_path, row['cut_label'], output_dir)
        
        # Check if this output path already exists in our list
        if str(output_path) in output_paths:
            duplicate_paths.append(f"Row {idx+1}: Duplicate output path: {output_path}")
        else:
            output_paths.append(str(output_path))
        
        # Check if file already exists at output path
        if output_path.exists():
            duplicate_paths.append(f"Row {idx+1}: File already exists at output path: {output_path}")
    
    return len(duplicate_paths) > 0, duplicate_paths

# New function to check and install Whisper
def check_and_install_whisper() -> Tuple[bool, str]:
    """
    Check if OpenAI Whisper is installed, and install it if it's not.
    
    Returns:
        Tuple of (success, version or error message)
    """
    try:
        # Check if whisper is already installed
        try:
            import openai_whisper
            whisper_installed = True
            whisper_module = openai_whisper
        except ImportError:
            try:
                import whisper
                whisper_installed = True
                whisper_module = whisper
            except ImportError:
                whisper_installed = False
                whisper_module = None
        
        if whisper_installed:
            # Get the version
            try:
                whisper_version = whisper_module.__version__
                logging.info(f"Whisper already installed (version {whisper_version})")
                return True, whisper_version
            except AttributeError:
                logging.info("Whisper already installed (version unknown)")
                return True, "unknown"
        
        # Install whisper if not installed
        logging.info("Installing OpenAI Whisper...")
        print("Installing OpenAI Whisper... This might take a few minutes.")
        
        # Install whisper using pip
        install_command = [
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "--upgrade", 
            "openai-whisper"
        ]
        
        result = subprocess.run(
            install_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logging.error(f"Failed to install Whisper: {result.stderr}")
            return False, f"Installation failed: {result.stderr}"
        
        # Verify the installation
        try:
            # Try to reload the module
            if "whisper" in sys.modules:
                importlib.reload(sys.modules["whisper"])
            else:
                import whisper
            
            # Get the version
            try:
                whisper_version = whisper.__version__
                logging.info(f"Whisper installed successfully (version {whisper_version})")
                return True, whisper_version
            except AttributeError:
                logging.info("Whisper installed successfully (version unknown)")
                return True, "unknown"
        except ImportError as e:
            logging.error(f"Failed to import Whisper after installation: {str(e)}")
            return False, f"Import error after installation: {str(e)}"
        
    except Exception as e:
        logging.error(f"Error checking/installing Whisper: {str(e)}")
        return False, f"Error: {str(e)}"

# New function to transcribe audio file using Whisper
def transcribe_audio_file(audio_path: Path, whisper_version: str, language: str = None) -> Tuple[bool, str, str]:
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file
        whisper_version: Version of Whisper being used
        language: Language of the audio (optional)
        
    Returns:
        Tuple of (success, transcript text or error message, output path)
    """
    try:
        logging.info(f"Transcribing: {audio_path}")
        if language:
            logging.info(f"Using language: {language}")
        
        # Import whisper
        try:
            import openai_whisper as whisper
        except ImportError:
            import whisper
        
        # Load the model
        model = whisper.load_model("large-v3")
        
        # Generate transcript with the specified language if provided
        transcribe_options = {}
        if language and language.strip():
            transcribe_options["language"] = language.strip().lower()
            result = model.transcribe(str(audio_path), **transcribe_options)
            language_suffix = f"-{language.strip().lower()}"
        else:
            result = model.transcribe(str(audio_path))
            language_suffix = ""
        
        # Extract the transcript text
        transcript = result["text"]
        
        # Create output filename with the whisper version and language
        transcript_path = audio_path.with_name(f"{audio_path.stem}_transcript-whisper-large-v3{language_suffix}-{whisper_version}.txt")
        
        # Save the transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        logging.info(f"Transcription saved to: {transcript_path}")
        
        return True, transcript, str(transcript_path)
    
    except Exception as e:
        error_msg = f"Error transcribing {audio_path}: {str(e)}"
        logging.error(error_msg)
        return False, error_msg, ""

def process_row(row: Dict[str, Any], output_dir: Optional[Path] = None, whisper_version: str = "unknown", enable_transcription: bool = False) -> Dict[str, Any]:
    """
    Process a single row from the input file.
    
    Args:
        row: Dictionary representing a row from the input file
        output_dir: Optional output directory
        whisper_version: Version of Whisper being used
        enable_transcription: Whether transcription is enabled
        
    Returns:
        Updated row with results
    """
    result = row.copy()
    result['output_filepath'] = ""
    result['status'] = ""
    result['message'] = ""
    result['transcript_filepath'] = ""
    result['transcript_status'] = ""
    # Don't add WARNINGS here - we'll add it at the end based on status/message
    
    try:
        # Parse source path and times
        source_path = Path(row['original_filepath'])
        
        # Convert times to HH:MM:SS format
        start_time = parse_time(row['cut_start_time'])
        end_time = parse_time(row['cut_stop_time'])
        
        # Get language for transcription (default to English if not specified)
        language = row.get('language', 'en')
        
        # Get output filepath
        output_path = get_output_filepath(source_path, row['cut_label'], output_dir)
        
        # Check if target file already exists
        if output_path.exists():
            # Create a copy with _copy-N suffix
            output_path = get_next_available_filename(output_path)
            logging.warning(f"Target file already exists, using: {output_path.name}")
        
        # Cut the file
        success, message = cut_media_file(source_path, output_path, start_time, end_time)
        
        if success:
            result['output_filepath'] = message
            result['status'] = "SUCCESS"
            result['message'] = f"File cut successfully"
            
            # If successful and transcription is enabled, transcribe the file
            if enable_transcription:
                output_file_path = Path(message)
                suffix = output_file_path.suffix.lower()
                
                # Only transcribe audio or video files
                if suffix in ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.mp4', '.mov', '.avi', '.mkv']:
                    trans_success, trans_message, trans_filepath = transcribe_audio_file(output_file_path, whisper_version, language)
                    result['transcript_filepath'] = trans_filepath
                    result['transcript_status'] = "SUCCESS" if trans_success else "FAILED"
                    
                    if not trans_success:
                        logging.error(f"Transcription failed: {trans_message}")
                else:
                    result['transcript_status'] = "SKIPPED"
                    logging.info(f"Skipped transcription for non-audio/video file: {output_file_path}")
            else:
                result['transcript_status'] = "DISABLED"
        else:
            result['status'] = "FAILED"
            result['message'] = message
            result['transcript_status'] = "SKIPPED"
    
    except Exception as e:
        result['status'] = "ERROR"
        result['message'] = str(e)
        result['transcript_status'] = "SKIPPED"
    
    return result

def detect_file_type(file_path: str) -> str:
    """
    Detect if the input file is CSV or Excel format.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        String indicating file type: 'csv', 'excel', or 'unknown'
    """
    # Check file extension first
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        return 'csv'
    elif ext in ['.xlsx', '.xls']:
        return 'excel'
    
    # If extension is ambiguous, try to determine by content
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type == 'text/csv':
            return 'csv'
        elif mime_type in ['application/vnd.ms-excel', 
                          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            return 'excel'
    except:
        pass
    
    # Last resort: try to open it as CSV first, then Excel
    try:
        pd.read_csv(file_path, nrows=1)
        return 'csv'
    except:
        try:
            pd.read_excel(file_path, nrows=1)
            return 'excel'
        except:
            return 'unknown'

def main():
    """
    Main function to process files based on the input file.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Cut audio/video files based on input CSV or Excel file and transcribe them using Whisper.')
    parser.add_argument('input_file', type=str, help='Input CSV or Excel file with cutting instructions')
    parser.add_argument('--output-dir', type=str, help='Output directory for cut files (optional)')
    parser.add_argument('--force', action='store_true', help='Force processing even if duplicate outputs detected')
    parser.add_argument('--save-log', action='store_true', help='Save results log file alongside the input file (default: off)')
    parser.add_argument('--transcribe', action='store_true', help='Enable transcription of the cut files using Whisper (default: off)')
    parser.add_argument('--default-language', type=str, default='en', help='Default language for transcription if not specified in the input file (default: en)')
    
    args = parser.parse_args()
    
    # Set up logging (console only)
    setup_logging()
    logging.info(f"Starting py-cutter with input: {args.input_file}")
    
    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        print("Error: FFmpeg is not installed or not found in PATH")
        print("Please install FFmpeg and make sure it's available in your PATH")
        print("See the README.md for installation instructions")
        sys.exit(1)
    
    # Check and install Whisper if not installed and transcription is enabled
    whisper_version = "unknown"
    if args.transcribe:
        whisper_success, whisper_version_or_error = check_and_install_whisper()
        if not whisper_success:
            print(f"Error: Failed to install or load Whisper: {whisper_version_or_error}")
            print("You can still proceed with cutting files without transcription")
            args.transcribe = False  # Disable transcription if Whisper installation failed
            logging.warning("Transcription disabled due to Whisper installation failure")
        else:
            whisper_version = whisper_version_or_error
            logging.info(f"Transcription enabled with Whisper version {whisper_version}")
    else:
        logging.info("Transcription not enabled (use --transcribe to enable)")

    
    try:
        # Verify input file exists
        input_file = args.input_file
        input_path = Path(input_file)
        if not input_path.exists():
            logging.error(f"Input file not found: {input_file}")
            print(f"Error: Input file not found: {input_file}")
            sys.exit(1)
        
        # Set output directory if specified
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Output directory: {output_dir}")
        
        # Detect file type
        file_type = detect_file_type(input_file)
        logging.info(f"Detected input file type: {file_type}")
        
        # Read the input file based on type
        if file_type == 'csv':
            df = pd.read_csv(input_file)
        elif file_type == 'excel':
            df = pd.read_excel(input_file)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            print(f"Error: Unsupported file type: {file_type}")
            sys.exit(1)
            
        logging.info(f"Read {len(df)} rows from input file")
        
        # Validate input data
        is_valid, validation_errors = validate_input_data(df)
        
        # If we have a default language from args, apply it to any rows with empty language values
        if 'language' in df.columns and args.default_language:
            df.loc[df['language'].isna() | (df['language'] == ''), 'language'] = args.default_language
            logging.info(f"Applied default language '{args.default_language}' to rows with empty language values")
            
        if not is_valid:
            logging.error("Validation errors found:")
            for error in validation_errors:
                logging.error(f"  - {error}")
            print("Error: Input data validation failed:")
            for error in validation_errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(validation_errors) > 5:
                print(f"  ... and {len(validation_errors) - 5} more errors")
            sys.exit(1)
        
        # Check for duplicate outputs
        has_duplicates, duplicate_paths = check_for_duplicate_outputs(df, output_dir)
        if has_duplicates and not args.force:
            logging.error("Duplicate output paths detected:")
            for duplicate in duplicate_paths:
                logging.error(f"  - {duplicate}")
            print("Error: Potential duplicate output files detected:")
            for duplicate in duplicate_paths[:5]:  # Show first 5 duplicates
                print(f"  - {duplicate}")
            if len(duplicate_paths) > 5:
                print(f"  ... and {len(duplicate_paths) - 5} more duplicates")
            print("\nThis could result in overwriting files or creating unintended _copy-N suffixes.")
            print("Options:")
            print("  1. Fix the input file to ensure unique outputs")
            print("  2. Run again with --force to proceed anyway (will add _copy-N suffix to duplicates)")
            sys.exit(1)
        elif has_duplicates and args.force:
            logging.warning("Duplicate output paths detected, proceeding with --force flag")
            print("Warning: Proceeding with duplicate output paths (--force flag enabled)")
            print("Files will be created with _copy-N suffixes as needed")
        
        # Process each row
        results = []
        for idx, row in df.iterrows():
            logging.info(f"Processing: {row['original_filepath']}")
            result = process_row(row.to_dict(), output_dir, whisper_version, args.transcribe)
            results.append(result)
            
            # Log result
            if result['status'] == "SUCCESS":
                logging.info(f"Successfully cut: {result['output_filepath']}")
                
                # Log transcription result if applicable
                if result['transcript_status'] == "SUCCESS":
                    logging.info(f"Successfully transcribed to: {result['transcript_filepath']}")
                elif result['transcript_status'] == "FAILED":
                    logging.warning(f"Failed to transcribe: {row['original_filepath']}")
            else:
                logging.warning(f"Failed to cut {row['original_filepath']}: {result['message']}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Add WARNINGS column
        results_df['WARNINGS'] = ""
        for idx, row in results_df.iterrows():
            warnings = []
            if row['status'] != "SUCCESS":
                warnings.append(row['message'])
            if row['transcript_status'] == "FAILED":
                warnings.append(f"Transcription failed")
            
            if warnings:
                results_df.at[idx, 'WARNINGS'] = "; ".join(warnings)
        
        # Count successes and failures
        cut_successes = len([r for r in results if r['status'] == "SUCCESS"])
        cut_failures = len(results) - cut_successes
        
        # Only count transcription statuses if transcription was enabled
        if args.transcribe:
            trans_successes = len([r for r in results if r['transcript_status'] == "SUCCESS"])
            trans_failures = len([r for r in results if r['transcript_status'] == "FAILED"])
            trans_skipped = len([r for r in results if r['transcript_status'] == "SKIPPED"])
        else:
            trans_successes = trans_failures = trans_skipped = 0
        
        # Only save results log if --save-log is specified
        if args.save_log:
            # Save results with timestamp suffix alongside the input file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Use same format as input file
            if file_type == 'csv':
                output_file = input_path.with_name(f"{input_path.stem}_results_{timestamp}.csv")
                results_df.to_csv(output_file, index=False)
            else:  # excel
                output_file = input_path.with_name(f"{input_path.stem}_results_{timestamp}.xlsx")
                results_df.to_excel(output_file, index=False)
                
            logging.info(f"Results saved to: {output_file}")
            print(f"Results saved to: {output_file}")
        
        print(f"\nProcessing complete! {len(results)} files processed:")
        print(f"  - {cut_successes} files successfully cut")
        print(f"  - {cut_failures} files failed")
        
        if args.transcribe:
            print(f"\nTranscription results:")
            print(f"  - {trans_successes} files successfully transcribed")
            print(f"  - {trans_failures} files failed transcription")
            print(f"  - {trans_skipped} files skipped transcription")
            
            if trans_successes > 0:
                print(f"\nTranscription model: Whisper large-v3 (version {whisper_version})")
        else:
            print(f"\nTranscription was not enabled. Use --transcribe to enable audio transcription.")
        
    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
