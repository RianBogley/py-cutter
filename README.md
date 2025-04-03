<img src="https://github.com/user-attachments/assets/2ff16776-3858-49b6-80f6-742b303453df" alt="Py-Cutter_Logo" width="25%" />

# py-cutter

A simple and efficient tool to cut audio and video files using timestamps from a CSV or Excel file, with optional transcription using OpenAI's Whisper model.

## Features

- Cut multiple audio or video files in batch using a simple CSV or Excel input file
- Automatic detection of audio vs. video files for optimized cutting
- Optional transcription of cut audio/video files using OpenAI's Whisper large-v3 model
- Support for multiple languages in transcription
- Validates input data including time formats and file durations
- Never overwrites existing files (automatically creates _copy-N suffix)
- Supports custom output directories
- Optional results logging for tracking processing status

## Requirements

- Python 3.6+
- FFmpeg installed and available in your PATH
- Internet connection (for installing Whisper when using transcription feature)

### Installing FFmpeg

FFmpeg is required for py-cutter to work. Here's how to install it:

#### Windows
1. Download a static build from [FFmpeg's official site](https://ffmpeg.org/download.html) or use a package manager like Chocolatey:
   ```
   choco install ffmpeg
   ```
2. Add FFmpeg to your PATH environment variable

#### macOS
Using Homebrew:
```
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```
sudo apt update
sudo apt install ffmpeg
```

#### Verify installation
To verify that FFmpeg is installed correctly, run:
```
ffmpeg -version
```

## Installation

### From GitHub

```bash
pip install git+https://github.com/RianBogley/py-cutter.git
```

### From Source

```bash
git clone https://github.com/RianBogley/py-cutter.git
cd py-cutter
pip install .
```

## Usage

### Basic Usage

1. Create a CSV or Excel file with these columns:
   - `original_filepath`: Full path to the source file
   - `cut_start_time`: Start time in HH:MM:SS format
   - `cut_stop_time`: Stop time in HH:MM:SS format
   - `cut_label`: Label to add to the end of the filename
   - `language`: (Optional) Language code for transcription (e.g., 'en', 'fr', 'zh', 'yue')

2. Run the command:

```bash
py-cutter '/path/to/your_input_file.csv'
```

or 

```bash
py-cutter /path/to/your_input_file.xlsx'
```

### Options

- `--output-dir PATH`: Save all output files to a specific directory
- `--force`: Process even if duplicate output paths are detected
- `--save-log`: Save a results log file alongside the input file (default: off)
- `--transcribe`: Enable transcription of cut files using Whisper (default: off)
- `--default-language CODE`: Default language for transcription if not specified in the input file (default: 'en')

### Example

Input CSV file (`cuts.csv`):

```
original_filepath,cut_start_time,cut_stop_time,cut_label,language
/path/to/video1.mp4,00:01:20,00:02:45,interview,en
/path/to/audio1.mp3,00:05:30,00:06:45,quote,fr
/path/to/mandarin_speech.mp3,00:00:10,00:01:30,segment1,zh
/path/to/cantonese_speech.mp3,00:02:00,00:04:00,segment2,yue
```

Command for cutting only:

```bash
py-cutter cuts.csv --output-dir '/path/to/output/'
```

Command for cutting and transcribing:

```bash
py-cutter cuts.csv --output-dir '/path/to/output/' --transcribe
```

Command for cutting and transcribing with a default language:

```bash
py-cutter cuts.csv --output-dir '/path/to/output/' --transcribe --default-language fr
```

Output with transcription enabled:
- `/path/to/output/video1_interview.mp4` (cut from 00:01:20 to 00:02:45)
- `/path/to/output/video1_interview_transcript-whisper-large-v3-en-{version}.txt` (English transcript)
- `/path/to/output/audio1_quote.mp3` (cut from 00:05:30 to 00:06:45)
- `/path/to/output/audio1_quote_transcript-whisper-large-v3-fr-{version}.txt` (French transcript)
- `/path/to/output/mandarin_speech_segment1.mp3` (cut from 00:00:10 to 00:01:30)
- `/path/to/output/mandarin_speech_segment1_transcript-whisper-large-v3-zh-{version}.txt` (Mandarin transcript)
- `/path/to/output/cantonese_speech_segment2.mp3` (cut from 00:02:00 to 00:04:00)
- `/path/to/output/cantonese_speech_segment2_transcript-whisper-large-v3-yue-{version}.txt` (Cantonese transcript)

With logging enabled:

```bash
py-cutter cuts.csv --output-dir '/path/to/output/' --transcribe --save-log
```

This will also generate a log file (e.g., `cuts_results_20250331_132045.csv`) with processing results.

## Transcription

### Enabling Transcription

Transcription is disabled by default and can be enabled with the `--transcribe` flag. When enabled, the script will:

1. Install OpenAI's Whisper if not already installed
2. Use the "large-v3" model to transcribe each cut audio/video file
3. Save transcripts in the same location as the cut files

### Language Support

Whisper supports many languages. Specify the language in the `language` column of your input file or use the `--default-language` option.

Common language codes:
- `en`: English
- `fr`: French
- `de`: German
- `es`: Spanish
- `zh`: Mandarin Chinese
- `yue`: Cantonese
- `ja`: Japanese
- `ko`: Korean
- `ru`: Russian
- `ar`: Arabic

If the `language` column is missing, the script will automatically add it with English (`en`) as the default value, or the value specified with `--default-language`.

### Transcript File Naming

Transcript files follow this naming convention:
`{original_filename}_{cut_label}_transcript-whisper-large-v3-{language}-{whisper_version}.txt`

Example: `video1_interview_transcript-whisper-large-v3-en-20250401.txt`

## Log Output

When the `--save-log` option is enabled, the script generates a results file next to your input file with the same format (CSV or Excel) containing:
- All original columns from your input
- `output_filepath`: Path to the created file
- `status`: SUCCESS or FAILED
- `message`: Additional information
- `transcript_filepath`: Path to the transcript file (if transcription enabled)
- `transcript_status`: SUCCESS, FAILED, SKIPPED, or DISABLED
- `WARNINGS`: Any warnings or errors that occurred

For example: `cuts_results_20250331_132045.csv` or `cuts_results_20250331_132045.xlsx`

## Notes

- Time formats in HH:MM:SS are supported along with various Excel and CSV time formats
- The script validates that cut times are within the bounds of the original files
- For files that already exist, a _copy-N suffix is added (e.g., `video1_interview_copy-1.mp4`)
- All errors are properly logged to the console regardless of whether you save a log file
- By default, no log files are created unless you specify the `--save-log` option
- Transcription is disabled by default unless you specify the `--transcribe` option

## License

MIT License - see LICENSE file for details.
