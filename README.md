# Video Translation and Subtitle Generator

A streamlined Python tool that automatically transcribes video audio and generates translated subtitles using Whisper and Google Translate.

## Overview

This tool processes video files to:
1. Extract audio from the video
2. Transcribe the audio using OpenAI's Whisper (via `faster-whisper`)
3. Translate the transcribed text using Google Translate
4. Generate synchronized SRT subtitle files

## Features

- **Multi-language Support**: Transcribe from any language to any language
- **Robust Translation**: Multiple translation strategies with automatic fallback and retry logic
- **Progress Tracking**: Real-time progress indicators during processing
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Flexible Output**: Generate subtitle files in the same directory as the input video
- **Logging**: Detailed logging to both console and file (`translation.log`)

## Requirements

### System Dependencies
- **ffmpeg**: Required for audio extraction from video files
- **Python 3.12+**: Required for running the script

### Python Dependencies
- `faster-whisper`: For audio transcription
- `deep-translator`: For translation services
- `uv`: For dependency management (recommended)

## Installation

1. **Clone or download the project**
2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Verify ffmpeg is installed**:
   ```bash
   ffmpeg -version
   ```

## Usage

### Basic Usage

```bash
uv run translate.py "path/to/video.mp4" --source-language ru --target-language en
```

### Command Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `video_path` | - | Yes | Path to the video file to process |
| `--source-language` | `-s` | Yes | Source language code (e.g., 'ru', 'es', 'fr') |
| `--target-language` | `-t` | No | Target language code (default: 'en') |
| `--output-dir` | - | No | Output directory (default: same as input video) |
| `--model-size` | `-m` | No | Whisper model size (default: 'base') |
| `--device` | `-d` | No | Device for Whisper (default: 'auto') |

### Language Codes

Use standard ISO 639-1 language codes:
- `ru` - Russian
- `es` - Spanish  
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- And many more...

### Examples

**Translate Russian video to English:**
```bash
uv run translate.py "movie.mp4" --source-language ru --target-language en
```

**Translate Spanish video to French:**
```bash
uv run translate.py "video.avi" --source-language es --target-language fr
```

**Use a larger Whisper model for better accuracy:**
```bash
uv run translate.py "video.mkv" --source-language ja --target-language en --model-size large
```

**Specify custom output directory:**
```bash
uv run translate.py "video.mp4" --source-language de --target-language en --output-dir ./subtitles
```

## How It Works

### Processing Pipeline

1. **Audio Extraction**: Uses ffmpeg to extract audio from the video file
2. **Transcription**: Uses Whisper to transcribe the audio into text segments with timestamps
3. **Translation**: Translates the transcribed text using Google Translate with multiple strategies:
   - **Strategy 1**: Batch translation of all text at once
   - **Strategy 2**: Context-aware batch translation (50 segments per batch)
   - **Strategy 3**: Individual segment translation (fallback)
4. **Subtitle Generation**: Creates an SRT subtitle file with synchronized timestamps

### Translation Strategies

The tool uses a multi-tier approach to ensure reliable translation:

1. **Batch Translation**: Attempts to translate all text as one large batch
2. **Context-Aware Batch Translation**: Processes text in smaller batches (50 segments) to maintain context
3. **Individual Translation**: Falls back to translating each segment individually

Each strategy includes retry logic with exponential backoff (1s, 2s, 4s delays) to handle temporary API issues.

### Output

- **Subtitle File**: Generated as `{video_name}_{target_language}.srt` in the same directory as the input video
- **Log File**: Detailed processing logs saved to `translation.log`
- **Console Output**: Real-time progress updates and status messages

## Configuration

### Whisper Model Sizes

- `tiny`: Fastest, least accurate
- `base`: Good balance of speed and accuracy (default)
- `small`: Better accuracy, slower
- `medium`: High accuracy, slower
- `large`: Highest accuracy, slowest

### Device Options

- `auto`: Automatically select best available device
- `cpu`: Force CPU processing
- `cuda`: Use CUDA GPU acceleration (if available)

## Troubleshooting

### Common Issues

**"ffmpeg not found"**
- Install ffmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)

**"faster-whisper not found"**
- Run `uv sync` to install dependencies

**Translation failures**
- Check internet connection (Google Translate requires internet)
- Verify language codes are correct
- Check the log file for detailed error messages

**Memory issues with large videos**
- Use a smaller Whisper model: `--model-size tiny`
- Process shorter video segments

### Log Files

Detailed logs are saved to `translation.log` with DEBUG level information, including:
- Audio extraction details
- Transcription progress
- Translation attempts and retries
- Error messages and stack traces

## Performance

- **Processing Speed**: Depends on video length, Whisper model size, and hardware
- **Accuracy**: Generally high with the default 'base' model
- **Memory Usage**: Moderate, scales with video length
- **Network**: Requires internet connection for translation

## Limitations

- Requires internet connection for translation
- Translation quality depends on Google Translate
- Processing time scales with video length
- Audio quality affects transcription accuracy

## License

This project is provided as-is for educational and personal use.

## Contributing

Feel free to submit issues or pull requests to improve the tool. 