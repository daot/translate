#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict
import faster_whisper
from deep_translator import GoogleTranslator

# Global configuration
DEFAULT_MODEL_SIZE = "base"
DEFAULT_DEVICE = "auto"
DEFAULT_SAMPLING_RATE = 16000  # Silero VAD works best with 16kHz


class VideoTranslator:
    def __init__(
        self, model_size: str = DEFAULT_MODEL_SIZE, device: str = DEFAULT_DEVICE
    ):
        self.model_size = model_size
        self.device = device
        self.whisper_model = None
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        # Create a custom formatter that's cleaner for console output
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler (clean output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)

        # File handler (detailed output)
        file_handler = logging.FileHandler("translation.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG, handlers=[console_handler, file_handler], force=True
        )
        self.logger = logging.getLogger(__name__)

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video using ffmpeg"""
        self.logger.info("Extracting audio from video...")

        try:
            cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(DEFAULT_SAMPLING_RATE),
                "-ac",
                "1",
                "-y",
                audio_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            self.logger.info("✓ Audio extracted successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ Failed to extract audio: {e}")
            return False

    def load_whisper_model(self):
        """Load Whisper model if not already loaded"""
        if self.whisper_model is None:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = faster_whisper.WhisperModel(
                self.model_size, device=self.device, compute_type="float32"
            )
            self.logger.info(f"✓ Whisper model loaded: {self.model_size}")

    def transcribe_with_whisper(
        self, audio_path: str, source_language: str = None
    ) -> List[Dict]:
        """Transcribe audio using faster-whisper"""
        try:
            # Load whisper model if not already loaded
            self.load_whisper_model()

            # Use faster-whisper's built-in translation
            self.logger.info("Transcribing audio with Whisper...")
            segments, info = self.whisper_model.transcribe(
                audio_path,
                language=source_language,  # Use specified source language or auto-detect
            )

            # Convert segments to our format with progress indicator
            processed_segments = []
            segment_count = 0

            self.logger.info("Processing transcription segments...")
            for segment in segments:
                segment_count += 1
                # Show progress every 10 segments
                if segment_count % 10 == 0:
                    self.logger.info(f"Processed {segment_count} segments...")

                processed_segments.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "language": (
                            info.language if hasattr(info, "language") else "unknown"
                        ),
                    }
                )

            self.logger.info(f"✓ Transcribed {len(processed_segments)} segments")
            return processed_segments

        except Exception as e:
            self.logger.error(f"✗ Transcription failed: {e}")
            raise

    def translate_segments(
        self, segments: List[Dict], target_language: str
    ) -> List[Dict]:
        """Translate segments using Google Translator"""
        if not segments:
            return segments

        if not target_language:
            self.logger.info("No target language provided, cannot translate...")
            return segments

        try:
            translator = GoogleTranslator(
                source="auto",
                target=target_language,
            )

            self.logger.info("Translating segments with GoogleTranslator...")

            # Strategy 1: Try batch translation of all text
            try:
                self.logger.info("Attempting batch translation...")
                return self._batch_translate_all(segments, translator, target_language)
            except Exception as e:
                self.logger.warning(f"Batch translation failed")
                self.logger.debug(f"Error: {e}")

            # Strategy 2: Try context-aware batch translation
            try:
                self.logger.info("Attempting context-aware batch translation...")
                return self._context_aware_batch_translate(
                    segments, translator, target_language
                )
            except Exception as e:
                self.logger.warning(f"Context-aware batch translation failed")
                self.logger.debug(f"Error: {e}")

            # Strategy 3: Fallback to individual translation
            self.logger.info("Falling back to individual translation...")
            return self._translate_segments_individual(
                segments, translator, target_language
            )

        except Exception as e:
            self.logger.error(f"✗ Translation failed")
            return segments

    def _batch_translate_all(
        self, segments: List[Dict], translator: GoogleTranslator, target_language: str
    ) -> List[Dict]:
        """Strategy 1: Translate all text as one batch with retry logic"""
        # Combine all text with separators
        combined_text = "\n".join([segment["text"] for segment in segments])

        # Retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay

        for retry_attempt in range(max_retries + 1):
            try:
                # Translate the entire text
                translated_text = translator.translate(combined_text)

                # Split back into segments (simple approach)
                translated_lines = translated_text.split("\n")

                # Create new segments with translated text
                translated_segments = []
                for i, segment in enumerate(segments):
                    translated_text = (
                        translated_lines[i]
                        if i < len(translated_lines)
                        else segment["text"]
                    )
                    translated_segments.append(
                        {
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": translated_text,
                            "language": target_language,
                        }
                    )

                self.logger.info(
                    f"✓ Batch translated {len(translated_segments)} segments"
                )
                return translated_segments

            except Exception as e:
                if retry_attempt < max_retries:
                    self.logger.warning(
                        f"Batch translation failed (attempt {retry_attempt + 1}/{max_retries + 1})"
                    )
                    self.logger.debug(f"Error: {e}")
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    import time

                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.warning(
                        f"Batch translation failed after {max_retries + 1} attempts"
                    )
                    self.logger.debug(f"Error: {e}")
                    raise  # Re-raise to trigger fallback to other strategies

    def _context_aware_batch_translate(
        self, segments: List[Dict], translator: GoogleTranslator, target_language: str
    ) -> List[Dict]:
        """Strategy 2: Context-aware batch translation with retry logic"""
        translated_segments = []

        # Process segments in batches of 100 for context
        batch_size = 50
        self.logger.info(
            f"Processing {len(segments)} segments in batches of {batch_size}..."
        )
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(segments) + batch_size - 1) // batch_size
            self.logger.info(f"Processing batch {batch_num} of {total_batches}...")

            # Combine batch text with clear separators
            batch_text = "\n---\n".join([segment["text"] for segment in batch])

            # Retry logic with exponential backoff
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay

            for retry_attempt in range(max_retries + 1):
                try:
                    # Translate the batch
                    translated_batch_text = translator.translate(batch_text)

                    # Split by separator and create segments
                    translated_batch_lines = translated_batch_text.split("\n---\n")

                    for j, segment in enumerate(batch):
                        translated_text = (
                            translated_batch_lines[j]
                            if j < len(translated_batch_lines)
                            else segment["text"]
                        )
                        translated_segments.append(
                            {
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": translated_text,
                                "language": target_language,
                            }
                        )

                    # Success - break out of retry loop
                    break

                except Exception as e:
                    if retry_attempt < max_retries:
                        self.logger.warning(
                            f"Batch {batch_num} translation failed (attempt {retry_attempt + 1}/{max_retries + 1})"
                        )
                        self.logger.debug(f"Error: {e}")
                        self.logger.info(f"Retrying in {retry_delay} seconds...")
                        import time

                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        self.logger.warning(
                            f"Batch {batch_num} translation failed after {max_retries + 1} attempts"
                        )
                        self.logger.debug(f"Error: {e}")
                        # Fallback to original text for this batch
                        for segment in batch:
                            translated_segments.append(
                                {
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": segment["text"],
                                    "language": target_language,
                                }
                            )

        self.logger.info(
            f"✓ Context-aware batch translated {len(translated_segments)} segments"
        )
        return translated_segments

    def _translate_segments_individual(
        self, segments: List[Dict], translator: GoogleTranslator, target_language: str
    ) -> List[Dict]:
        """Fallback: Translate segments individually with retry logic"""
        translated_segments = []

        for i, segment in enumerate(segments, 1):
            # Retry logic with exponential backoff
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay

            for retry_attempt in range(max_retries + 1):
                try:
                    # Translate the text
                    translated_text = translator.translate(segment["text"])

                    # Create new segment with translated text
                    translated_segment = {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": translated_text,
                        "language": target_language,
                    }
                    translated_segments.append(translated_segment)

                    # Success - break out of retry loop
                    break

                except Exception as e:
                    if retry_attempt < max_retries:
                        self.logger.warning(
                            f"Failed to translate segment {i} (attempt {retry_attempt + 1}/{max_retries + 1})"
                        )
                        self.logger.debug(f"Error: {e}")
                        self.logger.info(f"Retrying in {retry_delay} seconds...")
                        import time

                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        self.logger.warning(
                            f"Failed to translate segment {i} after {max_retries + 1} attempts"
                        )
                        self.logger.debug(f"Error: {e}")
                        # Keep original text if translation fails
                        translated_segments.append(segment)

        self.logger.info(f"✓ Translated {len(translated_segments)} segments")
        return translated_segments

    def generate_subtitles(self, segments: List[Dict], output_path: str) -> bool:
        """Generate SRT subtitle file"""
        try:
            self.logger.info("Generating SRT subtitles...")

            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = self._format_timestamp(segment["start"])
                    end_time = self._format_timestamp(segment["end"])

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{segment['text']}\n\n")

            self.logger.info(f"✓ SRT subtitles saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"✗ Failed to generate subtitles: {e}")
            return False

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def process_video(
        self,
        video_path: str,
        source_language: str,
        target_language: str,
        output_dir: str = None,
    ) -> bool:
        """Main processing function - streamlined pipeline"""

        video_path = Path(video_path)
        if not video_path.exists():
            self.logger.error(f"✗ Video file not found: {video_path}")
            return False

        if output_dir is None:
            output_dir = video_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        # Generate output path
        base_name = video_path.stem
        subtitle_path = output_dir / f"{base_name}_{target_language}.srt"

        # Extract audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        if not self.extract_audio(str(video_path), temp_audio_path):
            return False

        try:
            # Step 1: Transcribe using Whisper
            segments = self.transcribe_with_whisper(
                temp_audio_path,
                source_language,
            )

            # Step 2: Apply translation if needed
            if source_language != target_language:
                segments = self.translate_segments(segments, target_language)
            else:
                self.logger.info(
                    "Source and target languages are the same, skipping translation..."
                )

            # Step 3: Generate subtitles
            if not self.generate_subtitles(segments, str(subtitle_path)):
                self.logger.info("Failed to generate subtitles")
                return False

            self.logger.info("✓ Processing completed successfully!")
            self.logger.info(f"Subtitle file: {subtitle_path}")

            return True

        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="Video Translation and Subtitle Generator - Streamlined"
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--source-language",
        "-s",
        required=True,
        help="Source language (e.g., 'ru' for Russian, 'es' for Spanish)",
    )
    parser.add_argument(
        "--target-language", "-t", default="en", help="Target language (default: en)"
    )
    parser.add_argument(
        "--output-dir", help="Output directory (default: same as input video)"
    )
    parser.add_argument(
        "--model-size",
        "-m",
        default=DEFAULT_MODEL_SIZE,
        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE})",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=DEFAULT_DEVICE,
        help=f"Device for whisper (default: {DEFAULT_DEVICE})",
    )

    args = parser.parse_args()

    translator = VideoTranslator(model_size=args.model_size, device=args.device)

    success = translator.process_video(
        video_path=args.video_path,
        source_language=args.source_language,
        target_language=args.target_language,
        output_dir=args.output_dir,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
