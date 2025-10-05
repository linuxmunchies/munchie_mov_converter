#!/usr/bin/env python3
"""
Batch MOV to MP4 Converter with Quality Preservation
====================================================

This program batch converts .mov files to MP4 format while preserving maximum quality.
It automatically detects whether files can be transmuxed (codec copy) or require re-encoding.

Features:
- Cross-platform (Windows, macOS, Linux)
- Automatic codec detection via FFprobe
- Lossless transmuxing when possible
- High-quality re-encoding (CRF 17-23) when necessary
- Progress tracking and detailed logging
- Error handling for corrupted files
- Optional parallel processing

Requirements:
- Python 3.7+
- ffmpeg-python: pip install ffmpeg-python
- FFmpeg installed and accessible in PATH
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import ffmpeg
except ImportError:
    print("ERROR: ffmpeg-python library not found.")
    print("Install it with: pip install ffmpeg-python")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class CodecInfo:
    """Store codec information from FFprobe analysis"""
    video_codec: str
    audio_codec: str
    can_transmux: bool
    error: Optional[str] = None


@dataclass
class ConversionResult:
    """Store conversion result information"""
    input_file: str
    output_file: str
    success: bool
    method: str  # 'transmux' or 'reencode'
    error: Optional[str] = None


class VideoConverter:
    """Handles video conversion operations"""

    # MP4-compatible codecs that can be transmuxed without re-encoding
    MP4_COMPATIBLE_VIDEO_CODECS = {
        'h264', 'h.264', 'avc', 'avc1',
        'h265', 'h.265', 'hevc', 'hev1', 'hvc1',
        'mpeg4', 'mp4v'
    }

    MP4_COMPATIBLE_AUDIO_CODECS = {
        'aac', 'mp3', 'mp2', 'ac3', 'eac3'
    }

    def __init__(self, quality: int = 20, preset: str = 'medium'):
        """
        Initialize VideoConverter

        Args:
            quality: CRF value for re-encoding (17-23 recommended, lower=better)
            preset: FFmpeg preset (ultrafast, superfast, veryfast, faster, fast,
                    medium, slow, slower, veryslow)
        """
        self.quality = quality
        self.preset = preset
        self._validate_ffmpeg()

    @staticmethod
    def _validate_ffmpeg():
        """Validate that FFmpeg is installed and accessible"""
        try:
            ffmpeg.probe('non_existent_file.mp4')
        except ffmpeg.Error:
            # Expected error for non-existent file, but FFmpeg is working
            pass
        except FileNotFoundError:
            logger.error("FFmpeg not found in PATH. Please install FFmpeg.")
            sys.exit(1)

    def detect_codecs(self, input_file: str) -> CodecInfo:
        """
        Detect video and audio codecs using FFprobe

        Args:
            input_file: Path to input video file

        Returns:
            CodecInfo object with codec information
        """
        try:
            probe = ffmpeg.probe(input_file)

            video_codec = None
            audio_codec = None

            # Extract codec information from streams
            for stream in probe.get('streams', []):
                codec_type = stream.get('codec_type', '')
                codec_name = stream.get('codec_name', '').lower()

                if codec_type == 'video' and not video_codec:
                    video_codec = codec_name
                elif codec_type == 'audio' and not audio_codec:
                    audio_codec = codec_name

            # Determine if transmuxing is possible
            can_transmux = (
                video_codec in self.MP4_COMPATIBLE_VIDEO_CODECS and
                (audio_codec in self.MP4_COMPATIBLE_AUDIO_CODECS or audio_codec is None)
            )

            return CodecInfo(
                video_codec=video_codec or 'unknown',
                audio_codec=audio_codec or 'none',
                can_transmux=can_transmux
            )

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFprobe error for {input_file}: {error_msg}")
            return CodecInfo(
                video_codec='unknown',
                audio_codec='unknown',
                can_transmux=False,
                error=error_msg
            )
        except Exception as e:
            logger.error(f"Unexpected error probing {input_file}: {str(e)}")
            return CodecInfo(
                video_codec='unknown',
                audio_codec='unknown',
                can_transmux=False,
                error=str(e)
            )

    def transmux(self, input_file: str, output_file: str) -> ConversionResult:
        """
        Transmux (copy codecs without re-encoding) for lossless conversion

        Args:
            input_file: Path to input .mov file
            output_file: Path to output .mp4 file

        Returns:
            ConversionResult object
        """
        try:
            logger.info(f"Transmuxing (lossless): {input_file}")

            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(
                stream,
                output_file,
                codec='copy',  # Copy both video and audio streams
                movflags='faststart'  # Optimize for web streaming
            )

            # Run with overwrite enabled
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            logger.info(f"✓ Transmux complete: {output_file}")
            return ConversionResult(
                input_file=input_file,
                output_file=output_file,
                success=True,
                method='transmux'
            )

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Transmux failed for {input_file}: {error_msg}")
            return ConversionResult(
                input_file=input_file,
                output_file=output_file,
                success=False,
                method='transmux',
                error=error_msg
            )

    def reencode(self, input_file: str, output_file: str, codec_info: CodecInfo) -> ConversionResult:
        """
        Re-encode video with high quality settings

        Args:
            input_file: Path to input .mov file
            output_file: Path to output .mp4 file
            codec_info: Detected codec information

        Returns:
            ConversionResult object
        """
        try:
            logger.info(f"Re-encoding (CRF {self.quality}): {input_file}")

            stream = ffmpeg.input(input_file)

            # Choose encoder based on source codec
            if codec_info.video_codec in {'h265', 'h.265', 'hevc', 'hev1', 'hvc1'}:
                # Use H.265 encoder for HEVC sources
                video_encoder = 'libx265'
                extra_args = {
                    'crf': str(self.quality),
                    'preset': self.preset,
                    'x265-params': 'log-level=error'
                }
            else:
                # Use H.264 encoder for other sources
                video_encoder = 'libx264'
                extra_args = {
                    'crf': str(self.quality),
                    'preset': self.preset
                }

            # Handle audio encoding
            if codec_info.audio_codec in self.MP4_COMPATIBLE_AUDIO_CODECS:
                audio_encoder = 'copy'  # Copy compatible audio
            elif codec_info.audio_codec == 'none':
                audio_encoder = None  # No audio stream
            else:
                audio_encoder = 'aac'  # Re-encode incompatible audio to AAC
                extra_args['audio_bitrate'] = '192k'

            # Build output stream
            output_args = {
                'video_codec': video_encoder,
                'movflags': 'faststart',
                **extra_args
            }

            if audio_encoder:
                output_args['audio_codec'] = audio_encoder

            stream = ffmpeg.output(stream, output_file, **output_args)

            # Run conversion
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            logger.info(f"✓ Re-encode complete: {output_file}")
            return ConversionResult(
                input_file=input_file,
                output_file=output_file,
                success=True,
                method='reencode'
            )

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"Re-encode failed for {input_file}: {error_msg}")
            return ConversionResult(
                input_file=input_file,
                output_file=output_file,
                success=False,
                method='reencode',
                error=error_msg
            )

    def convert_file(self, input_file: str, output_file: str) -> ConversionResult:
        """
        Convert a single file (automatically choosing transmux or re-encode)

        Args:
            input_file: Path to input .mov file
            output_file: Path to output .mp4 file

        Returns:
            ConversionResult object
        """
        # Detect codecs
        codec_info = self.detect_codecs(input_file)

        if codec_info.error:
            return ConversionResult(
                input_file=input_file,
                output_file=output_file,
                success=False,
                method='detection',
                error=codec_info.error
            )

        logger.info(f"File: {input_file}")
        logger.info(f"  Video: {codec_info.video_codec} | Audio: {codec_info.audio_codec}")

        # Choose conversion method
        if codec_info.can_transmux:
            logger.info("  Method: Transmux (lossless, no re-encoding)")
            return self.transmux(input_file, output_file)
        else:
            logger.info("  Method: Re-encode (high quality)")
            return self.reencode(input_file, output_file, codec_info)


def process_single_file(args: Tuple[str, str, int, str]) -> ConversionResult:
    """
    Process a single file (used for parallel processing)

    Args:
        args: Tuple of (input_file, output_file, quality, preset)

    Returns:
        ConversionResult object
    """
    input_file, output_file, quality, preset = args
    converter = VideoConverter(quality=quality, preset=preset)
    return converter.convert_file(input_file, output_file)


def find_mov_files(input_dir: Path) -> List[Path]:
    """
    Find all .mov files in the input directory

    Args:
        input_dir: Directory to search

    Returns:
        List of Path objects for .mov files
    """
    mov_files = list(input_dir.glob('*.mov')) + list(input_dir.glob('*.MOV'))
    return sorted(mov_files)


def batch_convert(
    input_dir: str,
    output_dir: str,
    quality: int = 20,
    preset: str = 'medium',
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> Tuple[int, int]:
    """
    Batch convert all .mov files in a directory

    Args:
        input_dir: Input directory containing .mov files
        output_dir: Output directory for .mp4 files
        quality: CRF value for re-encoding (17-23 recommended)
        preset: FFmpeg preset
        parallel: Enable parallel processing
        max_workers: Maximum number of parallel workers (None = CPU count)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate directories
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find .mov files
    mov_files = find_mov_files(input_path)

    if not mov_files:
        logger.warning(f"No .mov files found in {input_dir}")
        return 0, 0

    logger.info(f"Found {len(mov_files)} .mov file(s) to convert")
    logger.info(f"Quality: CRF {quality} | Preset: {preset} | Parallel: {parallel}")
    logger.info("-" * 70)

    # Prepare conversion tasks
    tasks = []
    for mov_file in mov_files:
        output_file = output_path / f"{mov_file.stem}.mp4"
        tasks.append((str(mov_file), str(output_file), quality, preset))

    # Process files
    results = []

    if parallel and len(tasks) > 1:
        # Parallel processing
        logger.info(f"Using parallel processing with {max_workers or 'auto'} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, task): task for task in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                logger.info(f"Progress: {i}/{len(tasks)} files processed")
    else:
        # Sequential processing
        converter = VideoConverter(quality=quality, preset=preset)

        for i, (input_file, output_file, _, _) in enumerate(tasks, 1):
            logger.info(f"
Processing file {i}/{len(tasks)}")
            result = converter.convert_file(input_file, output_file)
            results.append(result)

    # Summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    logger.info("
" + "=" * 70)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    # Count methods used
    transmux_count = sum(1 for r in results if r.success and r.method == 'transmux')
    reencode_count = sum(1 for r in results if r.success and r.method == 'reencode')

    logger.info(f"
Conversion methods:")
    logger.info(f"  Transmux (lossless): {transmux_count}")
    logger.info(f"  Re-encode (high quality): {reencode_count}")

    if failed > 0:
        logger.info(f"
Failed conversions:")
        for result in results:
            if not result.success:
                logger.info(f"  - {result.input_file}")
                if result.error:
                    logger.info(f"    Error: {result.error[:200]}")

    logger.info("=" * 70)
    logger.info(f"Detailed log saved to: conversion.log")

    return successful, failed


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Batch convert MOV files to MP4 with quality preservation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (current directory)
  %(prog)s -i ./videos -o ./output

  # High quality (CRF 18, slow preset)
  %(prog)s -i ./videos -o ./output -q 18 -p slow

  # Parallel processing with 4 workers
  %(prog)s -i ./videos -o ./output --parallel -w 4

  # Near-lossless quality
  %(prog)s -i ./videos -o ./output -q 17 -p slower

CRF Values (lower = better quality, larger file):
  17-18: Near-lossless, very high quality
  19-20: High quality (recommended default)
  21-23: Good quality, smaller files

Presets (slower = better compression):
  ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing .mov files'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for .mp4 files'
    )

    parser.add_argument(
        '-q', '--quality',
        type=int,
        default=20,
        choices=range(0, 52),
        metavar='CRF',
        help='Quality (CRF value, 0-51): 17-23 recommended for high quality (default: 20)'
    )

    parser.add_argument(
        '-p', '--preset',
        default='medium',
        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                 'medium', 'slow', 'slower', 'veryslow'],
        help='Encoding preset (default: medium)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing for faster conversion'
    )

    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Run batch conversion
    try:
        successful, failed = batch_convert(
            input_dir=args.input,
            output_dir=args.output,
            quality=args.quality,
            preset=args.preset,
            parallel=args.parallel,
            max_workers=args.workers
        )

        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        logger.info("\nConversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
