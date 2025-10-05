# MOV to MP4 Batch Converter - Quick Start Guide

## Installation

1. **Install Python 3.7+** (if not already installed)

2. **Install FFmpeg:**
   - **Fedora/RHEL:** `sudo dnf install ffmpeg`
   - **Ubuntu/Debian:** `sudo apt install ffmpeg`
   - **macOS:** `brew install ffmpeg`
   - **Windows:** Download from https://ffmpeg.org/download.html

3. **Install Python dependencies:**
   ```bash
   pip install ffmpeg-python
   ```

## Basic Usage

```bash
# Convert all .mov files in a directory
python mov_to_mp4_converter.py -i /path/to/input -o /path/to/output

# High quality conversion (CRF 18)
python mov_to_mp4_converter.py -i ./videos -o ./output -q 18 -p slow

# Parallel processing (faster for multiple files)
python mov_to_mp4_converter.py -i ./videos -o ./output --parallel -w 4
```

## Quality Settings

### CRF Values (Constant Rate Factor)
- **17-18:** Near-lossless, visually identical (large files)
- **19-20:** High quality, minimal perceptible loss (recommended)
- **21-23:** Good quality, balanced size
- **24+:** Lower quality, smaller files

### Presets (encoding speed vs compression)
- **fast/faster/veryfast:** Faster encoding, larger files
- **medium:** Balanced (default)
- **slow/slower/veryslow:** Slower encoding, better compression

## Conversion Methods

The program automatically chooses the best method:

### 1. Transmux (Lossless)
- Used when .mov already contains MP4-compatible codecs (H.264/H.265 + AAC)
- Zero quality loss (just container change)
- Very fast (no re-encoding)

### 2. Re-encode (High Quality)
- Used when codecs need conversion
- Uses CRF mode for quality preservation
- Slower but maintains high quality

## Command Line Options

```
-i, --input       Input directory with .mov files (required)
-o, --output      Output directory for .mp4 files (required)
-q, --quality     Quality CRF value 0-51 (default: 20)
-p, --preset      Encoding preset (default: medium)
--parallel        Enable parallel processing
-w, --workers     Number of parallel workers
-v, --verbose     Verbose logging
```

## Examples

### Example 1: Standard Conversion
```bash
python mov_to_mp4_converter.py -i ~/Downloads/videos -o ~/Videos/converted
```

### Example 2: Maximum Quality
```bash
python mov_to_mp4_converter.py -i ./input -o ./output -q 17 -p slower
```

### Example 3: Fast Batch Processing
```bash
python mov_to_mp4_converter.py -i ./videos -o ./output --parallel -w 8
```

### Example 4: Balanced Quality & Speed
```bash
python mov_to_mp4_converter.py -i ./videos -o ./output -q 20 -p medium
```

## Output

The program provides:
- Real-time progress updates
- Codec detection results for each file
- Conversion method used (transmux vs re-encode)
- Summary statistics
- Detailed log file (`conversion.log`)

## Troubleshooting

### "FFmpeg not found"
Ensure FFmpeg is installed and in your system PATH.

### "Module 'ffmpeg' not found"
Install the Python library: `pip install ffmpeg-python`

### Conversion fails for specific files
Check `conversion.log` for detailed error messages. Files may be corrupted or use unsupported codecs.

### Out of memory errors
- Use parallel processing with fewer workers: `-w 2`
- Process files sequentially (don't use `--parallel`)
- Close other applications

## Quality vs File Size Guide

For 330+ files, here's what to expect:

| CRF | Quality      | File Size  | Speed    |
|-----|-------------|-----------|----------|
| 17  | Near-lossless | ~100%     | Slowest  |
| 18  | Excellent    | ~80-90%   | Very slow |
| 20  | High         | ~60-70%   | Medium   |
| 23  | Good         | ~40-50%   | Fast     |

## Performance Tips

1. **Use parallel processing** for large batches: `--parallel -w 4`
2. **Choose faster preset** if time is critical: `-p fast`
3. **Monitor first few files** to verify quality before batch processing
4. **Free up disk space** - conversions need space for both input and output

## Technical Details

- **Transmux:** Uses codec copy (`-c copy`) for lossless container change
- **Re-encode:** Uses libx264 (H.264) or libx265 (H.265) with CRF mode
- **Audio:** Copies AAC/MP3/AC3, re-encodes other codecs to AAC @ 192kbps
- **Optimization:** Adds `movflags=faststart` for web streaming compatibility
