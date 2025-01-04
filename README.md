# Video Mute Script

A Python script to process videos and manage audio-related operations.

## Features
- Mute audio in videos.
- Uses `whisper_mps` for transcription and other processing tasks.

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```  
2. Add desired words to mute to file:
   ```bash
   Add words to the words.txt file
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python video-mute.py
   ```

## Requirements
- Python 3.x
- FFmpeg
- Required Python packages listed in `requirements.txt`.

## TODO
- Add support for customized mute time intervals.
- Integrate GUI for device selection used by `whisper_mps`.
- Extend compatibility for platforms beyond Metal (MPS).
- Add alternative to FFMPEG that supports Metal.

## Credits
This project uses the following open-source packages:
- [`whisper_mps`](https://github.com/AtomGradient/whisper-mps) for transcription functionality.
- Any additional packages included in `requirements.txt`.

## License
[MIT License](LICENSE)

