#!/usr/bin/env python3

"""
Program: Video Word/Phrase Remover using AI (PyQt6 Version)
Description:
    This script provides a GUI (PyQt6) that allows the user to:
        1. Select an input directory containing video files (including subdirectories).
        2. Select an output directory to save the processed videos.
        3. Select a file containing words/phrases to detect and remove from audio.
        4. Use whisper-mps to detect word/phrase occurrences in the audio track.
        5. Remove (mute) the portions of audio containing the detected words/phrases.
        6. Display a progress bar and provide logging of steps.
        7. Export a transcript of each video's audio.

Features:
    - Supports multiple video formats: mp4, mkv, avi, mov, flv, wmv, mpeg, mpg, ogg, webm, etc.
    - Keeps the original resolution.
    - Exports with the same file format as the input.
    - Has a GUI to set input directory, output directory, and words file.
    - Provides a progress bar and logging.
    - Exports a text transcript for each processed video.

Dependencies:
    - Python 3.x
    - pip install:
        PyQt6
        moviepy==1.0.3
        pydub
        whisper-mps
        torch
    - ffmpeg (must be installed and accessible on PATH)

Usage:
    1. Install the above dependencies.
    2. Run this script: python video_word_remover_pyqt.py
    3. Fill in the fields in the GUI and click "Run".
    4. The processed files will be saved in the specified output directory.
    5. A transcript file (ending with `_transcript.txt`) will be saved alongside the output video.
"""

import os
import sys
import threading
import logging

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# From the whisper-mps repository
from whisper_mps.whisper.transcribe import transcribe

import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from moviepy.editor import AudioFileClip, VideoFileClip, CompositeVideoClip
from pydub import AudioSegment

# Configure logging
logging.basicConfig(
    filename='video_word_removal.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class Worker(QThread):
    """
    Worker thread to process videos in the background so the UI remains responsive.
    """
    progress_signal = pyqtSignal(float)
    message_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    def __init__(self, input_dir, output_dir, words_file, parent=None):
        super().__init__(parent)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.words_file = words_file

    def run(self):
        """
        Main workflow to process video files:
            1) Collect video files from the input directory
            2) Load the list of words/phrases to remove
            3) For each video:
                - Extract audio
                - Transcribe (whisper-mps) and export transcript
                - Identify segments that contain unwanted words
                - Mute those segments
                - Merge back into video
        """
        try:
            self.log_and_emit("Starting video processing workflow.")
            video_files = self.collect_video_files()
            if not video_files:
                self.log_and_emit("No video files found in the selected directory.")
                self.done_signal.emit()
                return

            # Load words to remove
            removal_words = self.load_removal_words()

            total_files = len(video_files)
            completed = 0

            for video_path in video_files:
                self.log_and_emit(f"Processing video: {video_path}")
                rel_path = os.path.relpath(video_path, self.input_dir)
                output_video_path = os.path.join(self.output_dir, rel_path)

                # Create subdirectories in output path if needed
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

                # Extract audio
                audio_path = self.extract_audio(video_path)

                # Transcribe with whisper-mps, using the "base" model by default
                transcription = self.transcribe_audio(audio_path)
                # Export transcript as a simple .txt file
                self.export_transcript(transcription, output_video_path)

                # Identify timestamps to mute
                timestamps_to_mute = self.identify_mute_segments(
                    transcription, removal_words
                )

                # Mute those audio segments
                if timestamps_to_mute:
                    muted_audio_path = self.mute_audio_segments(
                        audio_path, timestamps_to_mute
                    )
                else:
                    # If no segments need muting, use original audio
                    self.log_and_emit("No segments matched any removal words.")
                    muted_audio_path = audio_path

                # Merge muted audio with original video
                self.merge_audio_with_video(video_path, muted_audio_path, output_video_path)

                # Cleanup temp audio
                self.cleanup_files(audio_path, muted_audio_path)

                completed += 1
                progress_percent = (completed / total_files) * 100
                self.progress_signal.emit(progress_percent)

            # Finished processing
            self.log_and_emit("Processing completed for all videos.")
        except Exception as e:
            logging.error(str(e))
            self.error_signal.emit(f"An error occurred: {e}")

        self.done_signal.emit()

    def collect_video_files(self):
        """
        Collect all supported video paths from the input directory.
        """
        supported_ext = (
            ".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv",
            ".mpeg", ".mpg", ".ogg", ".webm"
        )
        video_files = []
        for root, _, files in os.walk(self.input_dir):
            for f in files:
                if f.lower().endswith(supported_ext):
                    full_path = os.path.join(root, f)
                    video_files.append(full_path)

        self.log_and_emit(f"Found {len(video_files)} video files in {self.input_dir}.")
        return video_files

    def load_removal_words(self):
        """
        Load words/phrases to remove from the specified file.
        """
        removal_words = []
        with open(self.words_file, "r", encoding="utf-8") as wf:
            removal_words = [line.strip().lower() for line in wf if line.strip()]
        self.log_and_emit(f"Loaded {len(removal_words)} removal words/phrases.")
        return removal_words

    def extract_audio(self, video_path):
        """
        Extract audio from video using moviepy and save as temporary WAV file.
        """
        self.log_and_emit(f"Extracting audio from {video_path}")
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_path = video_path + "_temp_audio.wav"
        audio_clip.write_audiofile(audio_path, logger=None)
        video_clip.close()
        audio_clip.close()
        self.log_and_emit(f"Audio extracted to {audio_path}")
        return audio_path

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using whisper-mps's transcribe function.
        """
        self.log_and_emit(f"Transcribing audio: {audio_path}")
        # Using the "base" model by default; you can adjust as needed
        transcription = transcribe(
            audio_path, 
            model="base",
            task="transcribe",
            language="en"
            )
        self.log_and_emit("Transcription complete.")
        return transcription

    def export_transcript(self, transcription, output_video_path):
        """
        Export the full transcription text into a file parallel to the output video.
        """
        base_no_ext, _ = os.path.splitext(output_video_path)
        transcript_path = base_no_ext + "_transcript.txt"
        text = transcription.get("text", "").strip()
        self.log_and_emit(f"Exporting transcript to {transcript_path}")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

        self.log_and_emit(f"Transcript saved: {transcript_path}")

    def identify_mute_segments(self, transcription, removal_words):
        """
        Identify segments to mute by comparing each segment text against removal words.
        """
        self.log_and_emit("Identifying segments to mute based on removal words.")
        segments = transcription.get('segments', [])
        timestamps_to_mute = []

        for seg in segments:
            segment_text = seg.get('text', "").lower()
            start_time = seg.get('start', 0.0)
            end_time = seg.get('end', 0.0)
            for w in removal_words:
                if w in segment_text:
                    # Log the matched word
                    self.log_and_emit(
                        f"Found word '{w}' in segment [{start_time}, {end_time}]"
                    )
                    timestamps_to_mute.append((start_time, end_time))
                    break  # no need to check other words for this segment

        self.log_and_emit(
            f"Found {len(timestamps_to_mute)} segments matching removal words."
        )
        return timestamps_to_mute

    def mute_audio_segments(self, audio_path, segments):
        """
        Mute the segments in the audio by applying heavy negative gain via pydub.
        """
        self.log_and_emit("Muting matched segments in audio.")
        original_audio = AudioSegment.from_file(audio_path, format="wav")

        for start_time, end_time in segments:
            start_ms = int(start_time * 900)
            end_ms = int(end_time * 900)
            segment_to_mute = original_audio[start_ms:end_ms].apply_gain(-100.0)
            original_audio = (
                original_audio[:start_ms] + segment_to_mute + original_audio[end_ms:]
            )

        muted_audio_path = audio_path.replace("_temp_audio.wav", "_temp_muted.wav")
        original_audio.export(muted_audio_path, format="wav")
        self.log_and_emit(f"Exported muted audio to {muted_audio_path}")
        return muted_audio_path

    def merge_audio_with_video(self, video_path, audio_path, output_video_path):
        """
        Merge the muted audio with the original video track, preserving resolution
        and container format. Compatible with MoviePy 1.0.x
        """
        self.log_and_emit(
            f"Merging muted audio with original video.\n"
            f"Video: {video_path}\nMuted audio: {audio_path}"
        )
        try:
            _, ext = os.path.splitext(video_path)
            ext = ext.lower()

            video_clip = VideoFileClip(video_path)
            new_audio_clip = AudioFileClip(audio_path)

            # MoviePy 1.0.x doesn't have set_audio() or an 'audio=' argument in CompositeVideoClip
            # so we do it step-by-step:
            final_clip = CompositeVideoClip([video_clip.set_duration(video_clip.duration)])
            final_clip.audio = new_audio_clip

            # Ensure output path has the same extension
            base_no_ext, _ = os.path.splitext(output_video_path)
            output_video_path_correct_ext = base_no_ext + ext

            self.log_and_emit(f"Writing final video to: {output_video_path_correct_ext}")
            final_clip.write_videofile(
                output_video_path_correct_ext,
                codec="h264_videotoolbox",
                audio_codec="aac",
                ffmpeg_params=["-q:v", "50"],
                logger=None
            )

            video_clip.close()
            new_audio_clip.close()
            final_clip.close()
            self.log_and_emit("Merging complete.")
        except Exception as e:
            raise RuntimeError(f"Error merging audio with video: {e}")

    def cleanup_files(self, audio_path, muted_audio_path):
        """
        Remove temporary audio files to keep things clean.
        """
        self.log_and_emit("Cleaning up temporary audio files.")
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                self.log_and_emit(f"Deleted temp file: {audio_path}")
            except Exception as e:
                self.log_and_emit(f"Error deleting temp file {audio_path}: {e}")

        if muted_audio_path != audio_path and os.path.exists(muted_audio_path):
            try:
                os.remove(muted_audio_path)
                self.log_and_emit(f"Deleted temp file: {muted_audio_path}")
            except Exception as e:
                self.log_and_emit(f"Error deleting temp file {muted_audio_path}: {e}")

    def log_and_emit(self, message):
        """
        Log a message to file and emit it to the GUI thread.
        """
        logging.info(message)
        self.message_signal.emit(message)


class VideoWordRemoverGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Word/Phrase Remover (PyQt6)")
        self.setGeometry(100, 100, 600, 200)

        # Main widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Layouts
        self.main_layout = QVBoxLayout()
        self.input_layout = QHBoxLayout()
        self.output_layout = QHBoxLayout()
        self.words_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()

        # Input Directory
        self.lbl_input_dir = QLabel("Input Directory:")
        self.edit_input_dir = QLineEdit()
        self.btn_input_dir = QPushButton("Browse")
        self.btn_input_dir.clicked.connect(self.browse_input_dir)
        self.input_layout.addWidget(self.lbl_input_dir)
        self.input_layout.addWidget(self.edit_input_dir)
        self.input_layout.addWidget(self.btn_input_dir)

        # Output Directory
        self.lbl_output_dir = QLabel("Output Directory:")
        self.edit_output_dir = QLineEdit()
        self.btn_output_dir = QPushButton("Browse")
        self.btn_output_dir.clicked.connect(self.browse_output_dir)
        self.output_layout.addWidget(self.lbl_output_dir)
        self.output_layout.addWidget(self.edit_output_dir)
        self.output_layout.addWidget(self.btn_output_dir)

        # Words File
        self.lbl_words_file = QLabel("Word/Phrase File:")
        self.edit_words_file = QLineEdit()
        self.btn_words_file = QPushButton("Browse")
        self.btn_words_file.clicked.connect(self.browse_words_file)
        self.words_layout.addWidget(self.lbl_words_file)
        self.words_layout.addWidget(self.edit_words_file)
        self.words_layout.addWidget(self.btn_words_file)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Run button
        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self.run_removal)
        self.button_layout.addWidget(self.btn_run)

        # Combine layouts
        self.main_layout.addLayout(self.input_layout)
        self.main_layout.addLayout(self.output_layout)
        self.main_layout.addLayout(self.words_layout)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addLayout(self.button_layout)

        self.main_widget.setLayout(self.main_layout)

        self.worker_thread = None  # Will hold our Worker QThread

    def browse_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.edit_input_dir.setText(directory)

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.edit_output_dir.setText(directory)

    def browse_words_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Word/Phrase File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.edit_words_file.setText(file_path)

    def run_removal(self):
        in_dir = self.edit_input_dir.text().strip()
        out_dir = self.edit_output_dir.text().strip()
        words_file = self.edit_words_file.text().strip()

        # Validate
        if not os.path.isdir(in_dir):
            QMessageBox.critical(self, "Error", "Input directory is invalid.")
            return
        if not os.path.isdir(out_dir):
            QMessageBox.critical(self, "Error", "Output directory is invalid.")
            return
        if not os.path.isfile(words_file):
            QMessageBox.critical(self, "Error", "Word/Phrase file is invalid.")
            return

        # Disable run button during processing
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)

        # Start the worker thread
        self.worker_thread = Worker(in_dir, out_dir, words_file)
        self.worker_thread.progress_signal.connect(self.on_progress_update)
        self.worker_thread.message_signal.connect(self.on_message)
        self.worker_thread.error_signal.connect(self.on_error)
        self.worker_thread.done_signal.connect(self.on_done)
        self.worker_thread.start()

    def on_progress_update(self, value):
        self.progress_bar.setValue(int(value))

    def on_message(self, message):
        # Print to console and also log
        print(message)
        logging.info(message)

    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def on_done(self):
        self.btn_run.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = VideoWordRemoverGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()