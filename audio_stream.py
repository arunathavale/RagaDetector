"""
Audio Stream Module - Threaded PyAudio implementation for live audio capture
Safely opens Mac microphone stream, processes audio chunks, and filters silence
"""

import pyaudio
import numpy as np
import threading
from collections import deque
import time
import logging
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    CHANNELS,
    AUDIO_DEVICE,
    LOG_LEVEL,
    LOG_FILE,
    LOG_TO_CONSOLE
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler() if LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioStream:
    """
    Threaded audio stream handler using PyAudio.
    Captures live audio from microphone, applies adaptive RMS threshold,
    and places audio chunks into a thread-safe queue.
    """
    
    def __init__(self, 
                 sample_rate=SAMPLE_RATE,
                 buffer_size=BUFFER_SIZE,
                 channels=CHANNELS,
                 device_index=AUDIO_DEVICE,
                 adaptive_threshold=True,
                 initial_rms_threshold=0.01,
                 threshold_adaptation_rate=0.001,
                 min_rms_threshold=0.005,
                 max_rms_threshold=0.05):
        """
        Initialize audio stream handler.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Number of samples per chunk
            channels: Number of audio channels (1=mono, 2=stereo)
            device_index: PyAudio device index (None for default)
            adaptive_threshold: Enable adaptive RMS threshold
            initial_rms_threshold: Initial RMS threshold for silence detection
            threshold_adaptation_rate: Rate at which threshold adapts
            min_rms_threshold: Minimum RMS threshold
            max_rms_threshold: Maximum RMS threshold
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        self.device_index = device_index
        self.adaptive_threshold = adaptive_threshold
        self.rms_threshold = initial_rms_threshold
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.min_rms_threshold = min_rms_threshold
        self.max_rms_threshold = max_rms_threshold
        
        # Thread-safe deque for audio chunks (drops oldest when full)
        self.audio_queue = deque(maxlen=100)
        
        # PyAudio instance
        self.pa = None
        self.stream = None
        
        # Threading control
        self.is_running = False
        self.capture_thread = None
        
        # Statistics
        self.total_chunks = 0
        self.silent_chunks = 0
        self.active_chunks = 0
        
        logger.info(f"AudioStream initialized: {sample_rate}Hz, {buffer_size} samples, {channels} channel(s)")
    
    def list_devices(self):
        """List available audio devices."""
        pa = pyaudio.PyAudio()
        logger.info("Available audio devices:")
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                logger.info(f"  Device {i}: {device_info['name']} "
                          f"(channels: {device_info['maxInputChannels']}, "
                          f"sample rate: {int(device_info['defaultSampleRate'])}Hz)")
        pa.terminate()
    
    def _calculate_rms(self, audio_data):
        """
        Calculate RMS (Root Mean Square) of audio data.
        
        Args:
            audio_data: numpy array of audio samples
        
        Returns:
            RMS value
        """
        return np.sqrt(np.mean(audio_data ** 2))
    
    def _adapt_threshold(self, current_rms):
        """
        Adaptively adjust RMS threshold based on current audio level.
        
        Args:
            current_rms: Current RMS value of audio chunk
        """
        if not self.adaptive_threshold:
            return
        
        # Gradually adapt threshold towards current RMS level
        # This helps the system adjust to changing noise floors
        if current_rms > self.rms_threshold:
            # Increase threshold slowly when above current threshold
            self.rms_threshold += self.threshold_adaptation_rate * (current_rms - self.rms_threshold)
        else:
            # Decrease threshold slowly when below current threshold
            self.rms_threshold -= self.threshold_adaptation_rate * (self.rms_threshold - current_rms)
        
        # Clamp threshold to valid range
        self.rms_threshold = np.clip(self.rms_threshold, 
                                      self.min_rms_threshold, 
                                      self.max_rms_threshold)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function for processing audio chunks.
        
        Args:
            in_data: Raw audio data from stream
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status
        
        Returns:
            Tuple of (audio_data, pyaudio.paContinue)
        """
        if status:
            logger.warning(f"Stream status: {status}")
        
        try:
            # Convert raw bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Reshape if stereo
            if self.channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            
            # Calculate RMS
            rms = self._calculate_rms(audio_data)
            
            # Adapt threshold if enabled
            self._adapt_threshold(rms)
            
            # Filter silence
            if rms > self.rms_threshold:
                # Audio is above threshold, add to deque
                # Deque automatically drops oldest when full
                self.audio_queue.append(audio_data)
                self.active_chunks += 1
            else:
                # Audio is below threshold (silence)
                self.silent_chunks += 1
            
            self.total_chunks += 1
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)
    
    def _capture_loop(self):
        """Main capture loop that runs in separate thread."""
        logger.info("Audio capture thread started")
        
        try:
            while self.is_running:
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
        finally:
            logger.info("Audio capture thread stopped")
    
    def start(self):
        """Start audio stream and capture thread."""
        if self.is_running:
            logger.warning("Audio stream already running")
            return
        
        try:
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Determine device index
            device_index = self.device_index
            if device_index is None:
                # Try to find default input device
                device_index = self.pa.get_default_input_device_info()['index']
                logger.info(f"Using default input device: {device_index}")
            
            # Get device info for validation
            device_info = self.pa.get_device_info_by_index(device_index)
            logger.info(f"Opening device: {device_info['name']}")
            
            # Validate sample rate
            supported_sample_rate = int(device_info['defaultSampleRate'])
            if self.sample_rate != supported_sample_rate:
                logger.warning(f"Sample rate {self.sample_rate}Hz may not be supported. "
                            f"Device default: {supported_sample_rate}Hz")
            
            # Open audio stream
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("Audio stream started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop audio stream and cleanup resources."""
        if not self.is_running:
            return
        
        logger.info("Stopping audio stream...")
        self.is_running = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        # Stop and close stream
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
            self.stream = None
        
        # Terminate PyAudio
        if self.pa:
            try:
                self.pa.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            self.pa = None
        
        # Log statistics
        logger.info(f"Audio stream stopped. Statistics:")
        logger.info(f"  Total chunks: {self.total_chunks}")
        logger.info(f"  Active chunks: {self.active_chunks}")
        logger.info(f"  Silent chunks: {self.silent_chunks}")
        logger.info(f"  Final RMS threshold: {self.rms_threshold:.6f}")
    
    def get_audio_chunk(self, timeout=0.1):
        """
        Get audio chunk from deque.
        
        Args:
            timeout: Maximum time to wait for chunk in seconds
        
        Returns:
            numpy array of audio samples, or None if empty
        """
        if len(self.audio_queue) > 0:
            return self.audio_queue.popleft()
        return None
    
    def get_current_rms(self):
        """
        Get the current RMS level of the most recent audio chunk.
        
        Returns:
            Current RMS value, or 0.0 if no recent chunk
        """
        try:
            # Get the most recent chunk without removing it
            if len(self.audio_queue) > 0:
                chunk = self.audio_queue[-1]
                return self._calculate_rms(chunk)
            return 0.0
        except Exception:
            return 0.0
    
    def is_below_silence_threshold(self):
        """
        Check if current audio is below silence threshold.
        
        Returns:
            True if below threshold (silence), False otherwise
        """
        # If no audio data available, don't treat as silence
        if len(self.audio_queue) == 0:
            return False
        
        current_rms = self.get_current_rms()
        return current_rms < self.rms_threshold
    
    def get_statistics(self):
        """
        Get current statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_chunks': self.total_chunks,
            'active_chunks': self.active_chunks,
            'silent_chunks': self.silent_chunks,
            'rms_threshold': self.rms_threshold,
            'queue_size': len(self.audio_queue)
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


if __name__ == '__main__':
    # Test the audio stream
    import sys
    
    print("Audio Stream Test")
    print("=================")
    
    # List available devices
    audio_stream = AudioStream()
    audio_stream.list_devices()
    
    print("\nStarting audio capture for 5 seconds...")
    print("Speak or play music to test the stream.\n")
    
    try:
        with AudioStream() as stream:
            start_time = time.time()
            chunk_count = 0
            
            while time.time() - start_time < 5:
                chunk = stream.get_audio_chunk(timeout=0.1)
                if chunk is not None:
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        stats = stream.get_statistics()
                        print(f"Chunks: {chunk_count}, "
                              f"RMS threshold: {stats['rms_threshold']:.6f}, "
                              f"Queue size: {stats['queue_size']}")
                
                # Print statistics every second
                if int(time.time() - start_time) > chunk_count // 10:
                    stats = stream.get_statistics()
                    print(f"Statistics: {stats}")
            
            print(f"\nTest completed. Received {chunk_count} audio chunks.")
            print(f"Final statistics: {stream.get_statistics()}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        sys.exit(1)
