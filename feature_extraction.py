"""
Feature Extraction Module - Pitch detection, tonic normalization, and histogram processing
Handles conversion of raw audio to 120-bin pitch class histograms for Raaga identification
"""

import numpy as np
import librosa
import logging
from config import (
    SAMPLE_RATE,
    BUFFER_SIZE,
    MIN_FREQUENCY,
    MAX_FREQUENCY,
    HISTOGRAM_BINS,
    BINS_PER_SEMITONE,
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


class FeatureExtractor:
    """
    Extracts pitch features from audio chunks and converts to 120-bin histograms.
    
    Core computational steps:
    1. Pitch Extraction: Autocorrelation-based fundamental frequency detection
    2. Tonic Normalization: Convert Hz to cents relative to tonic (Sa)
    3. Feature Processing: Wrap to 0-1200 cent octave and bin into 120-bin histogram
    """
    
    def __init__(self, 
                 sample_rate=SAMPLE_RATE,
                 min_freq=MIN_FREQUENCY,
                 max_freq=MAX_FREQUENCY,
                 histogram_bins=HISTOGRAM_BINS,
                 tonic_frequency=None):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            min_freq: Minimum frequency for pitch detection (Hz)
            max_freq: Maximum frequency for pitch detection (Hz)
            histogram_bins: Number of bins for pitch class histogram
            tonic_frequency: Manual tonic (Sa) frequency in Hz (None for auto-detection)
        """
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.histogram_bins = histogram_bins
        self.tonic_frequency = tonic_frequency
        
        # Cents per bin (1200 cents / 120 bins = 10 cents per bin)
        self.cents_per_bin = 1200.0 / histogram_bins
        
        # Internal histogram for incremental updates
        self.histogram = np.zeros(histogram_bins)
        
        # Statistics
        self.total_frames_processed = 0
        self.successful_detections = 0
        self.failed_detections = 0
        
        logger.info(f"FeatureExtractor initialized: sample_rate={sample_rate}Hz, "
                   f"freq_range={min_freq}-{max_freq}Hz, "
                   f"histogram_bins={histogram_bins}, "
                   f"tonic={'auto' if tonic_frequency is None else f'{tonic_frequency}Hz'}")
    
    def set_tonic(self, tonic_frequency):
        """
        Manually set the tonic (Sa) frequency.
        
        Args:
            tonic_frequency: Tonic frequency in Hz
        """
        if tonic_frequency <= 0:
            raise ValueError("Tonic frequency must be positive")
        self.tonic_frequency = tonic_frequency
        logger.info(f"Tonic frequency set to {tonic_frequency}Hz")
    
    def auto_detect_tonic(self, audio_chunks, method='median'):
        """
        Automatically detect tonic frequency from audio chunks.
        
        Args:
            audio_chunks: List of audio chunks (numpy arrays)
            method: Detection method ('median', 'mode', 'histogram')
        
        Returns:
            Detected tonic frequency in Hz
        """
        logger.info(f"Auto-detecting tonic using {method} method...")
        
        pitches = []
        for chunk in audio_chunks:
            pitch = self.extract_pitch(chunk)
            if pitch is not None:
                pitches.append(pitch)
        
        if len(pitches) == 0:
            logger.warning("No valid pitches found for tonic detection")
            return None
        
        pitches = np.array(pitches)
        
        if method == 'median':
            tonic = np.median(pitches)
        elif method == 'mode':
            # Use histogram to find most frequent pitch
            hist, bins = np.histogram(pitches, bins=100)
            tonic = bins[np.argmax(hist)]
        elif method == 'histogram':
            # More sophisticated histogram-based detection
            hist, bins = np.histogram(pitches, bins=100, density=True)
            # Find peaks in histogram
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
            if len(peaks) > 0:
                # Use the lowest significant peak as tonic (typically Sa)
                tonic = bins[peaks[0]]
            else:
                tonic = np.median(pitches)
        else:
            raise ValueError(f"Unknown tonic detection method: {method}")
        
        # Round to reasonable precision
        tonic = round(tonic, 2)
        
        # Validate tonic is within reasonable range
        if tonic < self.min_freq or tonic > self.max_freq * 2:
            logger.warning(f"Detected tonic {tonic}Hz is outside reasonable range, using median")
            tonic = np.median(pitches)
        
        self.tonic_frequency = tonic
        logger.info(f"Auto-detected tonic frequency: {tonic}Hz")
        return tonic
    
    def extract_pitch(self, audio_chunk, method='autocorrelation'):
        """
        Extract fundamental frequency from audio chunk.
        
        Args:
            audio_chunk: Audio samples as numpy array
            method: Pitch detection method ('autocorrelation', 'librosa', 'yin', 'pyin')

        Returns:
            Fundamental frequency in Hz, or None if detection fails
        """
        if len(audio_chunk) < 2:
            return None

        try:
            if method == 'autocorrelation':
                return self._pitch_autocorrelation(audio_chunk)
            elif method == 'librosa':
                return self._pitch_librosa(audio_chunk)
            elif method == 'yin':
                return self._pitch_yin(audio_chunk)
            elif method == 'pyin':
                return self._pitch_pyin(audio_chunk)
            else:
                raise ValueError(f"Unknown pitch detection method: {method}")
        
        except Exception as e:
            logger.debug(f"Pitch extraction failed: {e}")
            return None
    
    MIN_AUTOCORR_CONFIDENCE = 0.3  # peak value must be at least this fraction of the
    # zero-lag value to count as a real periodicity, not noise/unvoiced content
    SHORTEST_LAG_TOLERANCE = 0.85  # see _pitch_autocorrelation() - how close to the
    # tallest peak's height a shorter-lag peak must be to win over it

    def _pitch_autocorrelation(self, audio_chunk):
        """
        Fast autocorrelation-based pitch detection.

        Among local peaks within the valid [min_freq, max_freq] lag range,
        picks the SHORTEST-LAG peak that's still within SHORTEST_LAG_TOLERANCE
        of the tallest peak's height - not simply the first peak encountered,
        and not simply the tallest peak overall. Both simpler rules turned out
        to be real bugs, each confirmed on real recordings:

        - First-peak (the original logic): a strong harmonic or a burst of
          noise/breath often produces a taller, earlier peak at a short lag (=
          high, implausible frequency), while the true fundamental's peak sits
          further out, never examined. Confirmed on a real Bhairav recording -
          >60% of frames read as an implausible >600Hz, and for about half of
          those, a taller peak existed at a lower, more plausible frequency
          that first-peak logic skipped straight past.
        - Tallest-peak (this file's first fix): overcorrects the other way - on
          a different real recording (Bhimsen Joshi's Yaman, this project's
          most-validated reference file, known-correct tonic 110.79 Hz), the
          tallest peak in range sat at a LONGER lag than the true fundamental's
          own peak (plausibly a resonance or sub-harmonic that happened to be
          taller), flipping a confident, correct Yaman classification (100%
          vs. 43.7% runner-up) into a confident, wrong Asavari one. Isolated by
          reverting to first-peak logic and confirming it restored the
          original correct result.

        This middle ground keeps the Bhairav fix (a short-lag peak that's only
        weakly supported - the wrong peak there was ~70% of the correct one's
        height - still loses to a substantially taller peak further out) while
        no longer discarding a short-lag peak that's nearly as tall as the
        global max just because something slightly taller exists elsewhere
        (which is what broke the Bhimsen Joshi file). Not yet re-validated
        against every case this session touched - see PROJECT_PLAN.md for
        follow-up.

        Peak location intentionally uses the same index convention as the
        original first-peak logic (the index where the derivative's sign flips,
        not index+1, which is technically where the local max actually sits) -
        an initial version "corrected" this and it measurably hurt real
        detection: it shifts every single-chunk detection by exactly one lag
        sample (a few Hz, imperceptible on its own), but that shift was enough
        to push eval_harness.py's synthetic accuracy from 95.2% to 74.6%,
        concentrated in Marwa/Puriya - a same-thaat pair already distinguished
        only by fine-grained note-weighting, not note presence (see config.py),
        evidently right on the edge of that distinction.

        Args:
            audio_chunk: Audio samples as numpy array

        Returns:
            Fundamental frequency in Hz, or None if detection fails
        """
        # Apply windowing to reduce edge effects
        window = np.hanning(len(audio_chunk))
        audio_windowed = audio_chunk * window

        # Compute autocorrelation
        autocorr = np.correlate(audio_windowed, audio_windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        if len(autocorr) < 2 or autocorr[0] <= 0:
            return None

        # Find where derivative changes from positive to negative (local peaks)
        d = np.diff(autocorr)
        peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
        peaks = peaks[peaks > 0]  # exclude lag 0

        if len(peaks) == 0:
            return None

        # Restrict to peaks whose lag falls within the valid frequency range
        # BEFORE picking among them, so a strong out-of-range peak can't win
        lag_min = self.sample_rate / self.max_freq
        lag_max = self.sample_rate / self.min_freq
        valid_peaks = peaks[(peaks >= lag_min) & (peaks <= lag_max)]

        if len(valid_peaks) == 0:
            return None

        # valid_peaks is already in increasing-lag order (peaks was built by a
        # single forward scan) - walk it and take the first one tall enough
        # relative to the tallest, rather than unconditionally the tallest.
        # Threshold is shifted DOWN from tallest_height by a fraction of its
        # magnitude (not simply tallest_height * tolerance) so this stays
        # correct when tallest_height is negative (autocorrelation can dip
        # below zero) - a plain multiplicative threshold would then sit ABOVE
        # tallest_height, making the comparison below impossible to satisfy
        # even for the tallest peak itself.
        tallest_height = np.max(autocorr[valid_peaks])
        threshold = tallest_height - abs(tallest_height) * (1 - self.SHORTEST_LAG_TOLERANCE)
        best_peak = next(p for p in valid_peaks if autocorr[p] >= threshold)

        # Reject weak/ambiguous periodicity (noise, unvoiced consonants, breath)
        # rather than confidently returning a guess with no real basis
        if autocorr[best_peak] / autocorr[0] < self.MIN_AUTOCORR_CONFIDENCE:
            return None

        frequency = self.sample_rate / best_peak

        if not np.isfinite(frequency):
            return None
        if frequency < self.min_freq or frequency > self.max_freq:
            return None

        return frequency
    
    def _pitch_librosa(self, audio_chunk):
        """
        Librosa-based pitch detection (pyin).
        
        Args:
            audio_chunk: Audio samples as numpy array
        
        Returns:
            Fundamental frequency in Hz, or None if detection fails
        """
        try:
            # Use librosa's piptrack for pitch tracking
            pitches, magnitudes = librosa.piptrack(
                y=audio_chunk,
                sr=self.sample_rate,
                threshold=0.1,
                fmin=self.min_freq,
                fmax=self.max_freq
            )
            
            # Get the pitch with maximum magnitude
            pitch_track = pitches[np.argmax(magnitudes, axis=0)]
            pitch = pitch_track[np.argmax(magnitudes)]
            
            if pitch > 0 and np.isfinite(pitch):
                return pitch
            return None
        
        except Exception as e:
            logger.debug(f"Librosa pitch detection failed: {e}")
            return None
    
    def _pitch_yin(self, audio_chunk):
        """
        YIN algorithm for pitch detection (simplified implementation).
        
        Args:
            audio_chunk: Audio samples as numpy array
        
        Returns:
            Fundamental frequency in Hz, or None if detection fails
        """
        try:
            # Use librosa's yin implementation
            pitches = librosa.yin(
                y=audio_chunk,
                fmin=self.min_freq,
                fmax=self.max_freq,
                sr=self.sample_rate
            )
            
            # Return the median pitch (more robust)
            pitch = np.median(pitches)
            
            if pitch > 0 and np.isfinite(pitch):
                return pitch
            return None

        except Exception as e:
            logger.debug(f"YIN pitch detection failed: {e}")
            return None

    def _pitch_pyin(self, audio_chunk, voiced_prob_threshold=0.5):
        """
        Probabilistic YIN pitch detection. Unlike plain YIN (_pitch_yin above),
        librosa.pyin returns a per-frame voiced/unvoiced flag and confidence
        alongside each pitch estimate - frames it isn't confident are actually
        pitched (silence, percussion, noise) can be filtered out instead of forced
        into a pitch estimate regardless.

        Args:
            audio_chunk: Audio samples as numpy array
            voiced_prob_threshold: Minimum voiced-frame confidence to accept (0-1)

        Returns:
            Fundamental frequency in Hz, or None if no confident voiced pitch found
        """
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=audio_chunk,
                fmin=self.min_freq,
                fmax=self.max_freq,
                sr=self.sample_rate
            )

            valid = voiced_flag & (voiced_probs >= voiced_prob_threshold) & np.isfinite(f0)
            if not np.any(valid):
                return None

            pitch = np.median(f0[valid])

            if pitch > 0 and np.isfinite(pitch):
                return pitch
            return None

        except Exception as e:
            logger.debug(f"pYIN pitch detection failed: {e}")
            return None

    def frequency_to_cents(self, frequency, tonic_frequency=None):
        """
        Convert frequency to cents relative to tonic.
        
        Formula: cents = 1200 * log2(f_input / f_tonic)
        
        Args:
            frequency: Input frequency in Hz
            tonic_frequency: Tonic frequency in Hz (uses self.tonic_frequency if None)
        
        Returns:
            Cents value (can be negative), or None if conversion fails
        """
        if tonic_frequency is None:
            tonic_frequency = self.tonic_frequency
        
        # Check for edge cases
        if frequency is None or frequency <= 0:
            return None
        
        # Check for NaN or infinite values
        if not np.isfinite(frequency):
            return None
        
        if tonic_frequency is None or tonic_frequency <= 0:
            logger.warning("Tonic frequency not set, cannot convert to cents")
            return None
        
        # Check for NaN or infinite tonic
        if not np.isfinite(tonic_frequency):
            return None
        
        try:
            # Apply the cents formula
            cents = 1200.0 * np.log2(frequency / tonic_frequency)
            
            # Check if result is NaN or infinite
            if not np.isfinite(cents):
                return None
            
            return cents
        
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Frequency to cents conversion failed: {e}")
            return None
    
    def wrap_to_octave(self, cents):
        """
        Wrap cents to 0-1200 range (single octave).
        
        Args:
            cents: Cents value (can be negative or > 1200)
        
        Returns:
            Cents wrapped to 0-1200 range, or None if input is None
        """
        if cents is None:
            return None
        
        # Use modulo to wrap to 0-1200 range
        wrapped_cents = cents % 1200.0
        return wrapped_cents
    
    def cents_to_bin(self, cents):
        """
        Convert cents to histogram bin index.
        
        Args:
            cents: Cents value in 0-1200 range
        
        Returns:
            Bin index (0-119), or None if conversion fails
        """
        if cents is None:
            return None
        
        if cents < 0 or cents >= 1200:
            logger.warning(f"Cents {cents} outside valid range [0, 1200)")
            return None
        
        bin_index = int(cents / self.cents_per_bin)
        bin_index = min(bin_index, self.histogram_bins - 1)  # Clamp to valid range
        return bin_index
    
    def process_audio_chunk(self, audio_chunk):
        """
        Process a single audio chunk through the full pipeline.
        
        Pipeline:
        1. Extract pitch (fundamental frequency)
        2. Convert to cents relative to tonic
        3. Wrap to octave (0-1200 cents)
        4. Convert to bin index
        
        Args:
            audio_chunk: Audio samples as numpy array
        
        Returns:
            Bin index (0-119), or None if processing fails
        """
        self.total_frames_processed += 1
        
        # Step 1: Pitch extraction
        pitch = self.extract_pitch(audio_chunk)
        if pitch is None:
            self.failed_detections += 1
            return None
        
        # Step 2: Tonic normalization (convert to cents)
        if self.tonic_frequency is None:
            logger.warning("Tonic frequency not set, cannot process audio chunk")
            self.failed_detections += 1
            return None
        
        cents = self.frequency_to_cents(pitch)
        if cents is None:
            self.failed_detections += 1
            return None
        
        # Step 3: Wrap to octave
        wrapped_cents = self.wrap_to_octave(cents)
        if wrapped_cents is None:
            self.failed_detections += 1
            return None
        
        # Step 4: Convert to bin
        bin_index = self.cents_to_bin(wrapped_cents)
        if bin_index is None:
            self.failed_detections += 1
            return None
        
        self.successful_detections += 1
        return bin_index
    
    def build_histogram(self, audio_chunks, normalize=True):
        """
        Build a 120-bin pitch class histogram from multiple audio chunks.
        
        Args:
            audio_chunks: List of audio chunks (numpy arrays)
            normalize: Whether to normalize the histogram
        
        Returns:
            120-bin histogram as numpy array, or None if invalid
        """
        histogram = np.zeros(self.histogram_bins)
        
        for chunk in audio_chunks:
            bin_index = self.process_audio_chunk(chunk)
            if bin_index is not None:
                histogram[bin_index] += 1
        
        # Normalize histogram
        if normalize and np.sum(histogram) > 0:
            histogram = histogram / np.sum(histogram)
            
            # Check for NaN or infinite values after normalization
            if not np.all(np.isfinite(histogram)):
                logger.debug("Histogram contains NaN or infinite values after normalization")
                return None
        
        return histogram
    
    def add_to_histogram(self, pitch):
        """
        Add a pitch value to the internal histogram.
        
        Args:
            pitch: Fundamental frequency in Hz
        """
        if pitch is None or not np.isfinite(pitch):
            return
        
        # Convert to cents relative to tonic
        cents = self.frequency_to_cents(pitch)
        if cents is None:
            return
        
        # Wrap to octave
        wrapped_cents = self.wrap_to_octave(cents)
        if wrapped_cents is None:
            return
        
        # Convert to bin index
        bin_index = self.cents_to_bin(wrapped_cents)
        if bin_index is None:
            return
        
        # Increment the histogram bin
        self.histogram[bin_index] += 1
    
    def get_normalized_histogram(self):
        """
        Get the normalized histogram.
        
        Returns:
            Normalized 120-bin histogram as numpy array, or None if invalid
        """
        if np.sum(self.histogram) == 0:
            return None
        
        normalized = self.histogram / np.sum(self.histogram)
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(normalized)):
            return None
        
        return normalized
    
    def get_statistics(self):
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = 0.0
        if self.total_frames_processed > 0:
            success_rate = self.successful_detections / self.total_frames_processed
        
        return {
            'total_frames_processed': self.total_frames_processed,
            'successful_detections': self.successful_detections,
            'failed_detections': self.failed_detections,
            'success_rate': success_rate,
            'tonic_frequency': self.tonic_frequency
        }


if __name__ == '__main__':
    # Test the feature extractor
    import sys
    
    print("Feature Extraction Test")
    print("======================")
    
    # Generate test audio (sine wave at 440Hz), one BUFFER_SIZE chunk (~23ms) -
    # matching how extract_pitch is actually called in the real pipeline
    # (AudioStream and load_file_chunks() both deliver BUFFER_SIZE-sized chunks).
    # A full 1-second buffer was never representative of real usage and, once
    # _pitch_autocorrelation() started picking the tallest in-range peak instead
    # of just the first one, produced a wrong result specific to that unrealistic
    # buffer length (many genuine same-pitch harmonics fit in 1s, and one of the
    # later ones can outscore the fundamental's).
    test_frequency = 440.0  # Hz (A4)
    t = np.linspace(0, BUFFER_SIZE / SAMPLE_RATE, BUFFER_SIZE, endpoint=False)
    test_audio = 0.5 * np.sin(2 * np.pi * test_frequency * t)
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Set tonic to 220Hz (A3, one octave below test frequency)
    extractor.set_tonic(220.0)
    
    # Test pitch extraction
    print(f"\nTest audio: {test_frequency}Hz sine wave")
    print(f"Tonic frequency: {extractor.tonic_frequency}Hz")
    
    pitch = extractor.extract_pitch(test_audio)
    print(f"Detected pitch: {pitch}Hz")
    
    # Test cents conversion
    cents = extractor.frequency_to_cents(pitch)
    print(f"Cents relative to tonic: {cents:.2f}")
    
    # Test octave wrapping
    wrapped_cents = extractor.wrap_to_octave(cents)
    print(f"Wrapped to octave: {wrapped_cents:.2f}")
    
    # Test bin conversion
    bin_index = extractor.cents_to_bin(wrapped_cents)
    print(f"Bin index: {bin_index}")
    
    # Test full pipeline across several realistic-sized chunks (each chunk is
    # still BUFFER_SIZE samples - just more of them, to exercise build_histogram())
    print("\nTesting full pipeline on multiple chunks...")
    t_multi = np.linspace(0, 1.0, int(SAMPLE_RATE * 1.0), endpoint=False)
    multi_chunk_audio = 0.5 * np.sin(2 * np.pi * test_frequency * t_multi)
    chunks = [multi_chunk_audio[i:i+BUFFER_SIZE] for i in range(0, len(multi_chunk_audio), BUFFER_SIZE)]
    histogram = extractor.build_histogram(chunks)
    
    print(f"Histogram shape: {histogram.shape}")
    print(f"Histogram sum: {np.sum(histogram):.4f}")
    print(f"Peak bin: {np.argmax(histogram)}")
    
    # Print statistics
    stats = extractor.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\nTest completed successfully!")
