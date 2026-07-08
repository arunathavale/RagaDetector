"""
Configuration file for RagaDetector - CPU-Optimized Live Hindustani Classical Raaga Identifier
"""

import numpy as np


# =============================================================================
# AUDIO SETTINGS
# =============================================================================
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1
AUDIO_DEVICE = None  # None for default device


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================
FFT_SIZE = 2048
HOP_SIZE = 512
WINDOW_TYPE = 'hann'
HISTOGRAM_BINS = 120  # 120-bin histogram for pitch distribution


# =============================================================================
# CLASSIFICATION SETTINGS
# =============================================================================
CONFIDENCE_THRESHOLD = 0.7
MIN_DETECTION_DURATION = 2.0  # seconds
HISTORY_WINDOW = 10.0  # seconds
SMOOTHING_WINDOW = 5  # number of frames to smooth
ROLLING_WINDOW_SECONDS = 45.0  # seconds - increased historical memory for stability
DECAY_FACTOR = 0.98  # increased from 0.95 for longer context retention
TIE_BREAKER_WINDOW_SECONDS = 90.0  # seconds - long-term window for tie-breaking
TIE_BREAKER_THRESHOLD = 0.10  # 10% - if top two scores are within this, apply tie-breaker
TIE_BREAKER_ENERGY_THRESHOLD = 0.02  # 2% - if forbidden notes energy below this over 90s, break tie
TIE_BREAKER_BONUS = 0.20  # 20% - bonus to award to pentatonic raga when tie broken


# =============================================================================
# SWARA DEFINITIONS (12 semitones in Hindustani classical music)
# =============================================================================
# Mapping of swaras to semitone indices (0-11)
SWARA_MAPPING = {
    'Sa': 0,
    'Re_komal': 1,
    'Re_shuddha': 2,
    'Ga_komal': 3,
    'Ga_shuddha': 4,
    'Ma_shuddha': 5,
    'Ma_tivra': 6,
    'Pa': 7,
    'Dha_komal': 8,
    'Dha_shuddha': 9,
    'Ni_komal': 10,
    'Ni_shuddha': 11
}


# =============================================================================
# 120-BIN HISTOGRAM MATH
# =============================================================================
# 120 bins across the octave, 10 bins per semitone for finer resolution
# This allows detection of microtonal variations (shrutis)
BINS_PER_SEMITONE = HISTOGRAM_BINS // 12  # 10 bins per semitone


def semitone_to_bin(semitone_index):
    """Convert semitone index (0-11) to histogram bin range."""
    start_bin = semitone_index * BINS_PER_SEMITONE
    end_bin = start_bin + BINS_PER_SEMITONE
    return start_bin, end_bin


def compute_swara_weights(aroha, avroha, pakad, vadi=None, samvadi=None,
                           vadi_boost=1.5, samvadi_boost=1.25):
    """
    Weight each swara by how often it occurs across aroha/avroha/pakad, with
    extra emphasis on the vadi (king) and samvadi (queen) notes. Ragas that
    share the same note set (e.g. Jaunpuri/Asavari, Marwa/Puriya) still get
    distinct histograms this way, since they differ in melodic emphasis even
    when the underlying note set is identical.

    Returns:
        Dict mapping swara name -> relative weight (not yet normalized)
    """
    weights = {}
    for swara in aroha + avroha + pakad:
        weights[swara] = weights.get(swara, 0) + 1
    if vadi in weights:
        weights[vadi] *= vadi_boost
    if samvadi in weights:
        weights[samvadi] *= samvadi_boost
    return weights


def create_weighted_swara_histogram(swara_weights):
    """
    Create a 120-bin histogram from a dict of swara -> weight.

    Args:
        swara_weights: Dict mapping swara name to relative weight
                       (e.g., {'Sa': 5, 'Re_shuddha': 3})

    Returns:
        numpy array of shape (120,) with normalized histogram values
    """
    histogram = np.zeros(HISTOGRAM_BINS)

    for swara, weight in swara_weights.items():
        if swara in SWARA_MAPPING:
            semitone_idx = SWARA_MAPPING[swara]
            start_bin, end_bin = semitone_to_bin(semitone_idx)
            histogram[start_bin:end_bin] = weight

    # Normalize histogram
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)

    return histogram


def create_forbidden_notes_mask(forbidden_notes_list):
    """
    Create a 120-bin mask for forbidden notes (1 where forbidden, 0 where allowed).
    
    Args:
        forbidden_notes_list: List of forbidden swara names (e.g., ['Ma_tivra', 'Ni_shuddha'])
    
    Returns:
        numpy array of shape (120,) with 1.0 for forbidden bins, 0.0 for allowed
    """
    mask = np.zeros(HISTOGRAM_BINS)
    
    for swara in forbidden_notes_list:
        if swara in SWARA_MAPPING:
            semitone_idx = SWARA_MAPPING[swara]
            start_bin, end_bin = semitone_to_bin(semitone_idx)
            # Mark these bins as forbidden
            mask[start_bin:end_bin] = 1.0
    
    return mask


def create_required_notes_mask(required_notes_list):
    """
    Create a 120-bin mask for required notes (1 where required, 0 where not required).
    
    Args:
        required_notes_list: List of required swara names (e.g., ['Sa', 'Re_shuddha', 'Ga_shuddha'])
    
    Returns:
        numpy array of shape (120,) with 1.0 for required bins, 0.0 for not required
    """
    mask = np.zeros(HISTOGRAM_BINS)
    
    for swara in required_notes_list:
        if swara in SWARA_MAPPING:
            semitone_idx = SWARA_MAPPING[swara]
            start_bin, end_bin = semitone_to_bin(semitone_idx)
            # Mark these bins as required
            mask[start_bin:end_bin] = 1.0
    
    return mask


# =============================================================================
# RAAGA DEFINITIONS WITH 120-BIN HISTOGRAMS
# =============================================================================

RAAGA_DATABASE = {
    'Yaman': {
        'that': 'Kalyan',
        'aroha': ['Sa', 'Re_shuddha', 'Ga_shuddha', 'Ma_tivra', 'Pa', 'Dha_shuddha', 'Ni_shuddha', 'Sa'],
        'avroha': ['Sa', 'Ni_shuddha', 'Dha_shuddha', 'Pa', 'Ma_tivra', 'Ga_shuddha', 'Re_shuddha', 'Sa'],
        'vadi': 'Ga_shuddha',
        'samvadi': 'Ni_shuddha',
        'pakad': ['Ni_shuddha', 'Re_shuddha', 'Ga_shuddha', 'Re_shuddha', 'Sa', 'Pa', 'Ma_tivra', 
                  'Ga_shuddha', 'Re_shuddha', 'Sa'],
        'prahar': 'evening (first quarter)',
        'forbidden_notes': [],  # Yaman uses all 7 notes, no forbidden notes
        'required_notes': ['Sa', 'Re_shuddha', 'Ga_shuddha', 'Ma_tivra', 'Pa', 'Dha_shuddha', 'Ni_shuddha'],  # All 7 notes expected
        'characteristic_phrases': [
            ['Ni_shuddha', 'Re_shuddha', 'Ga_shuddha'],
            ['Ma_tivra', 'Pa', 'Dha_shuddha', 'Ni_shuddha', 'Sa'],
            ['Re_shuddha', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha', 'Sa']
        ]
    },
    
    'Bhupali': {
        'that': 'Bilaval',
        'aroha': ['Sa', 'Re_shuddha', 'Ga_shuddha', 'Pa', 'Dha_shuddha', 'Sa'],
        'avroha': ['Sa', 'Dha_shuddha', 'Pa', 'Ga_shuddha', 'Re_shuddha', 'Sa'],
        'vadi': 'Ga_shuddha',
        'samvadi': 'Dha_shuddha',
        'pakad': ['Ga_shuddha', 'Pa', 'Dha_shuddha', 'Sa', 'Dha_shuddha', 'Pa', 'Ga_shuddha',
                  'Re_shuddha', 'Sa'],
        'prahar': 'evening (first quarter)',
        'forbidden_notes': ['Ma_tivra', 'Ni_shuddha'],  # Bhupali is pentatonic, forbids Ma (tivra) and Ni
        'required_notes': ['Sa', 'Re_shuddha', 'Ga_shuddha', 'Pa', 'Dha_shuddha'],  # Only 5 notes expected
        'characteristic_phrases': [
            ['Ga_shuddha', 'Pa', 'Dha_shuddha', 'Sa'],
            ['Dha_shuddha', 'Pa', 'Ga_shuddha', 'Re_shuddha', 'Sa'],
            ['Sa', 'Re_shuddha', 'Ga_shuddha', 'Pa', 'Dha_shuddha', 'Sa']
        ]
    },

    'Bhairav': {
        'that': 'Bhairav',
        'aroha': ['Sa', 'Re_komal', 'Ga_shuddha', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Ni_shuddha', 'Sa'],
        'avroha': ['Sa', 'Ni_shuddha', 'Dha_komal', 'Pa', 'Ma_shuddha', 'Ga_shuddha', 'Re_komal', 'Sa'],
        'vadi': 'Re_komal',
        'samvadi': 'Dha_komal',
        'pakad': ['Sa', 'Re_komal', 'Ga_shuddha', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Pa',
                  'Ma_shuddha', 'Ga_shuddha', 'Re_komal', 'Sa'],
        'prahar': 'morning (dawn)',
        'forbidden_notes': ['Re_shuddha', 'Ga_komal', 'Ma_tivra', 'Dha_shuddha', 'Ni_komal'],
        'required_notes': ['Sa', 'Re_komal', 'Ga_shuddha', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Ni_shuddha'],
        'characteristic_phrases': [
            ['Re_komal', 'Ga_shuddha', 'Ma_shuddha'],
            ['Ma_shuddha', 'Pa', 'Dha_komal'],
            ['Dha_komal', 'Pa', 'Ma_shuddha', 'Ga_shuddha', 'Re_komal', 'Sa']
        ]
    },

    'Asavari': {
        'that': 'Asavari',
        'aroha': ['Sa', 'Re_shuddha', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Sa'],
        'avroha': ['Sa', 'Ni_komal', 'Dha_komal', 'Pa', 'Ma_shuddha', 'Ga_komal', 'Re_shuddha', 'Sa'],
        'vadi': 'Dha_komal',
        'samvadi': 'Ga_komal',
        'pakad': ['Ma_shuddha', 'Pa', 'Dha_komal', 'Ni_komal', 'Dha_komal', 'Pa',
                  'Ma_shuddha', 'Ga_komal', 'Re_shuddha', 'Sa'],
        'prahar': 'afternoon (second quarter)',
        'forbidden_notes': ['Re_komal', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha'],
        'required_notes': ['Sa', 'Re_shuddha', 'Ga_komal', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Ni_komal'],
        'characteristic_phrases': [
            ['Ma_shuddha', 'Pa', 'Dha_komal'],
            ['Dha_komal', 'Ni_komal', 'Dha_komal', 'Pa'],
            ['Pa', 'Ma_shuddha', 'Ga_komal', 'Re_shuddha', 'Sa']
        ]
    },

    'Jaunpuri': {
        'that': 'Asavari',
        'aroha': ['Sa', 'Re_shuddha', 'Ga_komal', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Sa'],
        'avroha': ['Sa', 'Ni_komal', 'Dha_komal', 'Pa', 'Ma_shuddha', 'Ga_komal', 'Re_shuddha', 'Sa'],
        'vadi': 'Dha_komal',
        'samvadi': 'Ga_komal',
        'pakad': ['Dha_komal', 'Ni_komal', 'Sa', 'Re_shuddha', 'Ga_komal', 'Re_shuddha', 'Sa'],
        'prahar': 'afternoon (second quarter)',
        'forbidden_notes': ['Re_komal', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha'],
        'required_notes': ['Sa', 'Re_shuddha', 'Ga_komal', 'Ma_shuddha', 'Pa', 'Dha_komal', 'Ni_komal'],
        'characteristic_phrases': [
            ['Dha_komal', 'Ni_komal', 'Sa'],
            ['Sa', 'Re_shuddha', 'Ga_komal', 'Re_shuddha', 'Sa'],
            ['Ga_komal', 'Ma_shuddha', 'Pa', 'Dha_komal']
        ]
    },

    'Marwa': {
        'that': 'Marwa',
        'aroha': ['Sa', 'Re_komal', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha', 'Sa'],
        'avroha': ['Sa', 'Ni_shuddha', 'Dha_shuddha', 'Ma_tivra', 'Ga_shuddha', 'Re_komal', 'Sa'],
        'vadi': 'Re_komal',
        'samvadi': 'Dha_shuddha',
        'pakad': ['Dha_shuddha', 'Re_komal', 'Ga_shuddha', 'Sa'],
        'prahar': 'evening (sunset)',
        'forbidden_notes': ['Re_shuddha', 'Ga_komal', 'Ma_shuddha', 'Pa', 'Ni_komal'],  # Pa is deliberately absent
        'required_notes': ['Sa', 'Re_komal', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha'],
        'characteristic_phrases': [
            ['Re_komal', 'Ga_shuddha', 'Ma_tivra'],
            ['Ma_tivra', 'Dha_shuddha', 'Ni_shuddha', 'Sa'],
            ['Dha_shuddha', 'Ma_tivra', 'Ga_shuddha', 'Re_komal']
        ]
    },

    'Puriya': {
        'that': 'Marwa',
        # Puriya's aroha traditionally opens on the lower-octave Ni; this schema does not
        # track octave, so it is represented here using the same Ni_shuddha swara identity.
        'aroha': ['Ni_shuddha', 'Re_komal', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha', 'Sa'],
        'avroha': ['Sa', 'Ni_shuddha', 'Dha_shuddha', 'Ma_tivra', 'Ga_shuddha', 'Re_komal', 'Ni_shuddha', 'Sa'],
        'vadi': 'Ni_shuddha',
        'samvadi': 'Ga_shuddha',
        'pakad': ['Ni_shuddha', 'Re_komal', 'Ga_shuddha', 'Ma_tivra'],
        'prahar': 'evening (sunset)',
        'forbidden_notes': ['Re_shuddha', 'Ga_komal', 'Ma_shuddha', 'Pa', 'Ni_komal'],  # Pa is deliberately absent
        'required_notes': ['Sa', 'Re_komal', 'Ga_shuddha', 'Ma_tivra', 'Dha_shuddha', 'Ni_shuddha'],
        'characteristic_phrases': [
            ['Ni_shuddha', 'Re_komal', 'Ga_shuddha'],
            ['Ga_shuddha', 'Ma_tivra', 'Dha_shuddha'],
            ['Dha_shuddha', 'Ma_tivra', 'Ga_shuddha', 'Re_komal']
        ]
    }
}

# Populate each raga's histogram from its own aroha/avroha/pakad/vadi/samvadi,
# so ragas sharing a note set (e.g. Jaunpuri/Asavari, Marwa/Puriya) still get
# distinct reference histograms based on melodic emphasis.
for _raaga_data in RAAGA_DATABASE.values():
    _weights = compute_swara_weights(
        _raaga_data['aroha'], _raaga_data['avroha'], _raaga_data['pakad'],
        vadi=_raaga_data['vadi'], samvadi=_raaga_data['samvadi']
    )
    _raaga_data['histogram'] = create_weighted_swara_histogram(_weights)


# =============================================================================
# PITCH DETECTION SETTINGS
# =============================================================================
MIN_FREQUENCY = 80.0  # Hz (approximate lower bound for vocal music)
MAX_FREQUENCY = 1200.0  # Hz (approximate upper bound for vocal music)
HOP_LENGTH = 512
N_FFT = 2048


# =============================================================================
# FEATURE EXTRACTION SETTINGS
# =============================================================================
N_MFCC = 13
N_CHROMA = 12
N_MELS = 128
FMIN = MIN_FREQUENCY
FMAX = MAX_FREQUENCY


# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
MAX_CONCURRENT_ANALYSIS = 2  # Maximum number of parallel analysis threads
CPU_CORES = None  # None for auto-detection, or specify integer
MEMORY_LIMIT_MB = 4096  # Maximum memory usage in MB


# =============================================================================
# LOGGING SETTINGS
# =============================================================================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = 'raga_detector.log'
LOG_TO_CONSOLE = True


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
OUTPUT_FORMAT = 'json'  # json, text, or both
SHOW_CONFIDENCE = True
SHOW_DETECTED_SWARAS = True
SHOW_HISTOGRAM = False  # Set to True for debugging


# =============================================================================
# VALIDATION
# =============================================================================
def validate_config():
    """Validate configuration parameters."""
    assert HISTOGRAM_BINS == 120, "Histogram bins must be 120"
    assert BINS_PER_SEMITONE == 10, "Bins per semitone must be 10"
    assert SAMPLE_RATE in [44100, 48000], "Sample rate must be 44100 or 48000"
    assert BUFFER_SIZE in [512, 1024, 2048], "Buffer size must be 512, 1024, or 2048"
    assert len(RAAGA_DATABASE) > 0, "Raaga database must not be empty"
    
    for raaga_name, raaga_data in RAAGA_DATABASE.items():
        assert 'histogram' in raaga_data, f"Raaga {raaga_name} missing histogram"
        assert raaga_data['histogram'].shape == (120,), f"Raaga {raaga_name} histogram must be 120 bins"
        assert np.isclose(np.sum(raaga_data['histogram']), 1.0, atol=0.01), \
            f"Raaga {raaga_name} histogram must be normalized"


if __name__ == '__main__':
    validate_config()
    print("Configuration validated successfully!")
    print(f"Number of Raagas defined: {len(RAAGA_DATABASE)}")
    for raaga_name in RAAGA_DATABASE.keys():
        print(f"  - {raaga_name}")
