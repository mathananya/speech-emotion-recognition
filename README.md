# Speech Emotion Recognition

## Project Status
In progress (Feature pipeline complete)

## Overview

The feature pipeline performs two main tasks:
1. **Feature Extraction** - Extracts acoustic features from audio files
2. **Data Augmentation** - Applies augmentation techniques to increase dataset diversity


## Dependencies

The notebook requires the following Python packages:
- `librosa` - Audio processing library for feature extraction
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `pandas` - Data manipulation and CSV handling
- `kagglehub` - Downloading datasets from Kaggle
- `IPython` - Interactive audio playback

Install these packages using:
```bash
pip3 install librosa numpy kagglehub IPython matplotlib pandas
```

## Feature Extraction Methods

### 1. MFCCs (Mel-Frequency Cepstral Coefficients)

MFCCs capture the characteristics of human speech by converting audio to the mel scale, which better aligns with human hearing perception.

**Implementation:**
- Extracts 15 MFCCs from each audio file
- Returns a 2D array with shape `(time_steps, 15)`

**Function:** `extract_mfccs(data, sample_rate)`

### 2. Chroma Features

Chroma features represent the energy distribution across the 12 pitch classes (semitones) in music/speech.

**Implementation:**
- Extracts 12 chroma features per frame
- Returns a 2D array with shape `(time_steps, 12)`
- Based on the STFT (Short-Time Fourier Transform)

**Function:** `extract_chroma(data, sample_rate)`

### 3. Mel Spectrogram

Mel spectrograms provide a frequency representation of audio on the mel scale, useful for capturing spectral characteristics of emotional speech.

**Implementation:**
- Extracts 128 mel frequency bins
- Returns a 2D array with shape `(time_steps, 128)`
- Helps capture timbral characteristics of emotions

**Function:** `extract_mel(data, sample_rate)`

## Data Augmentation Techniques used


### 1. Noise Injection

Simulates real-world microphone noise to increase model robustness.

**Details:**
- Uses Gaussian noise injection (white noise)
- Signal-to-Noise Ratio (SNR) set to 25dB
- Noise power calculated as: `noise_power = signal_power × 0.0032`
- Prevents overpowering the original signal while introducing noise

**Function:** `noiseInjection(data)`

### 2. Temporal Shifting

Shifts the audio signal temporally to simulate timing variations in speech.

**Details:**
- Randomly shifts audio by ±10% of the signal length
- Uses circular shift (wraparound) to preserve signal continuity
- Simulates temporal variations in speech

**Function:** `shifting(data)`

### 3. Pitch Stretching

Applies random pitch shifts to simulate variations in speaker pitch.

**Details:**
- Shifts pitch by ±5 half-notes (semitones)
- Simulates natural pitch variations in emotional speech

**Function:** `pitchStretching(data, sr)`

## Data Sources

The pipeline uses two publicly available datasets:

### 1. RAVDESS Dataset
- **Source:** [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Contains:** 1,440 audio files from 24 actors
- **Emotions:** Neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Metadata Encoding:** Emotion code (01-08) and gender information in filenames

### 2. CREMA-D Dataset
- **Source:** [CREMA-D Audio](https://www.kaggle.com/datasets/ejlok1/cremad)
- **Contains:** 7,442 audio files from 91 actors
- **Emotions:** Neutral, happy, sad, angry, fearful, disgust, surprised
- **Metadata Encoding:** Actor ID and emotion code  in filenames

## Pipeline Output

The pipeline generates the following outputs:

1. **labels.csv** - CSV file containing:
   - `filepath` - Path to audio file
   - `label` - Combined gender-emotion label (e.g., "male-happy", "female-sad")
   - `mfccs` - Extracted MFCC features
   - `chroma` - Extracted chroma features
   - `mel` - Extracted mel spectrogram features

2. **labels.npy** - NumPy array version of the dataframe for efficient loading in models


## Label Format

Labels combine gender and emotion information:
- Format: `{gender}-{emotion}`


