import os
import csv
import numpy as np
import librosa
from scipy.signal import find_peaks
import os
from typing import Optional
import scipy.signal as sps
def extract_spectral_features(
    audio_path: str,
    sr: Optional[float] = 64000,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    n_chroma: int = 12,
    n_bands: int = 6,          # spectral contrast bands
    cup_num: Optional[str] = None,
    fill_label: Optional[str] = None
) -> dict:
    """
    Load an audio file and return a flat dictionary of spectral features
    suitable for writing as a single CSV row.
 
    Parameters
    ----------
    audio_path  : path to the audio file (.wav, .mp3, .flac, etc.)
    sr          : target sample rate; None = use file's native rate
    n_mfcc      : number of MFCC coefficients to compute
    n_fft       : FFT window size
    hop_length  : hop size between frames
    n_mels      : number of Mel filter banks
    n_chroma    : number of chroma bins
    n_bands     : number of spectral-contrast bands
    label       : optional string label appended as the last column
 
    Returns
    -------
    dict  – one key per feature; values are scalars (mean/std/min/max
            over time or individual coefficients).
    """
 
    # ── 1. Load audio ────────────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
 
    row: dict = {
        "duration_s": round(duration, 4),
        "Cup": cup_num,
        "Fill": fill_label
    }
 
    # ── 2. RMS energy ────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    _add_stats(row, "rms", rms)
 
    # ── 3. Zero-crossing rate ────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]
    _add_stats(row, "zcr", zcr)
 
    # ── 4. Spectral centroid ─────────────────────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    _add_stats(row, "spectral_centroid", centroid)
 
    # ── 5. Spectral bandwidth ─────────────────────────────────────────────────
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    _add_stats(row, "spectral_bandwidth", bandwidth)
 
    # ── 6. Spectral rolloff (85 % and 95 % thresholds) ───────────────────────
    for pct in (0.85, 0.95):
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=pct
        )[0]
        _add_stats(row, f"spectral_rolloff_{int(pct*100)}", rolloff)
 
    # ── 7. Spectral flatness ──────────────────────────────────────────────────
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    _add_stats(row, "spectral_flatness", flatness)
 
    # ── 8. Spectral contrast ──────────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands
    )                                              # shape: (n_bands+1, frames)
    for band_idx in range(contrast.shape[0]):
        _add_stats(row, f"spectral_contrast_band{band_idx}", contrast[band_idx])
 
    # ── 9. MFCCs ─────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )                                              # shape: (n_mfcc, frames)
    for coeff_idx in range(n_mfcc):
        _add_stats(row, f"mfcc_{coeff_idx}", mfcc[coeff_idx])
 
    # MFCC delta (velocity) and delta-delta (acceleration) ───────────────────
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    for coeff_idx in range(n_mfcc):
        _add_stats(row, f"mfcc_delta_{coeff_idx}",  delta_mfcc[coeff_idx])
        _add_stats(row, f"mfcc_delta2_{coeff_idx}", delta2_mfcc[coeff_idx])
 
    # ── 10. Chroma features ───────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
    )                                              # shape: (n_chroma, frames)
    for chroma_idx in range(n_chroma):
        _add_stats(row, f"chroma_{chroma_idx}", chroma[chroma_idx])
 
    # ── 11. Mel-spectrogram global statistics ─────────────────────────────────
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    row["mel_db_mean"]   = round(float(mel_db.mean()), 6)
    row["mel_db_std"]    = round(float(mel_db.std()),  6)
    row["mel_db_min"]    = round(float(mel_db.min()),  6)
    row["mel_db_max"]    = round(float(mel_db.max()),  6)

 
    return row
 
 
# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
 
def _add_stats(row: dict, prefix: str, values: np.ndarray) -> None:
    """Append mean, std, min, max of a 1-D time series to row dict."""
    row[f"{prefix}_mean"] = round(float(values.mean()), 6)
    row[f"{prefix}_std"]  = round(float(values.std()),  6)
    row[f"{prefix}_min"]  = round(float(values.min()),  6)
    row[f"{prefix}_max"]  = round(float(values.max()),  6)
 


def save_summary_csv(summaries, csv_path, append=False):
    """
    Write one or more summaries to CSV (one row per recording).

    Parameters:
    -----------
    summaries : dict or list of dict
        Output(s) of extract_audio_summary().
    csv_path : str
        Destination file path.
    append : bool
        If True and the file already exists, append without rewriting the header.
    """
    if isinstance(summaries, dict):
        summaries = [summaries]

    fieldnames = list(summaries[0].keys())
    for s in summaries[1:]:
        for k in s:
            if k not in fieldnames:
                fieldnames.append(k)

    file_has_content = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    mode         = 'a' if (append and file_has_content) else 'w'
    write_header = not (append and file_has_content)

    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"Saved {len(summaries)} recording(s) → '{csv_path}'")


# New function
def analyze_audio_windows(audio_path: str, cup_num: str, fill_label: str, n_windows: int = 10, sr: float = 48000, low_cut_hz = 270) -> dict:
    """
    Load audio and compute spectral features with temporal dynamics for each window.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    n_windows : int
        Number of windows to split the audio into (default: 10)
    sr : int
        Sample rate for loading audio (default: 22050 Hz)
    
    Returns:
    --------
    Dict with window-by-window spectral features and temporal dynamics
    """
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    
    results: dict = {
        "duration_s": round(duration, 4),
        "Cup": cup_num,
        "Fill": fill_label
    }
    # Calculate window size in samples
    window_samples = len(y) // n_windows

    # Precompute STFT for spectral features
    window_type = "hann"
    S = librosa.stft(y, window=window_type)
    S_abs = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr)

    # Harmonic/Percussive decomposition
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Onset detection (across entire signal for context)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    prev_spectral_centroid = None
    prev_rms_energy = None
    
    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples if i < n_windows - 1 else len(y)
        
        window_audio = y[start_idx:end_idx]
        window_harmonic = y_harmonic[start_idx:end_idx]
        window_percussive = y_percussive[start_idx:end_idx]
        
        # Time bounds
        start_time = start_idx / sr
        end_time = end_idx / sr
        
        # === Original Features ===
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window_audio, sr=sr, window=window_type))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=window_audio, sr=sr, window=window_type))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(window_audio))
        mfcc = np.mean(librosa.feature.mfcc(y=window_audio, sr=sr, n_mfcc=13), axis=1)
        rms_energy = np.mean(librosa.feature.rms(y=window_audio)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=window_audio, sr=sr, window=window_type))
        
        # === NEW: Temporal Dynamics (Deltas) ===
        spectral_centroid_delta = None
        rms_energy_delta = None
        
        if prev_spectral_centroid is not None:
            spectral_centroid_delta = spectral_centroid - prev_spectral_centroid
            rms_energy_delta = rms_energy - prev_rms_energy
        
        prev_spectral_centroid = spectral_centroid
        prev_rms_energy = rms_energy
        
        # === NEW: Onset Detection in This Window ===
        onsets_in_window = onset_times[(onset_times >= start_time) & (onset_times < end_time)]
        onset_count = len(onsets_in_window)
        
        # Attack time: time from first onset to peak energy
        attack_time = None
        if onset_count > 0:
            # Find peak RMS in window
            frame_energies = librosa.feature.rms(y=window_audio)[0]
            peak_frame = np.argmax(frame_energies)
            peak_time_in_window = peak_frame / (len(window_audio) / (end_time - start_time))
            attack_time = peak_time_in_window
        
        # === NEW: Harmonic/Percussive Analysis ===
        harmonic_ratio = np.mean(librosa.feature.rms(y=window_harmonic)[0]) / (rms_energy + 1e-6)
        percussive_ratio = np.mean(librosa.feature.rms(y=window_percussive)[0]) / (rms_energy + 1e-6)
        
        # === NEW: Dominant Frequency Tracking ===
        S_window = S_abs[:, start_idx//512:(end_idx//512)+1]  # approximate frame correspondence
        if S_window.shape[1] > 0:
            mag_spectrum = np.mean(S_window, axis=1)
            dominant_freq = freqs[np.argmax(mag_spectrum)]
            dominant_freq_magnitude = np.max(mag_spectrum)
        else:
            dominant_freq = 0
            dominant_freq_magnitude = 0
        
        # === NEW: Spectral Peak Distribution ===
        # Find how many spectral peaks exist (noisiness vs tonality)
        S_smooth = librosa.feature.melspectrogram(y=window_audio, sr=sr, n_mels=40)
        spectral_peaks = np.sum(librosa.util.peak_pick(np.mean(S_smooth, axis=1), 
                                                        pre_max=3, post_max=3, 
                                                        pre_avg=3, post_avg=3, delta=0.1, wait=5))
        
        # === NEW: Temporal Envelope (Attack, Sustain, Release) ===
        frame_rms = librosa.feature.rms(y=window_audio)[0]
        max_energy_frame = np.argmax(frame_rms)
        
        if max_energy_frame > 0 and max_energy_frame < len(frame_rms) - 1:
            # Sustain level relative to peak
            sustain_level = np.mean(frame_rms[max_energy_frame:]) / (np.max(frame_rms) + 1e-6)
        else:
            sustain_level = 0
        
        # === NEW: Spectral Flatness (tonality measure) ===
        # Flat spectrum = more percussive/noisy (water pouring)
        # Sharp peaks = more tonal (cup resonance)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=window_audio))

        features_dict = {
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'zero_crossing_rate': float(zero_crossing_rate),
            'rms_energy': float(rms_energy),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_centroid_delta': float(spectral_centroid_delta) if spectral_centroid_delta is not None else 0,
            'rms_energy_delta': float(rms_energy_delta) if rms_energy_delta is not None else 0,
            'onset_count': int(onset_count),
            'attack_time': float(attack_time) if attack_time is not None else 0,
            'harmonic_ratio': float(harmonic_ratio),
            'percussive_ratio': float(percussive_ratio),
            'dominant_frequency': float(dominant_freq),
            'dominant_frequency_magnitude': float(dominant_freq_magnitude),
            'spectral_peaks_count': int(spectral_peaks),
            'spectral_flatness': float(spectral_flatness),
            'sustain_level': float(sustain_level),
        }

        for key, value in features_dict.items():
            results[f'{key}_window_{i}'] = value

        for index,coeff in enumerate(mfcc.tolist()):
            results[f'MFCC_{index}_window_{i}'] = coeff
            
    return results


def analyze_audio_windows_filtered(audio_path: str, cup_num: str, fill_label: str, n_windows: int = 10,
                          sr: float = 48000, low_cut_hz: float = 270.0, hp_order: int = 6) -> dict:
    """
    Load audio and compute spectral features with temporal dynamics for each window,
    filtering out frequencies below low_cut_hz.
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Apply zero-phase Butterworth high-pass filter to remove frequencies below low_cut_hz
    sos = sps.butter(hp_order, low_cut_hz, btype="highpass", fs=sr, output="sos")
    y = sps.sosfiltfilt(sos, y)

    results: dict = {
        "duration_s": round(duration, 4),
        "Cup": cup_num,
        "Fill": fill_label
    }

    # Calculate window size in samples
    window_samples = len(y) // n_windows

    # Precompute STFT for spectral features (use explicit n_fft and hop_length for consistent mapping)
    n_fft = 2048
    hop_length = 512
    window_type = "hann"
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window_type)
    S_abs = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Zero out STFT bins below low_cut_hz (extra safety)
    low_bin_mask = freqs < low_cut_hz
    if np.any(low_bin_mask):
        S_abs[low_bin_mask, :] = 0.0

    # Harmonic/Percussive decomposition (operate on filtered waveform)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Onset detection (across entire filtered signal for context)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length, n_fft=n_fft)

    prev_spectral_centroid = None
    prev_rms_energy = None

    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples if i < n_windows - 1 else len(y)

        window_audio = y[start_idx:end_idx]
        window_harmonic = y_harmonic[start_idx:end_idx]
        window_percussive = y_percussive[start_idx:end_idx]

        # Time bounds
        start_time = start_idx / sr
        end_time = end_idx / sr

        # === Original Features (note: spectral features computed on filtered audio) ===
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window_audio, sr=sr, window=window_type))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=window_audio, sr=sr, window=window_type))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(window_audio))
        mfcc = np.mean(librosa.feature.mfcc(y=window_audio, sr=sr, n_mfcc=13), axis=1)
        rms_energy = np.mean(librosa.feature.rms(y=window_audio)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=window_audio, sr=sr, window=window_type))

        # === NEW: Temporal Dynamics (Deltas) ===
        spectral_centroid_delta = None
        rms_energy_delta = None

        if prev_spectral_centroid is not None:
            spectral_centroid_delta = spectral_centroid - prev_spectral_centroid
            rms_energy_delta = rms_energy - prev_rms_energy

        prev_spectral_centroid = spectral_centroid
        prev_rms_energy = rms_energy

        # === NEW: Onset Detection in This Window ===
        onsets_in_window = onset_times[(onset_times >= start_time) & (onset_times < end_time)]
        onset_count = len(onsets_in_window)

        # Attack time: time from first onset to peak energy
        attack_time = None
        if onset_count > 0:
            frame_energies = librosa.feature.rms(y=window_audio)[0]
            peak_frame = np.argmax(frame_energies)
            # Convert peak_frame to seconds within window: librosa.feature.rms uses frames with hop_length
            peak_time_in_window = (peak_frame * hop_length) / sr
            attack_time = peak_time_in_window

        # === NEW: Harmonic/Percussive Analysis ===
        harmonic_ratio = np.mean(librosa.feature.rms(y=window_harmonic)[0]) / (rms_energy + 1e-6)
        percussive_ratio = np.mean(librosa.feature.rms(y=window_percussive)[0]) / (rms_energy + 1e-6)

        # === NEW: Dominant Frequency Tracking ===
        # Map audio sample indices to STFT frame indices
        start_frame = int(np.floor(start_idx / hop_length))
        end_frame = int(np.ceil(end_idx / hop_length))
        start_frame = max(0, start_frame)
        end_frame = min(S_abs.shape[1] - 1, end_frame)
        if end_frame >= start_frame:
            S_window = S_abs[:, start_frame:end_frame + 1]
            # ensure low bins already zeroed; when selecting dominant freq only consider >= low_cut_hz
            valid_bins = np.where(freqs >= low_cut_hz)[0]
            if valid_bins.size > 0 and S_window.shape[1] > 0:
                mag_spectrum = np.mean(S_window[valid_bins, :], axis=1)
                dom_idx_rel = np.argmax(mag_spectrum)
                dom_idx = valid_bins[dom_idx_rel]
                dominant_freq = float(freqs[dom_idx])
                dominant_freq_magnitude = float(np.max(mag_spectrum))
            else:
                dominant_freq = 0.0
                dominant_freq_magnitude = 0.0
        else:
            dominant_freq = 0.0
            dominant_freq_magnitude = 0.0

        # === NEW: Spectral Peak Distribution ===
        S_smooth = librosa.feature.melspectrogram(y=window_audio, sr=sr, n_mels=40)
        spectral_peaks = int(np.sum(librosa.util.peak_pick(np.mean(S_smooth, axis=1),
                                                           pre_max=3, post_max=3,
                                                           pre_avg=3, post_avg=3, delta=0.1, wait=5)))

        # === NEW: Temporal Envelope (Attack, Sustain, Release) ===
        frame_rms = librosa.feature.rms(y=window_audio)[0]
        max_energy_frame = int(np.argmax(frame_rms)) if frame_rms.size > 0 else 0

        if max_energy_frame > 0 and max_energy_frame < len(frame_rms) - 1:
            sustain_level = float(np.mean(frame_rms[max_energy_frame:]) / (np.max(frame_rms) + 1e-6))
        else:
            sustain_level = 0.0

        # === NEW: Spectral Flatness (tonality measure) ===
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=window_audio)))

        features_dict = {
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'zero_crossing_rate': float(zero_crossing_rate),
            'rms_energy': float(rms_energy),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_centroid_delta': float(spectral_centroid_delta) if spectral_centroid_delta is not None else 0.0,
            'rms_energy_delta': float(rms_energy_delta) if rms_energy_delta is not None else 0.0,
            'onset_count': int(onset_count),
            'attack_time': float(attack_time) if attack_time is not None else 0.0,
            'harmonic_ratio': float(harmonic_ratio),
            'percussive_ratio': float(percussive_ratio),
            'dominant_frequency': float(dominant_freq),
            'dominant_frequency_magnitude': float(dominant_freq_magnitude),
            'spectral_peaks_count': int(spectral_peaks),
            'spectral_flatness': float(spectral_flatness),
            'sustain_level': float(sustain_level),
        }

        for key, value in features_dict.items():
            results[f'{key}_window_{i}'] = value

        for index, coeff in enumerate(mfcc.tolist()):
            results[f'MFCC_{index}_window_{i}'] = float(coeff)

    return results

# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    rootdir = 'WaterSound - Dataset'
    recordings = []
    for root, dirs, files in os.walk('WaterSound - Dataset'):
        for file in files:
            filepath = os.path.join(root, file)
            cup_num, fill_perc = tuple(root.split('\\')[-1].split(' - '))
            recordings.append((filepath, cup_num, fill_perc))
    
    dataset = [analyze_audio_windows_filtered(path, cup_num=cup, fill_label=fil, n_windows=5) for path, cup, fil in recordings]
    [extract_spectral_features(path, cup_num=cup, fill_label=fil) for path, cup, fil in recordings]

    save_summary_csv(dataset, "cup_dataset.csv") 

#    dic = analyze_audio_windows_filtered(r'WaterSound - Dataset\Cup 3\3 - 100\Recording number - 140.m4a.mp4', '1', '100', 5)
#    for i in range(5):
#        print(i, dic[f'dominant_frequency_window_{i}'])