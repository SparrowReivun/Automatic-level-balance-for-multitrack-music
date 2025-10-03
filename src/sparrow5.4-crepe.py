# -*- coding: utf-8 -*-
"""
Input  : AIFF, FLAC, MP3, OGG, and WAV files across all platforms (file/folder).

Output : The final mixdown of multi-track audio files and stereo through 3 different level balance processes.


Created by jinjie.shi@qmul.ac.uk on 2024/Feb/26
Version 5.2 (March 10-25, 2024): Enhanced robustness when dealing with complex audio. Introduced a safety gain feature to 
ensure that gain processing does not increase peak amplitude beyond -1dB. Added active audio detection to avoid issues 
with infinite gain when processing inputs containing silent audio segments.

Version 5.3 (March 26 to present):
After testing over ten multitrack projects with more than 200 individual tracks, the system has shown remarkable stability. 
We are now focusing on enhancing the artistic expression of the system. An experienced mix engineer has joined me in reviewing 
the system's performance. We are marking tracks with inappropriate level balances in red. Currently, we have noticed that 
vocals, bass, and electric guitar levels are slightly lower than desired. We have found that setting intervals based on 
spectral centroid and spectral bandwidth is more appropriate. I have already incorporated filtering and level balance 
processing based on spectral centroid and bandwidth, and we are currently testing with new material.
C
#sr is same, don't know when bit depth change from 24bit to 16
"""


import librosa
import glob
import os
import json
import pandas as pd
import numpy as np
import crepe
import torchaudio
import pesto
import torch
from pedalboard import Pedalboard, Gain, HighpassFilter, LowpassFilter
import soundfile as sf
import pyloudnorm as pyln
import shutil  # Used for copying files

def read_audio(audio_path):
    """Reads audio from a given path, maintaining its original sample rate."""
    audio, sr = librosa.load(audio_path, sr=None)
    return audio, sr

def CREPE(audio, sr, viterbi=True):
    """Predicts pitch using CREPE and returns the average of valid frequencies."""
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=viterbi)
    valid_frequencies = frequency[~np.isnan(frequency)]
    return np.mean(valid_frequencies) if len(valid_frequencies) > 0 else None

def PYIN(audio, sr):
    """Predicts pitch using PYIN and returns the average of valid frequencies."""
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), sr=sr)
    f0_filtered = f0[~np.isnan(f0)]
    return np.mean(f0_filtered) if len(f0_filtered) > 0 else None

def PESTO(audio_path):
    """Predicts pitch using PESTO and returns the average of valid frequencies."""
    x, sr = torchaudio.load(audio_path)
    x = x.mean(dim=0)
    timesteps, pitch, confidence, activations = pesto.predict(x, sr)
    mask = ~torch.isnan(pitch)
    non_nan_pitches = pitch[mask]
    return torch.mean(non_nan_pitches).item() if non_nan_pitches.numel() > 0 else None

def spectral_centroid(audio, sr):
    """Calculates and returns the mean spectral centroid of the audio."""
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return np.mean(centroid)

def spectral_bandwidth(audio, sr):
    """Calculates and returns the mean spectral bandwidth of the audio."""
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    return np.mean(bandwidth)

def bandpass_c(audio, sr):
    """Applies a bandpass filter based on the spectral centroid and bandwidth."""
    centroid = spectral_centroid(audio, sr)
    bandwidth = spectral_bandwidth(audio, sr)
    highpass_c = max(centroid - bandwidth/2, 1)  # 确保highpass_c不小于1
    lowpass_c = min(centroid + bandwidth/2, 22000)  # 确保lowpass_c不大于22000
    board = Pedalboard([HighpassFilter(cutoff_frequency_hz=highpass_c), LowpassFilter(cutoff_frequency_hz=lowpass_c)])
    filtered_audio = board(audio, sr)
    return filtered_audio, highpass_c, lowpass_c

def bandpass(audio, sr, pitch):
    """Applies a bandpass filter based on pitch, with fallback values if pitch is None."""
    highpass, lowpass = (2 * pitch, 4 * pitch) if pitch is not None else (50, 10000)
    board = Pedalboard([HighpassFilter(cutoff_frequency_hz=highpass), LowpassFilter(cutoff_frequency_hz=lowpass)])
    filtered_audio = board(audio, sr)
    return filtered_audio, highpass, lowpass

def calculate_p_lufs(filtered_audio, sr):
    """Calculates and returns the integrated loudness (LUFS) of the filtered audio."""
    meter = pyln.Meter(sr)
    p_lufs = meter.integrated_loudness(filtered_audio)
    return p_lufs

def match_loudness(audio, sr, p_lufs, lufs, c_lufs, audio_path, output_dir, output_lufs_dir, output_centroid_dir):
    """Adjusts the loudness of audio to match a target LUFS level."""
    max_sample_value = np.max(np.abs(audio))
    peak_dB = 20 * np.log10(max_sample_value)
    safe_value = -1 - peak_dB

    p_lufs_adjustment = min(-50 - p_lufs, safe_value)
    lufs_adjustment = min(-50 - lufs, safe_value)
    c_lufs_adjustment = min(-50 - c_lufs, safe_value)

    filename = os.path.basename(audio_path)
    
    # Adjust and save audio for each LUFS target
    for adjustment, prefix, dir_path in [
        (p_lufs_adjustment, "p_lufs_", output_dir),
        (lufs_adjustment, "lufs_", output_lufs_dir),
        (c_lufs_adjustment, "c_lufs_", output_centroid_dir)
    ]:
        board = Pedalboard([Gain(gain_db=adjustment)])
        effected_audio = board(audio, sr)
        output_path = os.path.join(dir_path, f"{prefix}{filename}")
        sf.write(output_path, effected_audio, sr)
        print(f"Processed file saved as: {output_path}")

    return peak_dB, p_lufs_adjustment, lufs_adjustment, c_lufs_adjustment

def mix_to_stereo(audio_files):
    """Mixes audio files into a single stereo file."""
    mixed_audio = None
    for file in audio_files:
        audio, sr = librosa.load(file, sr=None, mono=False)
        if audio.ndim == 1:
            audio = np.tile(audio, (2, 1))
        if mixed_audio is None:
            mixed_audio = audio
        else:
            min_len = min(mixed_audio.shape[1], audio.shape[1])
            mixed_audio[:, :min_len] += audio[:, :min_len]
    return mixed_audio, sr

def bounce_to_stereo(path, folder_name, suffix):
    """Bounces all audio files in a folder to a single stereo file."""
    folder_path = os.path.join(path, folder_name)
    audio_files = glob.glob(os.path.join(folder_path, '*.wav'))
    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return

    mixed_audio, sr = mix_to_stereo(audio_files)
    mixed_audio = np.clip(mixed_audio, -1, 1)

    output_filename = os.path.basename(path) + suffix + '.wav'
    output_path = os.path.join(path, output_filename)
    sf.write(output_path, mixed_audio.T, sr)
    print(f"Mixed audio saved to: {output_path}")

def save_as_json(results, path, filename):
    """Saves the results dictionary as a JSON file, converting NumPy objects to lists."""
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        
    full_path = os.path.join(path, filename)
    with open(full_path, 'w') as f:
        json.dump(results, f, indent=4, default=default)

def save_as_csv(results, path, filename):
    """Saves the results dictionary as a CSV file."""
    flatten_data = []
    for audio_path, data in results.items():
        flat_data = {'audio_path': audio_path}
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_data[f'{key}_{sub_key}'] = sub_value
            else:
                flat_data[key] = value
        flatten_data.append(flat_data)

    df = pd.DataFrame(flatten_data)
    full_path = os.path.join(path, filename)
    df.to_csv(full_path, index=False)

def check_audio_events(audio, sr):
    """Detects whether there are effective audio events based on LUFS."""
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio)
    is_effective_event = lufs > -50
    return is_effective_event, lufs

def safe_max(values):
    """Returns the maximum value among the provided values, ignoring None values."""
    valid_values = [v for v in values if v is not None]
    return max(valid_values) if valid_values else None

def process_audio_files_in_path(path):
    audio_files = glob.glob(os.path.join(path, '*.wav'))
    results = {}

    # Create directories outside the loop
    output_dir = os.path.join(path, "p_lufs")
    output_lufs_dir = os.path.join(path, "lufs")
    output_centroid_dir = os.path.join(path, "centroid")

    for dir_path in [output_dir, output_lufs_dir, output_centroid_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for audio_path in audio_files:
        audio, sr = read_audio(audio_path)
        is_event, lufs_value = check_audio_events(audio, sr)
        if is_event:
            f_crepe = CREPE(audio, sr)
            f_pyin = PYIN(audio, sr)
            f_pesto = PESTO(audio_path)
            f0 = safe_max([f_crepe, f_pyin, f_pesto])

            filtered_audio, highpass, lowpass = bandpass(audio, sr, f0)
            filtered_audio_c, highpass_c, lowpass_c = bandpass_c(audio, sr)

            p_lufs = calculate_p_lufs(filtered_audio, sr)
            c_lufs = calculate_p_lufs(filtered_audio_c, sr)

            peak, p_lufs_adjustment, lufs_adjustment, c_lufs_adjustment = match_loudness(audio, sr, p_lufs, lufs_value, c_lufs, audio_path, output_dir, output_lufs_dir, output_centroid_dir)

            result = {
                'orig_lufs_narrowband': p_lufs,
                'orig_lufs_fullband': lufs_value,
                'peak': peak,
                'adjustment_narrowband': p_lufs_adjustment,
                'adjustment_fullband': lufs_adjustment,
                'adjustment_centroid': c_lufs_adjustment,
                'F0': f0,
                'CREPE': f_crepe,
                'PYIN': f_pyin,
                'PESTO': f_pesto,
                'bandpass': {'lowstop': highpass, 'highstop': lowpass},  
                'bandpass_c': {'lowstop': highpass_c, 'highstop': lowpass_c},            
                'spectral_centroid': spectral_centroid(audio, sr),
                'spectral_bandwidth': spectral_bandwidth(audio, sr),
            }
            results[audio_path] = result
        else:
            print(f"No effective audio event in {audio_path}, LUFS: {lufs_value}. Audio file copied to: {output_dir}, {output_lufs_dir}, and {output_centroid_dir}")

            # Copy the audio file to all directories as no effective audio event detected
            for dir_path in [output_dir, output_lufs_dir, output_centroid_dir]:
                shutil.copy(audio_path, dir_path)

            results[audio_path] = {'result': 'No effective audio event'}
    
    # Save the results in JSON and CSV formats
    save_as_json(results, path, 'results.json')
    save_as_csv(results, path, 'results.csv')

    # Bounce mixed audio to stereo for each directory
    bounce_to_stereo(path, 'p_lufs', '_equal_p_loudness')
    bounce_to_stereo(path, 'lufs', '_equal_loudness')
    bounce_to_stereo(path, 'centroid', '_centroid_adjustment')


if __name__ == '__main__':
    total_path = r'/Users/shijinjie/Downloads/SHELIDUAN STEM' # 这里假设是总路径`

    # 遍历总路径下的所有子文件夹
    for folder in os.listdir(total_path):
        folder_path = os.path.join(total_path, folder)
        if os.path.isdir(folder_path):
            print(f"Processing {folder_path}...")
            process_audio_files_in_path(folder_path)  



    
    



