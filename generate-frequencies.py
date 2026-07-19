#!/usr/bin/env python3
"""
Pillody Audio — Missing Frequency Generator

Generates 4 missing frequency files needed by the catalog:
  FREQ-10: Isochronic 16Hz beta (300s)
  FREQ-11: Binaural 0.5Hz epsilon (900s)
  FREQ-12: Binaural 1.0Hz delta (600s)
  FREQ-13: Isochronic 18Hz beta (180s)

Output: 48kHz / 24-bit / stereo FLAC
Requires: numpy, scipy, ffmpeg
"""

import numpy as np
import subprocess
import os
import sys

SAMPLE_RATE = 48000
BIT_DEPTH = 24  # int24 equivalent via int32 with padding
FADE_SECONDS = 3.0


def generate_isochronic(freq_hz, carrier_hz, duration_sec):
    """Generate isochronic tone: carrier sine wave with amplitude modulation at target freq."""
    total_samples = int(SAMPLE_RATE * duration_sec)
    t = np.arange(total_samples) / SAMPLE_RATE

    # Carrier sine wave
    carrier = np.sin(2 * np.pi * carrier_hz * t)

    # Amplitude modulation at target frequency (square wave pulse = on/off)
    # 50% duty cycle — sound on for first half of each cycle, off for second half
    am_freq = freq_hz
    am_phase = (t * am_freq) % 1.0  # 0..1 repeating
    am_envelope = np.where(am_phase < 0.5, 1.0, 0.0)

    # Smooth the edges slightly to avoid clicks (1ms ramp)
    ramp_samples = int(SAMPLE_RATE * 0.001)
    for i in range(len(am_envelope)):
        cycle_pos = (t[i] * am_freq) % 1.0
        if 0.5 - (ramp_samples / SAMPLE_RATE * am_freq) < cycle_pos < 0.5:
            # Ramp down
            local_t = (cycle_pos - (0.5 - ramp_samples / SAMPLE_RATE * am_freq)) / (ramp_samples / SAMPLE_RATE * am_freq)
            am_envelope[i] = 1.0 - local_t
        elif cycle_pos < ramp_samples / SAMPLE_RATE * am_freq:
            # Ramp up
            am_envelope[i] = cycle_pos / (ramp_samples / SAMPLE_RATE * am_freq)

    signal = carrier * am_envelope

    # Fade in/out
    fade_samples = int(SAMPLE_RATE * FADE_SECONDS)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out

    # Stereo (same signal both channels for isochronic)
    stereo = np.column_stack([signal, signal])
    return stereo


def generate_binaural(beat_hz, carrier_hz, duration_sec):
    """Generate binaural beat: left channel = carrier, right channel = carrier + beat freq."""
    total_samples = int(SAMPLE_RATE * duration_sec)
    t = np.arange(total_samples) / SAMPLE_RATE

    left = np.sin(2 * np.pi * carrier_hz * t)
    right = np.sin(2 * np.pi * (carrier_hz + beat_hz) * t)

    # Fade in/out
    fade_samples = int(SAMPLE_RATE * FADE_SECONDS)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    left[:fade_samples] *= fade_in
    left[-fade_samples:] *= fade_out
    right[:fade_samples] *= fade_in
    right[-fade_samples:] *= fade_out

    stereo = np.column_stack([left, right])
    return stereo


def save_as_flac(stereo_signal, output_path):
    """Save numpy array as 24-bit WAV then convert to FLAC via ffmpeg."""
    wav_path = output_path.replace('.flac', '.wav')

    # Convert float64 to int32 (24-bit audio in 32-bit container)
    # Normalize to -1.0..1.0 then scale to int32 range
    max_val = np.max(np.abs(stereo_signal))
    if max_val > 0:
        stereo_signal = stereo_signal / max_val * 0.9  # leave headroom

    int_data = (stereo_signal * (2**23 - 1)).astype(np.int32)

    # Write as 32-bit WAV (ffmpeg will convert 24-bit FLAC from this)
    from scipy.io import wavfile
    wavfile.write(wav_path, SAMPLE_RATE, int_data)

    # Convert to 24-bit FLAC using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', wav_path,
        '-c:a', 'flac',
        '-sample_fmt', 's32',
        '-ar', str(SAMPLE_RATE),
        '-ac', '2',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[:200]}")
        return False

    # Clean up WAV
    os.remove(wav_path)
    return True


def generate_sweep(start_hz, end_hz, carrier_hz, duration_sec):
    """Generate a binaural sweep from start_hz to end_hz beat frequency over duration."""
    total_samples = int(SAMPLE_RATE * duration_sec)
    t = np.arange(total_samples) / SAMPLE_RATE

    # Linear interpolation of beat frequency from start to end
    beat_freq = start_hz + (end_hz - start_hz) * (t / duration_sec)

    # Phase accumulation for time-varying frequency
    phase_left = 2 * np.pi * carrier_hz * t
    # Right channel: integrate the instantaneous frequency
    phase_right = 2 * np.pi * np.cumsum(beat_freq) / SAMPLE_RATE + 2 * np.pi * carrier_hz * t

    left = np.sin(phase_left)
    right = np.sin(phase_right)

    # Fade in/out
    fade_samples = int(SAMPLE_RATE * FADE_SECONDS)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    left[:fade_samples] *= fade_in
    left[-fade_samples:] *= fade_out
    right[:fade_samples] *= fade_in
    right[-fade_samples:] *= fade_out

    stereo = np.column_stack([left, right])
    return stereo


def main():
    print("=" * 60)
    print("Pillody Audio — Missing Frequency Generator")
    print("=" * 60)

    files_to_generate = [
        {
            'name': 'FREQ-10_isochronic_16hz_beta_loop.flac',
            'type': 'isochronic',
            'freq': 16.0,
            'carrier': 100.0,
            'duration': 300,
        },
        {
            'name': 'FREQ-11_binaural_0.5hz_epsilon_loop.flac',
            'type': 'binaural',
            'freq': 0.5,
            'carrier': 200.0,
            'duration': 900,
        },
        {
            'name': 'FREQ-12_binaural_1.0hz_delta_loop.flac',
            'type': 'binaural',
            'freq': 1.0,
            'carrier': 200.0,
            'duration': 600,
        },
        {
            'name': 'FREQ-13_isochronic_18hz_beta_loop.flac',
            'type': 'isochronic',
            'freq': 18.0,
            'carrier': 100.0,
            'duration': 180,
        },
        {
            'name': 'FREQ-14_binaural_0.25hz_slow_oscillation_loop.flac',
            'type': 'binaural',
            'freq': 0.25,
            'carrier': 250.0,
            'duration': 600,
        },
        {
            'name': 'SWEEP-03_theta_to_alpha_7_10hz.flac',
            'type': 'sweep',
            'start_hz': 7.0,
            'end_hz': 10.0,
            'carrier': 200.0,
            'duration': 600,
        },
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    success = 0
    failed = 0

    for f in files_to_generate:
        output_path = os.path.join(script_dir, f['name'])

        if os.path.exists(output_path):
            print(f"  SKIP {f['name']} (already exists)")
            success += 1
            continue

        freq_label = f"{f['start_hz']}→{f['end_hz']}Hz" if f['type'] == 'sweep' else f"{f['freq']}Hz"
        print(f"  Generating {f['name']} ({f['type']} {freq_label}, {f['duration']}s)...")

        try:
            if f['type'] == 'isochronic':
                signal = generate_isochronic(f['freq'], f['carrier'], f['duration'])
            elif f['type'] == 'binaural':
                signal = generate_binaural(f['freq'], f['carrier'], f['duration'])
            elif f['type'] == 'sweep':
                signal = generate_sweep(f['start_hz'], f['end_hz'], f['carrier'], f['duration'])
            else:
                print(f"  ERROR: Unknown type {f['type']}")
                failed += 1
                continue

            if save_as_flac(signal, output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  DONE {f['name']} ({size_mb:.1f} MB)")
                success += 1
            else:
                print(f"  FAILED {f['name']}")
                failed += 1
        except Exception as e:
            print(f"  ERROR {f['name']}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Summary: {success} generated, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
