#!/usr/bin/env python3
"""
Pillody Audio — rebuild the catalog tracks as seamless loops.

Synthetic tracks (tones, noise, breath pacers) are rendered so that every
component completes a whole number of cycles over the loop: the last sample
runs straight into the first, with no fade. Recorded tracks (nature, drones,
bowl) are trimmed past their baked-in silence and fades, then closed with a
crossfade from their tail into their head.

Everything is normalised to -16 LUFS integrated, true peak <= -1 dBTP, and
written as 48 kHz / 24-bit stereo FLAC under the same file names.

Usage:
  python build_tracks.py            # all tracks
  python build_tracks.py FREQ-05    # only names containing the given text
Requires: numpy, scipy, ffmpeg on PATH.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 48000
ROOT = Path(__file__).resolve().parent
TARGET_LUFS = -16.0
MAX_TRUE_PEAK = -1.0
LOOP = 60  # seconds; every synthetic frequency below fits it in whole cycles
HOLD = 60  # seconds a sweep holds its target at the end (looped by the app)
CHUNK = 10 * SR
# The commit holding the untouched recordings; trimming always starts here.
ORIGINALS = 'd241303'


# ── output ──────────────────────────────────────────────────────────────────

def _encode(chunks, path):
    """Stream float32 stereo chunks into a 24-bit FLAC."""
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR),
           '-ac', '2', '-i', '-', '-c:a', 'flac', '-sample_fmt', 's32',
           '-bits_per_raw_sample', '24', str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for c in chunks:
        proc.stdin.write(np.ascontiguousarray(c, dtype=np.float32).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {path.name}')


def _loudness(path):
    err = subprocess.run(
        ['ffmpeg', '-v', 'info', '-nostats', '-i', str(path), '-af',
         'ebur128=peak=true', '-f', 'null', '-'],
        capture_output=True, text=True).stderr
    s = err[err.rfind('Summary:'):]
    return (float(re.search(r'I:\s+(-?[\d.]+) LUFS', s).group(1)),
            float(re.search(r'Peak:\s+(-?[\d.]+) dBFS', s).group(1)))


def write(chunks_fn, name):
    """Render, measure, then apply one gain to hit the loudness target."""
    out = ROOT / name
    tmp = ROOT / f'.tmp_{name}'
    _encode(chunks_fn(), tmp)
    lufs, peak = _loudness(tmp)
    gain = min(TARGET_LUFS - lufs, MAX_TRUE_PEAK - peak)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(tmp), '-af',
                    f'volume={gain:.3f}dB', '-c:a', 'flac', '-sample_fmt',
                    's32', '-bits_per_raw_sample', '24', str(out)], check=True)
    tmp.unlink()
    lufs, peak = _loudness(out)
    print(f'  {name}: {lufs:.1f} LUFS, true peak {peak:.1f} dBTP')


def whole(x):
    """A finished in-memory signal as a one-chunk generator."""
    return lambda: iter([x])


# ── tones ───────────────────────────────────────────────────────────────────

def pulse(phase, ramp=0.12):
    """Isochronic pulse shape for a cycle phase in [0, 1).

    On for the first half of the cycle, with raised-cosine edges `ramp`
    cycles long, off for the second half: full depth, no clicks.
    """
    p = np.mod(phase, 1.0)
    up = 0.5 - 0.5 * np.cos(np.pi * np.clip(p / ramp, 0, 1))
    down = 0.5 - 0.5 * np.cos(np.pi * np.clip((0.5 - p) / ramp, 0, 1))
    return np.where(p < 0.5, np.minimum(up, down), 0.0)


def isochronic(rate, carrier=432.0, seconds=LOOP):
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * carrier * t) * pulse(rate * t)
    return whole(np.column_stack([x, x]))


def binaural(beat, carrier, seconds=LOOP):
    t = np.arange(int(seconds * SR)) / SR
    left = np.sin(2 * np.pi * (carrier - beat / 2) * t)
    right = np.sin(2 * np.pi * (carrier + beat / 2) * t)
    return whole(np.column_stack([left, right]))


def sweep(f0, f1, seconds, carrier=432.0):
    """Isochronic sweep: a linear ramp f0 -> f1, then HOLD s at f1.

    The hold has whole cycles of both the pulse and the carrier, so the app
    can loop exactly that last minute once the ramp has played.
    """
    ramp = seconds - HOLD
    total = int(seconds * SR)

    def chunks():
        for start in range(0, total, CHUNK):
            t = np.arange(start, min(start + CHUNK, total)) / SR
            in_ramp = t < ramp
            tr = np.minimum(t, ramp)
            phase = f0 * tr + (f1 - f0) * tr ** 2 / (2 * ramp)
            phase = phase + np.where(in_ramp, 0.0, f1 * (t - ramp))
            x = np.sin(2 * np.pi * carrier * t) * pulse(phase)
            yield np.column_stack([x, x])

    return chunks


# ── noise ───────────────────────────────────────────────────────────────────

def periodic_noise(exponent, seconds=LOOP, correlation=0.4, seed=7,
                   low_cut=25.0, band=None):
    """Coloured noise that is exactly periodic over `seconds`.

    Built in the frequency domain (amplitude ~ f^-exponent): an inverse FFT
    of one period has no seam. The two channels share `correlation` of
    their content: wide, but not split apart in headphones.
    """
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1 / SR)
    mag = np.zeros_like(f)
    nz = f > 0
    mag[nz] = f[nz] ** -exponent
    mag *= 1 / (1 + (low_cut / np.maximum(f, 1e-9)) ** 4)  # gentle high-pass
    if band:
        lo, hi = band
        mag *= 1 / (1 + (lo / np.maximum(f, 1e-9)) ** 4)
        mag *= 1 / (1 + (f / hi) ** 4)

    def spectrum():
        return (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))

    a, b = spectrum(), spectrum()
    left = np.fft.irfft(mag * a, n)
    right = np.fft.irfft(mag * (correlation * a +
                                np.sqrt(1 - correlation ** 2) * b), n)
    x = np.column_stack([left, right])
    return x / np.max(np.abs(x))


# ── breath pacers ───────────────────────────────────────────────────────────

def breath_pacer(phases, cycles):
    """A breathing guide: a soft A=432-family tone and airy breath noise.

    `phases` is a list of (kind, seconds) with kind in inhale/hold/exhale.
    The tone swells and brightens on the inhale, stays level on the hold and
    settles on the exhale; breath noise moves only while breathing. The
    loop holds `cycles` whole cycles, and every partial completes whole
    periods in it, so it repeats without a seam.
    """
    cycle = sum(s for _, s in phases)
    seconds = cycle * cycles
    n = int(round(seconds * SR))
    t = np.arange(n) / SR
    pos = np.mod(t, cycle)

    level = np.empty(n)
    motion = np.zeros(n)  # 1 while air moves
    start = 0.0
    floor, top = 0.18, 1.0
    for kind, dur in phases:
        m = (pos >= start) & (pos < start + dur)
        u = (pos[m] - start) / dur
        ease = 0.5 - 0.5 * np.cos(np.pi * u)
        if kind == 'inhale':
            level[m] = floor + (top - floor) * ease
            motion[m] = np.sin(np.pi * u) ** 0.7
        elif kind == 'hold':
            level[m] = top - 0.08 * np.sin(np.pi * u)
        else:
            level[m] = top - (top - floor) * ease
            motion[m] = np.sin(np.pi * u) ** 0.7
        start += dur

    bright = level ** 2
    tone = (np.sin(2 * np.pi * 216 * t) * 0.55 +
            np.sin(2 * np.pi * 324 * t) * 0.30 * level +
            np.sin(2 * np.pi * 432 * t) * 0.18 * bright +
            np.sin(2 * np.pi * 648 * t) * 0.06 * bright) * level
    air = periodic_noise(0.5, seconds=seconds, correlation=0.6, seed=11,
                         band=(500, 3500))
    x = np.column_stack([tone, tone]) + air * (0.22 * motion)[:, None]
    return whole(x)


# ── recorded loops ──────────────────────────────────────────────────────────

def read_original(name):
    """The original recording, so a rebuild never trims an already-trimmed
    file a second time."""
    flac = subprocess.run(['git', '-C', str(ROOT), 'show', f'{ORIGINALS}:{name}'],
                          capture_output=True, check=True).stdout
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-f',
                          'f32le', '-ac', '2', '-ar', str(SR), '-'],
                         input=flac, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def _settled(x, from_end, ref_db, margin_db=3.0):
    """Index where the level first comes within margin of the body level."""
    w = int(0.05 * SR)
    m = x.mean(axis=1)
    if from_end:
        m = m[::-1]
    for i in range(0, len(m) - w, w):
        if 20 * np.log10(np.sqrt(np.mean(m[i:i + w] ** 2)) + 1e-12) > ref_db - margin_db:
            return i
    return 0


def crossfade_loop(name, fade_s, equal_power=True, margin_db=3.0):
    """Trim the silence and fades off both ends, then close the loop.

    The last `fade_s` seconds are blended into the first, so playback runs
    from the end of the file into its start without a gap or dip.
    """
    def chunks():
        x = read_original(name)
        mid = x[len(x) // 3: 2 * len(x) // 3].mean(axis=1)
        ref = 20 * np.log10(np.sqrt(np.mean(mid ** 2)) + 1e-12)
        a = _settled(x, False, ref, margin_db)
        b = len(x) - _settled(x, True, ref, margin_db)
        body = x[a:b]
        c = int(fade_s * SR)
        u = np.linspace(0, 1, c, endpoint=False)[:, None]
        fin, fout = ((np.sin(u * np.pi / 2), np.cos(u * np.pi / 2))
                     if equal_power else (u, 1 - u))
        out = body[:len(body) - c].copy()
        out[:c] = body[:c] * fin + body[len(body) - c:] * fout
        yield out

    return chunks


# ── catalog ─────────────────────────────────────────────────────────────────

TRACKS = {
    # isochronic, 432 Hz carrier
    'FREQ-01_isochronic_14hz_smr_loop.flac': isochronic(14),
    'FREQ-02_isochronic_10hz_alpha_loop.flac': isochronic(10),
    'FREQ-03_isochronic_40hz_gamma_loop.flac': isochronic(40),
    # binaural: carriers where the beat is clearly heard, above the
    # brown-noise floor most recipes lay under them
    'FREQ-05_binaural_2.5hz_delta_loop.flac': binaural(2.5, 200),
    'FREQ-06_binaural_2.0hz_delta_loop.flac': binaural(2.0, 200),
    'FREQ-07_binaural_3.0hz_delta_loop.flac': binaural(3.0, 200),
    'FREQ-08_binaural_6hz_theta_loop.flac': binaural(6.0, 250),
    'FREQ-09_binaural_7hz_theta_loop.flac': binaural(7.0, 250),
    # sweeps: ramp, then a loopable minute at the target
    'SWEEP-01_theta_to_smr_7_14hz.flac': sweep(7, 14, 720),
    'SWEEP-02_theta_to_delta_6_2hz.flac': sweep(6, 2, 1200),
    'SWEEP-03_theta_to_alpha_7_10hz.flac': sweep(7, 10, 600),
    'SWEEP-04_alpha_to_smr_10_14hz.flac': sweep(10, 14, 480),
    'SWEEP-05_smr_to_lowbeta_14_18hz.flac': sweep(14, 18, 900),
    # noise
    'NOISE-01_brown_noise_loop.flac': whole(periodic_noise(1.0, seed=1)),
    'NOISE-02_pink_noise_loop.flac': whole(periodic_noise(0.5, seed=2)),
    'NOISE-03_pink_brown_blend_loop.flac': whole(periodic_noise(0.75, seed=3)),
    # breath pacers
    'BREATH-01_nefes_4_7_8_rehber.flac': breath_pacer(
        [('inhale', 4), ('hold', 7), ('exhale', 8)], cycles=5),
    'BREATH-02_koherent_nefes_5_5_loop.flac': breath_pacer(
        [('inhale', 5.5), ('exhale', 5.5)], cycles=10),
    # drones and bowl: tonal, so an equal-gain crossfade
    'DRONE-01_vagal_60hz_pad_loop.flac': crossfade_loop(
        'DRONE-01_vagal_60hz_pad_loop.flac', 4, equal_power=False),
    'DRONE-02_hum_432hz_drone_loop.flac': crossfade_loop(
        'DRONE-02_hum_432hz_drone_loop.flac', 4, equal_power=False),
    'DRONE-03_atmospheric_432hz_pad_loop.flac': crossfade_loop(
        'DRONE-03_atmospheric_432hz_pad_loop.flac', 4, equal_power=False),
    'DRONE-04_sleep_432hz_minimal_pad_loop.flac': crossfade_loop(
        'DRONE-04_sleep_432hz_minimal_pad_loop.flac', 4, equal_power=False),
    'NAT-13_singing_bowl_432hz_loop.flac': crossfade_loop(
        'NAT-13_singing_bowl_432hz_loop.flac', 3, equal_power=False),
}

# Nature recordings with a gap or fade at the seam. NAT-01, NAT-04, NAT-05
# and NAT-16 already loop cleanly and are left as they are.
for _name in [
    'NAT-01b_rain_heavy_loop.flac', 'NAT-02_stream_loop.flac',
    'NAT-03_birds_morning_loop.flac', 'NAT-04b_urban_ambient_loop.flac',
    'NAT-06_thunder_loop.flac', 'NAT-07_ocean_waves_loop.flac',
    'NAT-08_breeze_trees_loop.flac', 'NAT-09_thunder_rain_loop.flac',
    'NAT-10_night_forest_loop.flac', 'NAT-10b_crickets_loop.flac',
    'NAT-11_rain_tent_loop.flac', 'NAT-12_waterfall_loop.flac',
    'NAT-12b_waterfall_woods_loop.flac', 'NAT-14_campfire_crackles_loop.flac',
    'NAT-14b_campfire_burn_loop.flac', 'NAT-15_wind_zen_loop.flac',
    'NAT-16b_jungle_rain_loop.flac',
]:
    TRACKS[_name] = crossfade_loop(_name, 2.0)

# These two fade in and out slowly over several seconds: trim closer to the
# body level and blend longer, or the seam still dips.
for _name in ['NAT-01b_rain_heavy_loop.flac', 'NAT-15_wind_zen_loop.flac']:
    TRACKS[_name] = crossfade_loop(_name, 3.0, margin_db=1.0)


def main():
    wanted = sys.argv[1:]
    for name, render in TRACKS.items():
        if wanted and not any(w in name for w in wanted):
            continue
        print(name)
        write(render, name)


if __name__ == '__main__':
    main()
