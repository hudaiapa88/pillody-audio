#!/usr/bin/env python3
"""
Pillody Audio — build the app's sound library from the approved catalog.

Reads Studio/library/catalog.json (owner-approved, licence-clean sounds) and
writes app-ready files next to this script:

  SND-<key>.flac      looping beds and synth beds: trimmed past any fade,
                      closed with a crossfade from tail into head, -16 LUFS
  EVT-<key>-<n>.flac  one-shot events (every variation): edges faded,
                      -18 LUFS, played by the app at intervals

All files are 48 kHz stereo FLAC, true peak <= -1 dBTP. Beds and events are
16-bit with triangular dither: the sources are generated audio whose lowest
bits are noise, and 16-bit halves every download. FREQ/SWEEP layers are not
touched here and stay 24-bit (build_tracks.py).

Sounds the app already has from our own synthesis are not rebuilt:
brown/pink noise (NOISE-01/02) and the 432 Hz drone and pads (DRONE-02/03/04).

Usage:
  python build_library.py                 # everything missing
  python build_library.py rain_ fire_     # only keys containing the text
  python build_library.py --force         # rebuild existing files too
Requires: numpy, ffmpeg on PATH, the Studio Drive folder.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from build_tracks import MAX_TRUE_PEAK, SR, _loudness, _settled

ROOT = Path(__file__).resolve().parent
DRIVE = Path("G:/Drive'ım")
CATALOG = DRIVE / 'Studio/library/catalog.json'
MANIFEST = ROOT / 'library.json'

BED_LUFS = -16.0
EVENT_LUFS = -18.0
LIMIT_DB = -1.5   # limiter ceiling (sample peak), true peak stays <= -1
MAX_LIMIT = 18.0   # most the limiter may take off the highest peak, dB

# Already in the app from our own synthesis (build_tracks.py / SOURCES.md).
EXISTING = {
    'noise_brown': 'NOISE-01_brown_noise_loop.flac',
    'noise_pink': 'NOISE-02_pink_noise_loop.flac',
    'drone_432_hum': 'DRONE-02_hum_432hz_drone_loop.flac',
    'pad_432_atmospheric': 'DRONE-03_atmospheric_432hz_pad_loop.flac',
    'pad_432_sleep_minimal': 'DRONE-04_sleep_432hz_minimal_pad_loop.flac',
}

# Long synth renders: loop a slice instead of the whole ten minutes.
# (start s, length s, crossfade s, equal_power)
SYNTH_SLICE = {
    'noise_white': (60, 62, 2.0, True),
    'noise_green': (60, 62, 2.0, True),
    'noise_grey': (60, 62, 2.0, True),
    'singing_bowl_432': (60, 124, 4.0, False),
}


def read(path, start=None, length=None):
    cmd = ['ffmpeg', '-v', 'error']
    if start is not None:
        cmd += ['-ss', str(start), '-t', str(length)]
    cmd += ['-i', str(path), '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def close_loop(x, fade_s, equal_power=True, trim=True):
    """Trim baked-in fades off both ends, then blend the tail into the head."""
    if trim:
        mid = x[len(x) // 3: 2 * len(x) // 3].mean(axis=1)
        ref = 20 * np.log10(np.sqrt(np.mean(mid ** 2)) + 1e-12)
        # Only real silence or a fade counts: 12 dB under the body level.
        a = _settled(x, False, ref, 12.0)
        b = len(x) - _settled(x, True, ref, 12.0)
        # A sound with quiet stretches (drips, owls) can read as "not settled"
        # deep into the file; never trim more than 5 s off either end.
        a = min(a, 5 * SR)
        b = max(b, len(x) - 5 * SR)
        x = x[a:b]
    c = int(fade_s * SR)
    u = np.linspace(0, 1, c, endpoint=False)[:, None]
    fin, fout = ((np.sin(u * np.pi / 2), np.cos(u * np.pi / 2))
                 if equal_power else (u, 1 - u))
    out = x[:len(x) - c].copy()
    out[:c] = x[:c] * fin + x[len(x) - c:] * fout
    return out


def shape_event(x):
    """Cut leading/trailing silence and fade the edges so nothing clicks."""
    m = np.abs(x).max(axis=1)
    loud = np.nonzero(m > m.max() * 10 ** (-50 / 20))[0]
    if len(loud):
        a = max(loud[0] - int(0.02 * SR), 0)
        b = min(loud[-1] + int(0.25 * SR), len(x))
        x = x[a:b].copy()
    fi, fo = int(0.01 * SR), min(int(0.3 * SR), len(x) // 3)
    x[:fi] *= np.linspace(0, 1, fi)[:, None]
    x[len(x) - fo:] *= np.linspace(1, 0, fo)[:, None]
    return x


def _render(tmp, out, gain, trim=0.0):
    """Apply gain, hold peaks under the ceiling with a look-ahead limiter,
    then `trim` dB after it, and write 16-bit."""
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', str(tmp), '-af',
         f'volume={gain:.3f}dB,'
         f'alimiter=limit={10 ** (LIMIT_DB / 20):.4f}:attack=5:release=80:'
         'level=disabled:latency=1,'
         f'volume={trim:.3f}dB,'
         'aresample=osf=s16:dither_method=triangular',
         '-c:a', 'flac', '-sample_fmt', 's16', '-compression_level', '8',
         str(out)], check=True)


def write(x, out, target):
    """Bring the sound to the loudness target.

    Rain on glass, fire and drips carry short peaks 30 dB over their body, so
    plain gain would leave them far too quiet. A limiter shaves those peaks;
    it may take off at most MAX_LIMIT dB, beyond which the sound stays
    quieter instead of being crushed.
    """
    tmp = out.with_name(f'.tmp_{out.name}')
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac',
         '2', '-i', '-', '-c:a', 'flac', '-sample_fmt', 's32',
         '-bits_per_raw_sample', '24', str(tmp)], stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f'ffmpeg failed for {out.name}')
    lufs, peak = _loudness(tmp)
    ceiling = LIMIT_DB - peak + MAX_LIMIT
    gain = min(target - lufs, ceiling)
    _render(tmp, out, gain)
    got, _ = _loudness(out)
    # Limiting lowers loudness a little; make it up once.
    if target - got > 0.3 and gain < ceiling:
        gain = min(gain + (target - got), ceiling)
        _render(tmp, out, gain)
    # The limiter works on samples; peaks between samples can still pass
    # the true-peak ceiling. Take the excess off after the limiter.
    lufs, peak = _loudness(out)
    trim = 0.0
    for _ in range(3):
        if peak <= MAX_TRUE_PEAK:
            break
        trim -= peak - MAX_TRUE_PEAK + 0.1
        _render(tmp, out, gain, trim)
        lufs, peak = _loudness(out)
    tmp.unlink()
    return round(lufs, 1), round(peak, 1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def entry(sound, name, source, seconds, lufs, peak):
    return {
        'file': name, 'seconds': round(seconds, 2), 'lufs': lufs,
        'true_peak': peak, 'sha256': sha256(ROOT / name),
        'source': source, 'licence': sound['licence'],
    }


def main():
    args = sys.argv[1:]
    force = '--force' in args
    wanted = [a for a in args if not a.startswith('--')]
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    manifest = (json.loads(MANIFEST.read_text(encoding='utf-8'))
                if MANIFEST.exists() else {})

    for sound in catalog['sounds']:
        key = sound['key']
        if wanted and not any(w in key for w in wanted):
            continue
        base = {k: sound[k] for k in
                ('label_tr', 'kind', 'category', 'uses', 'sleep_safe')}
        if key in EXISTING:
            manifest[key] = {**base, 'files': [{'file': EXISTING[key],
                                                'existing': True}]}
            continue
        done = manifest.get(key, {}).get('files', [])
        if (not force and done and
                all((ROOT / f['file']).exists() for f in done)):
            continue
        files = []
        try:
            if sound['kind'] == 'event':
                for n, v in enumerate(sound['variations'], 1):
                    name = f'EVT-{key}-{n}.flac'
                    x = shape_event(read(DRIVE / v['file']))
                    lufs, peak = write(x, ROOT / name, EVENT_LUFS)
                    files.append(entry(sound, name, v['file'], len(x) / SR,
                                       lufs, peak))
            else:
                v = sound['variations'][0]
                name = f'SND-{key}.flac'
                if key in SYNTH_SLICE:
                    start, length, fade, eq = SYNTH_SLICE[key]
                    x = close_loop(read(DRIVE / v['file'], start, length),
                                   fade, eq, trim=False)
                else:
                    x = close_loop(read(DRIVE / v['file']), 2.0)
                lufs, peak = write(x, ROOT / name, BED_LUFS)
                files.append(entry(sound, name, v['file'], len(x) / SR,
                                   lufs, peak))
        except Exception as e:  # keep going; report at the end
            print(f'FAILED {key}: {e}', flush=True)
            continue
        manifest[key] = {**base, 'files': files}
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                            encoding='utf-8')
        for f in files:
            print(f"{f['file']}: {f['seconds']} s, {f['lufs']} LUFS, "
                  f"peak {f['true_peak']} dBTP", flush=True)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                        encoding='utf-8')
    missing = [s['key'] for s in catalog['sounds'] if s['key'] not in manifest]
    print(f'done: {len(manifest)} of {len(catalog["sounds"])} sounds'
          + (f', missing: {missing}' if missing else ''), flush=True)


if __name__ == '__main__':
    main()
