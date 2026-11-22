"""Local onset suggestions for cuts (not a tempo grid or a snap operation)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def suggest_cut_onsets(audio, *, max_points=2400):
    """Pick strong, separated rises in multiband spectral energy; return seconds."""
    source_rate = int(audio["sample_rate"])
    # Downsample cheaply with an averaging filter, preserving each stereo channel.
    divisor = max(1, source_rate // 8000)
    rate = source_rate / divisor
    wave = audio["waveform"].squeeze(0)
    if wave.shape[-1] < source_rate / 5:
        return []
    if divisor > 1:
        wave = F.avg_pool1d(wave, divisor, stride=divisor)
    size, hop = 512, max(1, round(rate / 100))
    windows = F.pad(wave, (size // 2, size // 2)).unfold(-1, size, hop)
    win = torch.hann_window(size, dtype=wave.dtype, device=wave.device)
    gain = 2 / win.sum()
    bands = [(max(1, round(low * size / rate)), min(size // 2 + 1, round(high * size / rate)), weight)
             for low, high, weight in ((35, 180, 0.50), (180, 1800, 0.32), (1800, 6500, 0.18))]
    parts, prev = [], None
    for batch in windows.split(1024, dim=1):
        magnitude = torch.fft.rfft(batch * win).abs().square().mean(dim=0).sqrt() * gain
        compressed = torch.log1p(1000 * magnitude)
        if prev is None:
            prev = compressed[:1]
        delta = torch.cat((compressed[:1] - prev, compressed[1:] - compressed[:-1])).clamp_min_(0)
        prev = compressed[-1:]
        parts.append(sum(delta[:, lo:hi].mean(dim=1) * weight for lo, hi, weight in bands if hi > lo))
    flux = torch.cat(parts)
    signal = flux.view(1, 1, -1)
    baseline = F.avg_pool1d(signal, 71, stride=1, padding=35, count_include_pad=False).flatten()
    maxima = F.max_pool1d(signal, 7, stride=1, padding=3).flatten()
    threshold = torch.maximum(baseline * 1.8, torch.full_like(baseline, 0.045))
    indexes = torch.where((flux >= maxima) & (flux > threshold))[0].tolist()
    values = flux.tolist()
    candidates = sorted(indexes, key=values.__getitem__, reverse=True)
    import bisect

    chosen = []
    gap = round(0.18 * rate / hop)
    for index in candidates:
        at = bisect.bisect_left(chosen, index)
        if (at and index - chosen[at - 1] < gap) or (at < len(chosen) and chosen[at] - index < gap):
            continue
        bisect.insort(chosen, index)
        if len(chosen) >= max_points:
            break
    duration = audio["waveform"].shape[-1] / source_rate
    return [round(i * hop / rate, 4) for i in chosen if i * hop / rate < duration]
