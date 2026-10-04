"""Translate text to timed input spikes and output-neuron activity to tokens."""

from __future__ import annotations

import math
import time
from typing import Callable, List, Tuple

from brain import (
    MAX_SIM_FREQUENCY_HZ,
    MIN_SIM_FREQUENCY_HZ,
    BiologicalBrain,
)

OUTPUT_WINDOW_NS = 1_000_000_000
HIGH_FREQUENCY_HZ = 10.0
ProgressCallback = Callable[[int, int], None]


def text_to_spikes(
    text: str,
    brain: BiologicalBrain,
    sim_frequency: float,
    progress_callback: ProgressCallback | None = None,
) -> int:
    """Inject ASCII-derived pulses into input neuron 0 at the selected tick rate."""
    if not math.isfinite(sim_frequency) or not (
        MIN_SIM_FREQUENCY_HZ <= sim_frequency <= MAX_SIM_FREQUENCY_HZ
    ):
        raise ValueError(
            f"sim_frequency must be between {MIN_SIM_FREQUENCY_HZ:g} and "
            f"{MAX_SIM_FREQUENCY_HZ:g} Hz"
        )

    tick_seconds = 1.0 / sim_frequency
    ascii_values = [
        ord(character) if ord(character) < 128 else ord("?") for character in text
    ]
    total_spikes = sum(int(ascii_value / 10) for ascii_value in ascii_values)
    injected_spikes = 0
    for character_index, ascii_value in enumerate(ascii_values):
        spike_count = int(ascii_value / 10)
        for _ in range(spike_count):
            tick_started = time.perf_counter()
            brain.inject_impulse(0, 0.4)
            injected_spikes += 1
            if progress_callback is not None:
                progress_callback(injected_spikes, total_spikes)
            remaining_tick = tick_seconds - (time.perf_counter() - tick_started)
            if remaining_tick > 0.0:
                time.sleep(remaining_tick)
        if character_index + 1 < len(text):
            time.sleep(tick_seconds * 5)
    return injected_spikes


def _decode_ascii_pattern(spikes: List[Tuple[int, int]]) -> str:
    """Decode groups of seven output events as an optional 7-bit text stream."""
    bits: List[int] = []
    for _, neuron_id in spikes:
        if neuron_id == 8:
            bits.append(1)
        elif neuron_id == 9:
            bits.append(0)

    decoded: List[str] = []
    for offset in range(0, len(bits) - 6, 7):
        value = 0
        for bit in bits[offset : offset + 7]:
            value = (value << 1) | bit
        if 32 <= value <= 126:
            decoded.append(chr(value))
    return "".join(decoded).strip()


def spikes_to_output(brain: BiologicalBrain) -> str:
    """Map recent output-neuron rates to tokens and decode optional 7-bit patterns."""
    now_ns = time.time_ns()
    recent_spikes = sorted(
        (
            (timestamp_ns, neuron.neuron_id)
            for neuron_id in brain.output_neuron_ids
            for neuron in (brain.neurons[neuron_id],)
            for timestamp_ns in neuron.spike_times_ns
            if 0 <= now_ns - timestamp_ns <= OUTPUT_WINDOW_NS
        ),
        key=lambda spike: (spike[0], spike[1]),
    )

    tokens: List[str] = []
    for neuron_id, low_token, high_token in (
        (8, "if", "while"),
        (9, "print(", ": "),
    ):
        count = sum(1 for _, spike_neuron_id in recent_spikes if spike_neuron_id == neuron_id)
        if count:
            rate_hz = count / (OUTPUT_WINDOW_NS / 1_000_000_000)
            tokens.append(
                low_token if rate_hz < HIGH_FREQUENCY_HZ else high_token
            )

    name_pattern = _decode_ascii_pattern(recent_spikes)
    if name_pattern:
        tokens.append(name_pattern)
    return "".join(tokens)
