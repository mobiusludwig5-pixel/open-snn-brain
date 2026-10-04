"""Translate text to timed input spikes and output-neuron activity to tokens."""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Tuple

from brain import (
    MAX_SIM_FREQUENCY_HZ,
    MIN_SIM_FREQUENCY_HZ,
    BiologicalBrain,
)

OUTPUT_WINDOW_NS = 1_000_000_000
HIGH_FREQUENCY_HZ = 10.0
ProgressCallback = Callable[[int, int], None]
TELEMETRY_WINDOW_SECONDS = 0.01
MAX_TELEMETRY_FREQUENCY_HZ = 5000.0


def _normalized_telemetry_value(metric_name: str, value: float) -> float:
    """Normalize supported physical units to a bounded 0..1 activation value."""
    if not math.isfinite(value):
        raise ValueError(f"{metric_name} must be finite")

    normalized_name = metric_name.lower()
    if (
        "battery" in normalized_name
        or normalized_name.endswith("_pct")
        or normalized_name.endswith("_percent")
    ):
        minimum, maximum = 0.0, 100.0
    elif "temperature" in normalized_name or normalized_name.endswith("_c"):
        minimum, maximum = -40.0, 125.0
    else:
        minimum, maximum = 0.0, 200.0
    return min(1.0, max(0.0, (value - minimum) / (maximum - minimum)))


def telemetry_to_spikes(
    data_dict: dict[str, float], brain: BiologicalBrain
) -> Dict[str, float]:
    """Normalize a physical telemetry packet and inject rate-coded input spikes.

    Battery metrics target input neuron 1. Other numeric physical metrics target
    input neuron 2; distances in centimeters produce higher rates at closer range.
    The returned mapping contains the rate selected for each metric in hertz.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("data_dict must be a dictionary")

    rates_hz: Dict[str, float] = {}
    with brain._lock:
        for metric_name, raw_value in data_dict.items():
            if not isinstance(metric_name, str) or not metric_name.strip():
                raise ValueError("telemetry metric names must be non-empty strings")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError(f"{metric_name} must be a numeric value")

            value = float(raw_value)
            normalized_name = metric_name.lower()
            is_battery = "battery" in normalized_name
            is_distance = "distance" in normalized_name and normalized_name.endswith(
                "_cm"
            )
            normalized = _normalized_telemetry_value(metric_name, value)
            if is_battery and value < 30.0:
                rate_hz = 1000.0 + (30.0 - value) / 30.0 * (
                    MAX_TELEMETRY_FREQUENCY_HZ - 1000.0
                )
            elif is_distance:
                rate_hz = 1.0 + (1.0 - normalized) * 99.0
            else:
                rate_hz = 1.0 + normalized * 99.0

            neuron_id = 1 if is_battery else 2
            brain.register_telemetry_path(metric_name, neuron_id)
            if is_battery and value < 30.0:
                brain.stress_level = 80.0

            pulse_count = max(1, round(rate_hz * TELEMETRY_WINDOW_SECONDS))
            period_ns = max(1, round(1_000_000_000 / rate_hz))
            start_ns = max(time.time_ns(), brain._last_decay_time_ns)
            for pulse_index in range(pulse_count):
                brain.inject_impulse(
                    neuron_id,
                    1.0,
                    start_ns + pulse_index * period_ns,
                )
            rates_hz[metric_name] = rate_hz
    return rates_hz


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
