"""Biologically inspired spiking neural network with STP and STDP."""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Tuple

MIN_SIM_FREQUENCY_HZ = 0.01
MAX_SIM_FREQUENCY_HZ = 5000.0
DEFAULT_SIM_FREQUENCY_HZ = 200.0
STDP_WINDOW_MS = 50.0
STP_RECOVERY_MS = 200.0

ProgressCallback = Callable[[int, int, str], None]


def text_to_spikes(text: str) -> List[Tuple[str, int]]:
    """Return ASCII characters and their code values as pulse counts."""
    german_transliteration = str.maketrans(
        {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",
            "Ä": "Ae",
            "Ö": "Oe",
            "Ü": "Ue",
            "ß": "ss",
        }
    )
    ascii_text = text.translate(german_transliteration).encode("ascii", errors="replace")
    return [(chr(code), code) for code in ascii_text]


@dataclass
class DigitalNeuron:
    """Integrate charge, count spikes, and estimate the observed firing rate."""

    neuron_id: int
    threshold: float = 1.0
    charge: float = 0.0
    spike_count: int = 0
    last_spike_time: Optional[float] = None
    _created_at: float = field(default_factory=time.perf_counter, repr=False)

    @property
    def firing_rate_hz(self) -> float:
        elapsed = time.perf_counter() - self._created_at
        return self.spike_count / elapsed if elapsed > 0 else 0.0

    def receive_impulse(self, amount: float) -> bool:
        """Add charge and record a spike timestamp in monotonic milliseconds."""
        self.charge += amount
        if self.charge < self.threshold:
            return False

        self.charge = 0.0
        self.spike_count += 1
        self.last_spike_time = time.perf_counter_ns() / 1_000_000
        return True


@dataclass
class DigitalSynapse:
    """A bounded long-term weight plus a recovering short-term gain."""

    source_id: int
    target_id: int
    weight: float = 0.5
    learning_rate: float = 0.04
    short_term_gain: float = 1.0
    last_pre_spike_time: Optional[float] = None
    last_post_spike_time: Optional[float] = None

    def on_pre_spike(self, spike_time_ms: float, events: List[str]) -> float:
        """Apply anti-causal LTD, facilitate STP, and return transmitted charge."""
        previous_pre_time = self.last_pre_spike_time
        if self.last_post_spike_time is not None:
            delta_ms = spike_time_ms - self.last_post_spike_time
            if 0 < delta_ms <= STDP_WINDOW_MS:
                self._change_weight(-self.learning_rate, events, "LTD")

        if previous_pre_time is not None:
            elapsed_ms = max(0.0, spike_time_ms - previous_pre_time)
            recovery = math.exp(-elapsed_ms / STP_RECOVERY_MS)
            self.short_term_gain = 1.0 + (self.short_term_gain - 1.0) * recovery
        self.short_term_gain = min(1.5, self.short_term_gain + 0.1)
        self.last_pre_spike_time = spike_time_ms
        return self.weight * self.short_term_gain

    def on_post_spike(self, spike_time_ms: float, events: List[str]) -> None:
        """Apply causal LTP only when the preceding spike is within 50 ms."""
        if self.last_pre_spike_time is not None:
            delta_ms = spike_time_ms - self.last_pre_spike_time
            if 0 < delta_ms <= STDP_WINDOW_MS:
                self._change_weight(self.learning_rate, events, "LTP")
        self.last_post_spike_time = spike_time_ms

    def _change_weight(self, delta: float, events: List[str], rule: str) -> None:
        previous_weight = self.weight
        self.weight = min(1.0, max(0.0, self.weight + delta))
        if self.weight != previous_weight:
            events.append(
                f"{rule}: Synapse {self.source_id} -> {self.target_id} "
                f"({previous_weight:.2f} -> {self.weight:.2f})"
            )


class SpikingNeuralNetwork:
    """Persistent network state and paced input/feedback operations."""

    def __init__(
        self,
        neuron_count: int = 10,
        connection_probability: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        if neuron_count < 1:
            raise ValueError("neuron_count must be at least 1")
        if not 0.0 <= connection_probability <= 1.0:
            raise ValueError("connection_probability must be between 0 and 1")

        self._random = random.Random(seed)
        self.neurons = [DigitalNeuron(neuron_id=i) for i in range(neuron_count)]
        self.synapses = [
            DigitalSynapse(source_id=source, target_id=target)
            for source in range(neuron_count)
            for target in range(neuron_count)
            if source != target and self._random.random() < connection_probability
        ]
        self._outgoing = {neuron.neuron_id: [] for neuron in self.neurons}
        self._incoming = {neuron.neuron_id: [] for neuron in self.neurons}
        for synapse in self.synapses:
            self._outgoing[synapse.source_id].append(synapse)
            self._incoming[synapse.target_id].append(synapse)

        self._simulation_epoch_ms = time.perf_counter_ns() / 1_000_000
        self._events: Deque[str] = deque(maxlen=2000)

    @staticmethod
    def _validate_frequency(sim_frequency: float) -> None:
        if not MIN_SIM_FREQUENCY_HZ <= sim_frequency <= MAX_SIM_FREQUENCY_HZ:
            raise ValueError(
                f"sim_frequency must be between {MIN_SIM_FREQUENCY_HZ:g} and "
                f"{MAX_SIM_FREQUENCY_HZ:g} Hz"
            )

    @staticmethod
    def _pace_tick(tick_started: float, target_frame_time: float) -> None:
        """Sleep for most of the remaining frame, then finish against the clock."""
        remaining_time = target_frame_time - (time.perf_counter() - tick_started)
        sleep_margin = min(0.0001, target_frame_time / 2)
        if remaining_time > sleep_margin:
            time.sleep(remaining_time - sleep_margin)
        while time.perf_counter() - tick_started < target_frame_time:
            pass

    def stimulate(
        self,
        neuron_id: int,
        amount: float = 1.1,
        sim_frequency: float = DEFAULT_SIM_FREQUENCY_HZ,
    ) -> None:
        """Inject one impulse and pace each resulting network processing tick."""
        if neuron_id not in range(len(self.neurons)):
            raise IndexError(f"Unknown neuron: {neuron_id}")
        self._validate_frequency(sim_frequency)

        target_frame_time = 1.0 / sim_frequency
        self._events.append(f"Reiz: Neuron {neuron_id} erhält {amount:.2f} Ladung.")
        pending = deque([(neuron_id, amount)])
        processed = set()

        while pending:
            current_id, impulse = pending.popleft()
            if current_id in processed:
                continue
            processed.add(current_id)

            tick_started = time.perf_counter()
            neuron = self.neurons[current_id]
            if not neuron.receive_impulse(impulse):
                self._events.append(
                    f"Neuron {current_id}: Ladung {neuron.charge:.2f}, kein Spike."
                )
            else:
                spike_time_ms = neuron.last_spike_time
                relative_time_ms = spike_time_ms - self._simulation_epoch_ms
                self._events.append(
                    f"Neuron {current_id} feuert bei {relative_time_ms:.2f} ms "
                    f"(Rate {neuron.firing_rate_hz:.2f} Hz)."
                )
                for synapse in self._incoming[current_id]:
                    synapse.on_post_spike(spike_time_ms, self._events)
                for synapse in self._outgoing[current_id]:
                    transmitted_charge = synapse.on_pre_spike(
                        spike_time_ms, self._events
                    )
                    pending.append((synapse.target_id, transmitted_charge))

            self._pace_tick(tick_started, target_frame_time)

    def encode_text(
        self,
        text: str,
        sim_frequency: float = DEFAULT_SIM_FREQUENCY_HZ,
        input_neuron_id: int = 0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> int:
        """Encode each ASCII code as that many paced impulses on the input neuron."""
        self._validate_frequency(sim_frequency)
        if input_neuron_id not in range(len(self.neurons)):
            raise IndexError(f"Unknown input neuron: {input_neuron_id}")

        encoded_characters = text_to_spikes(text)
        total_pulses = 0
        total_characters = len(encoded_characters)
        for character_index, (character, pulse_count) in enumerate(
            encoded_characters, start=1
        ):
            self._events.append(
                f"Textzeichen {character!r} (ASCII {pulse_count}) -> "
                f"{pulse_count} Impulse auf Neuron {input_neuron_id}."
            )
            for _ in range(pulse_count):
                self.stimulate(
                    input_neuron_id,
                    sim_frequency=sim_frequency,
                )
            total_pulses += pulse_count
            if progress_callback is not None:
                progress_callback(character_index, total_characters, character)

        return total_pulses

    def apply_reward(
        self, sim_frequency: float = DEFAULT_SIM_FREQUENCY_HZ
    ) -> None:
        """Send an even paced signal and reinforce recently active synapses."""
        self._validate_frequency(sim_frequency)
        now_ms = time.perf_counter_ns() / 1_000_000
        active_synapses = [
            synapse
            for synapse in self.synapses
            if any(
                spike_time is not None and 0 <= now_ms - spike_time <= STDP_WINDOW_MS
                for spike_time in (
                    synapse.last_pre_spike_time,
                    synapse.last_post_spike_time,
                )
            )
        ]

        self._events.append("Feedback BELONUNG: harmonisches Signal gestartet.")
        neuron_count = len(self.neurons)
        for index, neuron in enumerate(self.neurons):
            harmonic_charge = 0.1 + 0.05 * (
                1.0 + math.sin(2.0 * math.pi * index / neuron_count)
            )
            self.stimulate(neuron.neuron_id, harmonic_charge, sim_frequency)

        for synapse in active_synapses:
            synapse._change_weight(0.02, self._events, "Belohnung/LTP")
        self._events.append(
            f"Belohnung abgeschlossen: {len(active_synapses)} aktive Synapsen gefestigt."
        )

    def apply_punishment(
        self, sim_frequency: float = DEFAULT_SIM_FREQUENCY_HZ
    ) -> None:
        """Send random impulses to every neuron and weaken low-weight pathways."""
        self._validate_frequency(sim_frequency)
        self._events.append("Feedback BESTRAFUNG: chaotisches Rauschen gestartet.")
        for neuron in self.neurons:
            self.stimulate(
                neuron.neuron_id,
                amount=self._random.uniform(0.2, 1.25),
                sim_frequency=sim_frequency,
            )

        if self.synapses:
            weak_pathways = sorted(self.synapses, key=lambda synapse: synapse.weight)
            pathway_count = max(1, len(weak_pathways) // 3)
            for synapse in weak_pathways[:pathway_count]:
                synapse._change_weight(-0.08, self._events, "Bestrafung/LTD")
        self._events.append("Bestrafung abgeschlossen: Rauschen und Pfadabschwächung angewendet.")

    def drain_events(self) -> List[str]:
        """Return and clear events generated since the previous drain."""
        events = list(self._events)
        self._events.clear()
        return events