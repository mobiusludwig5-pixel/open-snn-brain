"""A small, biologically inspired spiking neural network with online STDP."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

MIN_SIM_FREQUENCY_HZ = 1.0
MAX_SIM_FREQUENCY_HZ = 1000.0
STDP_WINDOW_MS = 50.0


@dataclass
class DigitalNeuron:
    """Integrate incoming charge and emit a spike at the firing threshold."""

    neuron_id: int
    threshold: float = 1.0
    charge: float = 0.0
    last_spike_time: Optional[float] = None

    def receive_impulse(self, amount: float) -> bool:
        """Add charge and return whether this impulse caused a spike."""
        self.charge += amount
        if self.charge < self.threshold:
            return False

        self.charge = 0.0
        self.last_spike_time = time.perf_counter_ns() / 1_000_000
        return True


@dataclass
class DigitalSynapse:
    """A weighted connection with a pair-based, bounded STDP rule."""

    source_id: int
    target_id: int
    weight: float
    learning_rate: float = 0.04
    last_pre_spike_time: Optional[float] = None
    last_post_spike_time: Optional[float] = None

    def on_pre_spike(self, spike_time_ms: float, events: List[str]) -> None:
        self.last_pre_spike_time = spike_time_ms
        if self.last_post_spike_time is not None:
            delta_ms = spike_time_ms - self.last_post_spike_time
            if 0 < delta_ms <= STDP_WINDOW_MS:
                self._change_weight(-self.learning_rate, events, "LTD")

    def on_post_spike(self, spike_time_ms: float, events: List[str]) -> None:
        self.last_post_spike_time = spike_time_ms
        if self.last_pre_spike_time is not None:
            delta_ms = spike_time_ms - self.last_pre_spike_time
            if 0 < delta_ms <= STDP_WINDOW_MS:
                self._change_weight(self.learning_rate, events, "LTP")

    def _change_weight(self, delta: float, events: List[str], rule: str) -> None:
        previous_weight = self.weight
        self.weight = min(1.0, max(0.0, self.weight + delta))
        if self.weight != previous_weight:
            events.append(
                f"STDP {rule}: Synapse {self.source_id} -> {self.target_id} "
                f"({previous_weight:.2f} -> {self.weight:.2f})"
            )


class SpikingNeuralNetwork:
    """Network state and event-producing stimulation/feedback operations."""

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
            DigitalSynapse(source_id=source, target_id=target, weight=self._random.uniform(0.2, 0.8))
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
        self._events: Deque[str] = deque()

    def stimulate(
        self,
        neuron_id: int,
        amount: float = 1.1,
        sim_frequency: float = 200.0,
    ) -> None:
        """Inject charge and propagate spikes at the configured tick frequency."""
        if neuron_id not in range(len(self.neurons)):
            raise IndexError(f"Unknown neuron: {neuron_id}")
        if not MIN_SIM_FREQUENCY_HZ <= sim_frequency <= MAX_SIM_FREQUENCY_HZ:
            raise ValueError(
                f"sim_frequency must be between {MIN_SIM_FREQUENCY_HZ:g} and "
                f"{MAX_SIM_FREQUENCY_HZ:g} Hz"
            )

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
            did_spike = neuron.receive_impulse(impulse)
            if not did_spike:
                self._events.append(
                    f"Neuron {current_id}: Ladung {neuron.charge:.2f}, kein Spike."
                )
            else:
                spike_time_ms = neuron.last_spike_time
                relative_spike_time_ms = spike_time_ms - self._simulation_epoch_ms
                self._events.append(
                    f"Neuron {current_id} feuert bei {relative_spike_time_ms:.2f} ms."
                )
                for synapse in self._incoming[current_id]:
                    synapse.on_post_spike(spike_time_ms, self._events)
                for synapse in self._outgoing[current_id]:
                    synapse.on_pre_spike(spike_time_ms, self._events)
                    pending.append((synapse.target_id, synapse.weight * 1.5))

            elapsed_time = time.perf_counter() - tick_started
            remaining_time = target_frame_time - elapsed_time
            sleep_margin = min(0.0001, target_frame_time / 2)
            if remaining_time > sleep_margin:
                time.sleep(remaining_time - sleep_margin)
            while time.perf_counter() - tick_started < target_frame_time:
                pass

    def apply_reward(self) -> None:
        """Placeholder for reward-modulated LTP or firing-rate changes."""
        pass

    def apply_punishment(self) -> None:
        """Placeholder for punishment-modulated LTD or firing-rate changes."""
        pass

    def drain_events(self) -> List[str]:
        """Return and clear events generated since the previous drain."""
        events = list(self._events)
        self._events.clear()
        return events