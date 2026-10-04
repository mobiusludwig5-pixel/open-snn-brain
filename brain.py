"""Thread-safe, event-driven spiking neural network with bounded STDP."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

MIN_SIM_FREQUENCY_HZ = 0.01
MAX_SIM_FREQUENCY_HZ = 5000.0
DEFAULT_SIM_FREQUENCY_HZ = 200.0
STDP_WINDOW_NS = 50_000_000
ACTIVE_WINDOW_NS = 200_000_000
LEAK_TAU_NS = 50_000_000
MAX_SPIKE_HISTORY = 10_000


def get_brain_state_path() -> Path:
    """Return a persistent state path outside PyInstaller's temporary bundle."""
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().with_name("my_biological_brain.json")

    app_name = "Bio-Tabula-Rasa"
    if sys.platform == "win32":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        data_dir = Path(base_dir) / app_name if base_dir else Path.home() / app_name
    elif sys.platform == "darwin":
        data_dir = Path.home() / "Library" / "Application Support" / app_name
    else:
        state_home = os.environ.get("XDG_STATE_HOME")
        base_dir = Path(state_home) if state_home else Path.home() / ".local" / "state"
        data_dir = base_dir / "bio-tabula-rasa"
    return data_dir / "my_biological_brain.json"


def _require_timestamp(current_time_ns: int) -> None:
    if not isinstance(current_time_ns, int) or isinstance(current_time_ns, bool):
        raise TypeError("current_time_ns must be an integer number of nanoseconds")
    if current_time_ns < 0:
        raise ValueError("current_time_ns must not be negative")


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


class DigitalNeuron:
    """Leaky integrate-and-fire neuron with a bounded spike-time history."""

    def __init__(
        self,
        neuron_id: int,
        charge: float = 0.0,
        threshold: float = 1.0,
        spike_counter: int = 0,
    ) -> None:
        if neuron_id < 0:
            raise ValueError("neuron_id must not be negative")
        if not math.isfinite(charge):
            raise ValueError("charge must be finite")
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("threshold must be finite and greater than zero")
        if spike_counter < 0:
            raise ValueError("spike_counter must not be negative")

        self.neuron_id = neuron_id
        self.charge = charge
        self.threshold = threshold
        self.last_spike_time = 0
        self.last_impulse_time_ns = 0
        self.spike_counter = spike_counter
        self.spike_times_ns: Deque[int] = deque(maxlen=MAX_SPIKE_HISTORY)
        self._has_received_impulse = False
        self._lock = threading.RLock()

    def receive_impulse(self, force: float, current_time_ns: int) -> bool:
        """Leak charge since the prior impulse, integrate force, and possibly spike."""
        _require_timestamp(current_time_ns)
        _require_finite(force, "force")
        with self._lock:
            if self._has_received_impulse:
                elapsed_ns = current_time_ns - self.last_impulse_time_ns
                if elapsed_ns < 0:
                    raise ValueError("impulses must use nondecreasing timestamps")
                self.charge *= math.exp(-elapsed_ns / LEAK_TAU_NS)
            self.charge += force
            self.last_impulse_time_ns = current_time_ns
            self._has_received_impulse = True
            if self.charge < self.threshold:
                return False
            self.spike(current_time_ns)
            return True

    def spike(self, current_time_ns: Optional[int] = None) -> int:
        """Record a spike; an explicit timestamp makes simulations reproducible."""
        spike_time_ns = time.time_ns() if current_time_ns is None else current_time_ns
        _require_timestamp(spike_time_ns)
        with self._lock:
            self.spike_counter += 1
            self.charge = 0.0
            self.last_spike_time = spike_time_ns
            self.spike_times_ns.append(spike_time_ns)
            return spike_time_ns


class DigitalSynapse:
    """Directed connection with pairwise, bounded spike-timing plasticity."""

    def __init__(
        self,
        source: DigitalNeuron,
        target: DigitalNeuron,
        weight: float = 0.5,
    ) -> None:
        if source is target:
            raise ValueError("self-connections are not supported")
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0.0 and 1.0")
        self.source = source
        self.target = target
        self.weight = weight
        self.last_activation_time = 0
        self._last_stdp_pair: Optional[Tuple[int, int]] = None
        self._lock = threading.RLock()

    def apply_stdp(self, current_time_ns: int) -> float:
        """Update once per spike pair inside the 50 ms STDP window."""
        _require_timestamp(current_time_ns)
        with self._lock:
            first, second = sorted(
                (self.source, self.target), key=lambda neuron: neuron.neuron_id
            )
            with first._lock, second._lock:
                source_time = self.source.last_spike_time
                target_time = self.target.last_spike_time
                pair = (source_time, target_time)
                if self.source.spike_counter == 0 or self.target.spike_counter == 0:
                    return 0.0
                if pair == self._last_stdp_pair:
                    return 0.0

                self._last_stdp_pair = pair
                delta_ns = target_time - source_time
                if not 0 < abs(delta_ns) <= STDP_WINDOW_NS:
                    return 0.0

                change = 0.1 if delta_ns > 0 else -0.1
                previous_weight = self.weight
                self.weight = min(1.0, max(0.0, self.weight + change))
                self.last_activation_time = current_time_ns
                return self.weight - previous_weight


class BiologicalBrain:
    """A deterministic ten-neuron network with synchronized state and persistence."""

    def __init__(
        self,
        max_ram_limit_gb: float = 2.0,
        create_default_synapses: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.neurons = [DigitalNeuron(neuron_id=i) for i in range(10)]
        self.input_neuron_ids = tuple(range(0, 3))
        self.interneuron_ids = tuple(range(3, 8))
        self.output_neuron_ids = tuple(range(8, 10))
        self.synapses: List[DigitalSynapse] = []
        if create_default_synapses:
            connection_pairs = [
                (source_id, target_id)
                for source_id in self.input_neuron_ids
                for target_id in self.interneuron_ids
            ] + [
                (source_id, target_id)
                for source_id in self.interneuron_ids
                for target_id in self.output_neuron_ids
            ]
            self.synapses = [
                DigitalSynapse(self.neurons[source_id], self.neurons[target_id])
                for source_id, target_id in connection_pairs
            ]
        self._stress_level = 0.0
        self.health_status = "Stabil"
        self.telemetry_paths: Dict[str, int] = {}
        self._max_ram_limit_gb = 2.0
        self.set_ram_limit(max_ram_limit_gb)
        self._last_decay_time_ns = 0
        self._events: Deque[str] = deque()
        self._outgoing: Dict[int, List[DigitalSynapse]] = {}
        self._rebuild_outgoing()
        self._update_health_status()

    def _rebuild_outgoing(self) -> None:
        self._outgoing = {neuron.neuron_id: [] for neuron in self.neurons}
        for synapse in self.synapses:
            self._outgoing[synapse.source.neuron_id].append(synapse)

    def _record(self, message: str, current_time_ns: int) -> None:
        self._events.append(f"{current_time_ns} ns | {message}")

    def _update_health_status(self) -> None:
        if self.stress_level > 70.0:
            self.health_status = "⚠️ Depressiv / Apathisch"
        elif self.stress_level >= 30.0:
            self.health_status = "Gestresst"
        else:
            self.health_status = "Stabil"

    def set_ram_limit(self, limit_gb: float) -> None:
        _require_finite(limit_gb, "max_ram_limit_gb")
        if not 1.0 <= limit_gb <= 16.0:
            raise ValueError("max_ram_limit_gb must be between 1.0 and 16.0 GB")
        with self._lock:
            self._max_ram_limit_gb = float(limit_gb)

    @property
    def max_ram_limit_gb(self) -> float:
        return self._max_ram_limit_gb

    @max_ram_limit_gb.setter
    def max_ram_limit_gb(self, limit_gb: float) -> None:
        self.set_ram_limit(limit_gb)

    @property
    def stress_level(self) -> float:
        return self._stress_level

    @stress_level.setter
    def stress_level(self, level: float) -> None:
        _require_finite(level, "stress_level")
        if not 0.0 <= level <= 100.0:
            raise ValueError("stress_level must be between 0 and 100")
        with self._lock:
            self._stress_level = float(level)
            if hasattr(self, "health_status"):
                self._update_health_status()

    def register_telemetry_path(self, metric_name: str, neuron_id: int) -> None:
        """Remember which input neuron receives a physical telemetry metric."""
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if neuron_id not in (1, 2):
            raise ValueError("telemetry metrics must target input neuron 1 or 2")
        with self._lock:
            self.telemetry_paths[metric_name] = neuron_id

    def calculate_ram_usage(self) -> float:
        """Return the recursively counted object-graph size in megabytes."""
        with self._lock:
            pending: List[Any] = [self]
            seen: Set[int] = set()
            total_bytes = 0
            while pending:
                obj = pending.pop()
                identity = id(obj)
                if identity in seen:
                    continue
                seen.add(identity)
                total_bytes += sys.getsizeof(obj)
                if isinstance(obj, dict):
                    pending.extend(obj.keys())
                    pending.extend(obj.values())
                elif isinstance(obj, (list, tuple, set, frozenset, deque)):
                    pending.extend(obj)
                elif hasattr(obj, "__dict__") and not isinstance(
                    obj, (type, threading.Thread)
                ):
                    pending.append(vars(obj))
            return total_bytes / 1_000_000.0

    def check_memory_and_grow(self, source_id: int, target_id: int) -> bool:
        """Add a connection only when it is absent and the configured cap permits it."""
        with self._lock:
            self._validate_neuron_id(source_id)
            self._validate_neuron_id(target_id)
            if source_id == target_id:
                raise ValueError("source_id and target_id must differ")
            if any(
                edge.source.neuron_id == source_id
                and edge.target.neuron_id == target_id
                for edge in self.synapses
            ):
                return False
            if self.calculate_ram_usage() >= self.max_ram_limit_gb * 1000.0:
                self._record(
                    "Synaptogenese blockiert: RAM-Limit erreicht; Sparse Computing aktiv.",
                    time.time_ns(),
                )
                return False
            self.synapses.append(
                DigitalSynapse(self.neurons[source_id], self.neurons[target_id])
            )
            self._rebuild_outgoing()
            self._record(
                f"Synapse {source_id} -> {target_id} neu gebildet.", time.time_ns()
            )
            return True

    def _validate_neuron_id(self, neuron_id: int) -> None:
        if not isinstance(neuron_id, int) or not 0 <= neuron_id < len(self.neurons):
            raise IndexError(f"Unknown neuron: {neuron_id}")

    def _apply_apathy_decay(self, current_time_ns: int) -> bool:
        if self._last_decay_time_ns == 0:
            self._last_decay_time_ns = current_time_ns
            return False
        elapsed_ns = current_time_ns - self._last_decay_time_ns
        if elapsed_ns < 0:
            raise ValueError("simulation timestamps must be nondecreasing")
        self._last_decay_time_ns = current_time_ns
        if self.stress_level <= 70.0 or elapsed_ns == 0:
            return False
        decrement = 0.05 * (elapsed_ns / 1_000_000_000)
        changed = False
        for synapse in self.synapses:
            with synapse._lock:
                previous_weight = synapse.weight
                synapse.weight = max(0.0, synapse.weight - decrement)
                weight_changed = synapse.weight != previous_weight
                if weight_changed:
                    synapse.last_activation_time = current_time_ns
            if weight_changed:
                changed = True
                self._record(
                    f"Apathie: Synapse {synapse.source.neuron_id} -> "
                    f"{synapse.target.neuron_id} {previous_weight:.3f} -> "
                    f"{synapse.weight:.3f}.",
                    current_time_ns,
                )
        return changed

    def update_time(self, current_time_ns: Optional[int] = None) -> bool:
        """Apply time-dependent decay and report whether any weight changed."""
        if current_time_ns is not None:
            _require_timestamp(current_time_ns)
        with self._lock:
            now_ns = (
                max(time.time_ns(), self._last_decay_time_ns)
                if current_time_ns is None
                else current_time_ns
            )
            return self._apply_apathy_decay(now_ns)

    def inject_impulse(
        self, neuron_id: int, force: float, current_time_ns: Optional[int] = None
    ) -> List[int]:
        """Inject an impulse and propagate each resulting spike through the graph."""
        if current_time_ns is not None:
            _require_timestamp(current_time_ns)
        with self._lock:
            now_ns = (
                max(time.time_ns(), self._last_decay_time_ns)
                if current_time_ns is None
                else current_time_ns
            )
            self._validate_neuron_id(neuron_id)
            self._apply_apathy_decay(now_ns)
            fired_ids: List[int] = []
            pending: Deque[Tuple[int, float]] = deque([(neuron_id, force)])
            processed: Set[int] = set()
            while pending:
                current_id, impulse = pending.popleft()
                if current_id in processed:
                    continue
                neuron = self.neurons[current_id]
                fired = neuron.receive_impulse(impulse, now_ns)
                if not fired:
                    self._record(
                        f"Neuron {current_id}: Ladung {neuron.charge:.3f}, kein Spike.",
                        now_ns,
                    )
                    continue

                processed.add(current_id)
                fired_ids.append(current_id)
                self._record(
                    f"Spike Neuron {current_id} (Nr. {neuron.spike_counter}).", now_ns
                )
                related_synapses = [
                    edge
                    for edge in self.synapses
                    if edge.source is neuron or edge.target is neuron
                ]
                for edge in related_synapses:
                    change = edge.apply_stdp(now_ns)
                    if change:
                        self._record(
                            f"STDP {edge.source.neuron_id} -> {edge.target.neuron_id}: "
                            f"{edge.weight - change:.3f} -> {edge.weight:.3f}.",
                            now_ns,
                        )
                for edge in self._outgoing[current_id]:
                    pending.append((edge.target.neuron_id, edge.weight))
            return fired_ids

    def _active_synapses(self, current_time_ns: int) -> List[DigitalSynapse]:
        return [
            edge
            for edge in self.synapses
            if any(
                0 <= current_time_ns - neuron.last_spike_time <= ACTIVE_WINDOW_NS
                for neuron in (edge.source, edge.target)
                if neuron.spike_counter > 0
            )
        ]

    def strengthen_active_synapses(
        self, current_time_ns: Optional[int] = None
    ) -> int:
        if current_time_ns is not None:
            _require_timestamp(current_time_ns)
        with self._lock:
            now_ns = (
                max(time.time_ns(), self._last_decay_time_ns)
                if current_time_ns is None
                else current_time_ns
            )
            self._apply_apathy_decay(now_ns)
            changed = 0
            for edge in self._active_synapses(now_ns):
                with edge._lock:
                    previous = edge.weight
                    edge.weight = min(1.0, edge.weight + 0.1)
                    weight_changed = edge.weight != previous
                    if weight_changed:
                        edge.last_activation_time = now_ns
                if weight_changed:
                    changed += 1
                    self._record(
                        f"Lob/LTP Synapse {edge.source.neuron_id} -> "
                        f"{edge.target.neuron_id}: {previous:.3f} -> "
                        f"{edge.weight:.3f}.",
                        now_ns,
                    )
            return changed

    def weaken_active_synapses(self, current_time_ns: Optional[int] = None) -> int:
        if current_time_ns is not None:
            _require_timestamp(current_time_ns)
        with self._lock:
            now_ns = (
                max(time.time_ns(), self._last_decay_time_ns)
                if current_time_ns is None
                else current_time_ns
            )
            self._apply_apathy_decay(now_ns)
            changed = 0
            for edge in self._active_synapses(now_ns):
                with edge._lock:
                    previous = edge.weight
                    edge.weight = max(0.0, edge.weight - 0.1)
                    weight_changed = edge.weight != previous
                    if weight_changed:
                        edge.last_activation_time = now_ns
                if weight_changed:
                    changed += 1
                    self._record(
                        f"Bestrafung/LTD Synapse {edge.source.neuron_id} -> "
                        f"{edge.target.neuron_id}: {previous:.3f} -> "
                        f"{edge.weight:.3f}.",
                        now_ns,
                    )
            self.stress_level = min(100.0, self.stress_level + 10.0)
            self._update_health_status()
            self._record(
                f"Stresslevel auf {self.stress_level:.1f}% erhöht.", now_ns
            )
            return changed

    def run_therapy(self, sim_frequency: float, steps: int = 10) -> None:
        """Apply gentle sinusoidal input, reduce stress, and restore weak paths."""
        _require_finite(sim_frequency, "sim_frequency")
        if not MIN_SIM_FREQUENCY_HZ <= sim_frequency <= MAX_SIM_FREQUENCY_HZ:
            raise ValueError(
                f"sim_frequency must be between {MIN_SIM_FREQUENCY_HZ:g} and "
                f"{MAX_SIM_FREQUENCY_HZ:g} Hz"
            )
        if steps < 1:
            raise ValueError("steps must be at least one")
        with self._lock:
            start_ns = max(time.time_ns(), self._last_decay_time_ns)
            for step in range(steps):
                now_ns = max(time.time_ns(), start_ns, self._last_decay_time_ns)
                self._apply_apathy_decay(now_ns)
                for neuron in self.neurons:
                    force = 0.08 + 0.02 * (
                        1.0 + math.sin(2.0 * math.pi * step / steps)
                    )
                    self.inject_impulse(neuron.neuron_id, force, now_ns)
                self.stress_level = max(0.0, self.stress_level - 2.0)
            for edge in self.synapses:
                with edge._lock:
                    previous = edge.weight
                    if edge.weight < 0.2:
                        edge.weight = 0.2
                    weight_changed = edge.weight != previous
                    if weight_changed:
                        edge.last_activation_time = time.time_ns()
                if weight_changed:
                    self._record(
                        f"Therapie: Synapse {edge.source.neuron_id} -> "
                        f"{edge.target.neuron_id}: {previous:.3f} -> 0.200.",
                        time.time_ns(),
                    )
            self._update_health_status()
            self._record(
                f"Therapie abgeschlossen; Stresslevel {self.stress_level:.1f}%.",
                time.time_ns(),
            )

    def drain_events(self) -> List[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def save_brain_state(self, filepath: str = "my_biological_brain.json") -> None:
        """Atomically persist the public neural state as readable JSON."""
        path = Path(filepath)
        with self._lock:
            state = {
                "version": 1,
                "max_ram_limit_gb": self.max_ram_limit_gb,
                "stress_level": self.stress_level,
                "health_status": self.health_status,
                "last_decay_time_ns": self._last_decay_time_ns,
                "telemetry_paths": dict(self.telemetry_paths),
                "neurons": [
                    {
                        "neuron_id": neuron.neuron_id,
                        "charge": neuron.charge,
                        "threshold": neuron.threshold,
                        "last_spike_time": neuron.last_spike_time,
                        "last_impulse_time_ns": neuron.last_impulse_time_ns,
                        "has_received_impulse": neuron._has_received_impulse,
                        "spike_counter": neuron.spike_counter,
                        "spike_times_ns": list(neuron.spike_times_ns),
                    }
                    for neuron in self.neurons
                ],
                "synapses": [
                    {
                        "source_id": edge.source.neuron_id,
                        "target_id": edge.target.neuron_id,
                        "weight": edge.weight,
                        "last_activation_time": edge.last_activation_time,
                    }
                    for edge in self.synapses
                ],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = temporary_file.name
                    json.dump(state, temporary_file, ensure_ascii=False, indent=2)
                    temporary_file.write("\n")
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                os.replace(temporary_path, path)
            except OSError:
                if temporary_path is not None:
                    try:
                        os.unlink(temporary_path)
                    except FileNotFoundError:
                        pass
                raise

    def load_brain_state(self, filepath: str = "my_biological_brain.json") -> None:
        """Load and validate a saved state before changing the live network."""
        with Path(filepath).open("r", encoding="utf-8") as state_file:
            state = json.load(state_file)
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported or invalid brain-state file")
        try:
            neuron_states = state["neurons"]
            synapse_states = state["synapses"]
            stress_level = float(state["stress_level"])
            ram_limit = float(state["max_ram_limit_gb"])
            last_decay_time_ns = int(state["last_decay_time_ns"])
            telemetry_paths = state.get("telemetry_paths", {})
            if not isinstance(telemetry_paths, dict):
                raise ValueError("telemetry_paths must be a dictionary")
            parsed_telemetry_paths: Dict[str, int] = {}
            for metric_name, neuron_id in telemetry_paths.items():
                if (
                    not isinstance(metric_name, str)
                    or not metric_name.strip()
                    or not isinstance(neuron_id, int)
                    or isinstance(neuron_id, bool)
                    or neuron_id not in (1, 2)
                ):
                    raise ValueError("Invalid saved telemetry path")
                parsed_telemetry_paths[metric_name] = neuron_id
            if len(neuron_states) != 10:
                raise ValueError("Saved brain state must contain exactly 10 neurons")
            if not 0.0 <= stress_level <= 100.0:
                raise ValueError("stress_level must be between 0 and 100")
            if not 1.0 <= ram_limit <= 16.0:
                raise ValueError("max_ram_limit_gb must be between 1 and 16")
            _require_timestamp(last_decay_time_ns)

            parsed_neurons = []
            for expected_id, item in enumerate(neuron_states):
                if int(item["neuron_id"]) != expected_id:
                    raise ValueError("Neuron IDs in saved state must be 0 through 9")
                charge = float(item["charge"])
                threshold = float(item["threshold"])
                if not math.isfinite(charge) or not math.isfinite(threshold):
                    raise ValueError("Neuron values must be finite")
                if threshold <= 0:
                    raise ValueError("Neuron thresholds must be greater than zero")
                spike_times = [int(value) for value in item["spike_times_ns"]]
                if any(value < 0 for value in spike_times):
                    raise ValueError("Spike timestamps must not be negative")
                parsed_neurons.append(
                    {
                        "charge": charge,
                        "threshold": threshold,
                        "last_spike_time": int(item["last_spike_time"]),
                        "last_impulse_time_ns": int(item["last_impulse_time_ns"]),
                        "has_received_impulse": bool(
                            item.get("has_received_impulse", False)
                        ),
                        "spike_counter": int(item["spike_counter"]),
                        "spike_times_ns": spike_times,
                    }
                )

            parsed_synapses: List[Tuple[int, int, float, int]] = []
            for item in synapse_states:
                source_id = int(item["source_id"])
                target_id = int(item["target_id"])
                weight = float(item["weight"])
                activation_time = int(item["last_activation_time"])
                if (
                    not 0 <= source_id < 10
                    or not 0 <= target_id < 10
                    or source_id == target_id
                    or not math.isfinite(weight)
                    or not 0.0 <= weight <= 1.0
                ):
                    raise ValueError("Invalid saved synapse")
                _require_timestamp(activation_time)
                parsed_synapses.append(
                    (source_id, target_id, weight, activation_time)
                )
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("Malformed brain-state file") from exc

        with self._lock:
            for neuron, item in zip(self.neurons, parsed_neurons):
                neuron.charge = item["charge"]
                neuron.threshold = item["threshold"]
                neuron.last_spike_time = item["last_spike_time"]
                neuron.last_impulse_time_ns = item["last_impulse_time_ns"]
                neuron._has_received_impulse = item["has_received_impulse"]
                neuron.spike_counter = item["spike_counter"]
                neuron.spike_times_ns.clear()
                neuron.spike_times_ns.extend(item["spike_times_ns"])
            self.synapses = [
                DigitalSynapse(self.neurons[source], self.neurons[target], weight)
                for source, target, weight, _ in parsed_synapses
            ]
            for edge, (_, _, _, activation_time) in zip(
                self.synapses, parsed_synapses
            ):
                edge.last_activation_time = activation_time
            self.stress_level = stress_level
            self.max_ram_limit_gb = ram_limit
            self._last_decay_time_ns = last_decay_time_ns
            self.telemetry_paths = parsed_telemetry_paths
            self._events.clear()
            self._rebuild_outgoing()
            self._update_health_status()
