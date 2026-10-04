"""Streamlit dashboard for the interactive biological SNN."""

import time

import streamlit as st

from brain import (
    DEFAULT_SIM_FREQUENCY_HZ,
    MAX_SIM_FREQUENCY_HZ,
    MIN_SIM_FREQUENCY_HZ,
    BiologicalBrain,
    get_brain_state_path,
)
from translator import spikes_to_output, telemetry_to_spikes, text_to_spikes

STATE_FILE = get_brain_state_path()

st.set_page_config(page_title="Bio-Tabula-Rasa", page_icon="🧠", layout="wide")
st.title("Bio-Tabula-Rasa")
st.caption(
    "Ein ereignisgesteuertes Spiking Neural Network, das durch Feedback lernt."
)

if "brain" not in st.session_state:
    st.session_state.brain = BiologicalBrain()
    if STATE_FILE.exists():
        st.session_state.brain.load_brain_state(str(STATE_FILE))
    else:
        st.session_state.brain.save_brain_state(str(STATE_FILE))
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "generated_output" not in st.session_state:
    st.session_state.generated_output = ""

brain: BiologicalBrain = st.session_state.brain
previous_ram_limit = brain.max_ram_limit_gb

sim_frequency = st.sidebar.slider(
    "Taktfrequenz",
    min_value=MIN_SIM_FREQUENCY_HZ,
    max_value=MAX_SIM_FREQUENCY_HZ,
    value=DEFAULT_SIM_FREQUENCY_HZ,
    step=0.01,
    format="%.2f Hz",
    key="sim_frequency",
)
if sim_frequency > 1000.0:
    st.sidebar.warning(
        "⚠️ Achtung: Du verlässt die biologische Taktgeschwindigkeit! Die Lernzeiten und Exponenten entsprechen ab jetzt nicht mehr der Realität. Der Super-Hirn-Modus ist aktiv."
    )
st.sidebar.caption("Habe viel Spaß mit dem Super-Hirn!")

max_ram_limit_gb = st.sidebar.slider(
    "Maximales RAM-Limit",
    min_value=1.0,
    max_value=16.0,
    value=brain.max_ram_limit_gb,
    step=0.1,
    format="%.1f GB",
    key="max_ram_limit_gb",
)
brain.set_ram_limit(max_ram_limit_gb)
decay_changed = brain.update_time()

live_column, plasticity_column = st.columns((1, 2))
with live_column:
    st.subheader("Live-Spikes")
    charge_chart = st.empty()

with plasticity_column:
    st.subheader("Synaptische Plastizität")
    synapse_table = st.empty()


def render_charge_chart() -> None:
    charge_data = [
        {"Neuron": f"Neuron {neuron.neuron_id}", "Ladung": neuron.charge}
        for neuron in brain.neurons
    ]
    charge_chart.bar_chart(
        charge_data, x="Neuron", y="Ladung", height=350, width="stretch"
    )


def render_synapse_table() -> None:
    synapse_data = [
        {
            "Quelle": edge.source.neuron_id,
            "Ziel": edge.target.neuron_id,
            "Gewicht": round(edge.weight, 4),
            "Letzte Aktivierung (ns)": edge.last_activation_time,
        }
        for edge in brain.synapses
    ]
    synapse_table.dataframe(
        synapse_data,
        width="stretch",
        hide_index=True,
        height=350,
    )


render_charge_chart()
render_synapse_table()

st.subheader("Interaktion")
with st.form("translator_form"):
    user_input = st.text_input(
        "Frage oder Codesignal an das Netzwerk",
        placeholder="Zum Beispiel: Wie heißt du?",
        max_chars=256,
    )
    text_submitted = st.form_submit_button(
        "Text als Spikes einspeisen", type="primary", width="stretch"
    )

st.subheader("🛰️ Telemetrie- & Hardware-Schnittstelle")
telemetry_columns = st.columns(2)
with telemetry_columns[0]:
    battery_level = st.slider(
        "🔋 Simulierte Akkuspannung (%)",
        min_value=0.0,
        max_value=100.0,
        value=100.0,
        step=1.0,
        key="telemetry_battery_level",
    )
with telemetry_columns[1]:
    sensor_distance = st.slider(
        "📏 Physischer Sensorabstand (cm)",
        min_value=0.0,
        max_value=200.0,
        value=200.0,
        step=1.0,
        key="telemetry_sensor_distance",
    )

telemetry_values = {
    "battery_level": battery_level,
    "distance_sensor_cm": sensor_distance,
}
telemetry_changed = (
    st.session_state.get("last_telemetry_values") != telemetry_values
)
if battery_level < 30.0:
    telemetry_status = "Zustand: Nahrungsknappheit (Stress-Spikes aktiv)"
elif sensor_distance < 20.0:
    telemetry_status = "Zustand: Physische Kollisionsgefahr!"
else:
    telemetry_status = "Zustand: Telemetrie stabil"
st.info(telemetry_status)
telemetry_rates = st.session_state.get("telemetry_rates", {})
if telemetry_rates:
    st.caption(
        "Einspeiseraten: "
        + " · ".join(
            f"{metric}: {rate:.1f} Hz" for metric, rate in telemetry_rates.items()
        )
    )

st.subheader("Generierter KI-Output")
output_placeholder = st.empty()

st.subheader("Konditionierung")
action_columns = st.columns(3)
with action_columns[0]:
    correct_clicked = st.button("🟢 KORREKT (Pfad verstärken)", width="stretch")
with action_columns[1]:
    incorrect_clicked = st.button("🔴 INKORREKT (Pfad schwächen)", width="stretch")
with action_columns[2]:
    therapy_clicked = st.button("💊 THERAPIE-MODUS", width="stretch")

state_changed = brain.max_ram_limit_gb != previous_ram_limit or decay_changed
if text_submitted and user_input:
    last_chart_update = [time.perf_counter()]

    def refresh_live_chart(completed: int, total: int) -> None:
        now = time.perf_counter()
        if completed == 1 or completed == total or now - last_chart_update[0] >= 0.1:
            render_charge_chart()
            last_chart_update[0] = now

    spike_count = text_to_spikes(
        user_input,
        brain,
        sim_frequency,
        progress_callback=refresh_live_chart,
    )
    st.session_state.generated_output = spikes_to_output(brain)
    st.session_state.activity_log.append(
        f"Textreiz abgeschlossen: {spike_count} Impulse für {len(user_input)} Zeichen."
    )
    brain.save_brain_state(str(STATE_FILE))
elif correct_clicked:
    changed = brain.strengthen_active_synapses()
    st.session_state.activity_log.append(
        f"Manuelles Lob: {changed} aktive Synapsen verstärkt."
    )
    brain.save_brain_state(str(STATE_FILE))
elif incorrect_clicked:
    changed = brain.weaken_active_synapses()
    st.session_state.activity_log.append(
        f"Manuelle Bestrafung: {changed} aktive Synapsen geschwächt."
    )
    brain.save_brain_state(str(STATE_FILE))
elif therapy_clicked:
    brain.run_therapy(sim_frequency)
    st.session_state.activity_log.append("Therapie-Modus abgeschlossen.")
    brain.save_brain_state(str(STATE_FILE))
elif telemetry_changed:
    telemetry_rates = telemetry_to_spikes(telemetry_values, brain)
    st.session_state.telemetry_rates = telemetry_rates
    st.session_state.last_telemetry_values = telemetry_values
    st.session_state.activity_log.append(
        "Telemetrie eingespeist: "
        + ", ".join(f"{metric} {rate:.1f} Hz" for metric, rate in telemetry_rates.items())
    )
    brain.save_brain_state(str(STATE_FILE))
elif state_changed:
    brain.save_brain_state(str(STATE_FILE))

st.session_state.activity_log.extend(brain.drain_events())
ram_usage_mb = brain.calculate_ram_usage()
st.sidebar.subheader("System-Metriken")
st.sidebar.metric("Aktuelle Frequenz", f"{sim_frequency:.2f} Hz")
st.sidebar.metric(
    "RAM-Verbrauch", f"{ram_usage_mb:.3f} MB / {ram_usage_mb * 1000:.1f} KB"
)
st.sidebar.metric("Stresslevel", f"{brain.stress_level:.1f}%")
st.sidebar.metric("Zustand", brain.health_status)

render_charge_chart()
render_synapse_table()
output_placeholder.text_area(
    "Ausgabe aus den Spike-Mustern der Output-Neuronen 8 und 9",
    value=st.session_state.generated_output,
    placeholder="Noch kein Output-Spike-Muster vorhanden.",
    height=120,
    disabled=True,
)
st.caption(
    "Ausgaberegel: Unter 10 Hz entstehen die Niedrigfrequenz-Tokens; "
    "ab 10 Hz die Hochfrequenz-Tokens. Sieben Output-Spikes können zusätzlich "
    "als 7-Bit-ASCII-Zeichen dekodiert werden."
)

with st.expander("System-Log", expanded=False):
    if st.session_state.activity_log:
        st.text("\n".join(reversed(st.session_state.activity_log)))
    else:
        st.caption("Noch keine Spike- oder Lernereignisse.")
