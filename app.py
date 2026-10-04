"""Streamlit control panel for the persistent Bio-Tabula-Rasa SNN."""

import random

import streamlit as st

from brain import (
    DEFAULT_SIM_FREQUENCY_HZ,
    MAX_SIM_FREQUENCY_HZ,
    MIN_SIM_FREQUENCY_HZ,
    SpikingNeuralNetwork,
)


st.set_page_config(page_title="Bio-Tabula-Rasa", page_icon="🧠", layout="wide")
st.title("Bio-Tabula-Rasa")
st.caption("Ein digitales, durch Reize und Neuroplastizität lernendes Nervensystem.")

sim_frequency = st.sidebar.slider(
    "Simulationsfrequenz",
    min_value=MIN_SIM_FREQUENCY_HZ,
    max_value=MAX_SIM_FREQUENCY_HZ,
    value=DEFAULT_SIM_FREQUENCY_HZ,
    step=0.01,
    format="%.2f Hz",
    key="sim_frequency",
)
st.sidebar.caption(f"Tickdauer: {1.0 / sim_frequency:.3f} s")

if "brain" not in st.session_state:
    st.session_state.brain = SpikingNeuralNetwork(neuron_count=10)
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

brain = st.session_state.brain


def collect_brain_events() -> None:
    st.session_state.activity_log.extend(brain.drain_events())
    st.session_state.activity_log = st.session_state.activity_log[-200:]


left, right = st.columns(2)
with left:
    st.subheader("Neuronale Aktivität")
    chart_placeholder = st.empty()

with right:
    st.subheader("Textreiz & Feedback")
    with st.form("text_input_form"):
        text_input = st.text_input(
            "Text als neuronalen Reiz eingeben",
            placeholder="Wie heißt du?",
            max_chars=64,
        )
        text_submitted = st.form_submit_button(
            "Text in Spikes umwandeln", type="primary", use_container_width=True
        )

    feedback_columns = st.columns(2)
    reward_clicked = feedback_columns[0].button(
        "🟢 BELOHNEN", use_container_width=True
    )
    punishment_clicked = feedback_columns[1].button(
        "🔴 BESTRAFEN", use_container_width=True
    )
    random_stimulus_clicked = st.button(
        "Zufälligen Reiz einspeisen", use_container_width=True
    )
    reset_clicked = st.button("Gehirn zurücksetzen", use_container_width=True)
    progress_placeholder = st.empty()

st.subheader("Biologisches Aktivitäten-Protokoll")
log_placeholder = st.empty()


def render_chart() -> None:
    charges = {
        f"Neuron {neuron.neuron_id}": neuron.charge for neuron in brain.neurons
    }
    chart_placeholder.bar_chart(charges, y_label="Ladung", x_label="Neuron")


def render_log() -> None:
    if st.session_state.activity_log:
        log_placeholder.text("\n".join(reversed(st.session_state.activity_log[-30:])))
    else:
        log_placeholder.caption("Noch keine Aktivität.")


def refresh_dashboard() -> None:
    collect_brain_events()
    render_chart()
    render_log()


if reset_clicked:
    st.session_state.brain = SpikingNeuralNetwork(neuron_count=10)
    st.session_state.activity_log = ["Gehirn mit 10 Neuronen zurückgesetzt."]
    brain = st.session_state.brain
elif text_submitted and text_input:
    def update_text_progress(completed: int, total: int, character: str) -> None:
        progress_placeholder.progress(
            completed / total,
            text=f"Zeichen {completed}/{total}: {character!r}",
        )
        refresh_dashboard()

    brain.encode_text(
        text_input,
        sim_frequency=sim_frequency,
        input_neuron_id=0,
        progress_callback=update_text_progress,
    )
    progress_placeholder.empty()
elif reward_clicked:
    brain.apply_reward(sim_frequency=sim_frequency)
elif punishment_clicked:
    brain.apply_punishment(sim_frequency=sim_frequency)
elif random_stimulus_clicked:
    brain.stimulate(
        random.randrange(len(brain.neurons)), sim_frequency=sim_frequency
    )

refresh_dashboard()