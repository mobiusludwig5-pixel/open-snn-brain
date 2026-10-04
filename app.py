"""Streamlit dashboard for the persistent spiking neural network."""

import random

import streamlit as st

from brain import SpikingNeuralNetwork


st.set_page_config(page_title="Bio-Tabula-Rasa", page_icon="🧠", layout="wide")
st.title("Bio-Tabula-Rasa")
st.caption("Ein digitales Gehirn, das durch Reize und STDP lernt.")

sim_frequency = st.sidebar.slider(
    "Simulationsfrequenz",
    min_value=1,
    max_value=1000,
    value=200,
    step=1,
    format="%d Hz",
    key="sim_frequency",
)

if "brain" not in st.session_state:
    st.session_state.brain = SpikingNeuralNetwork(neuron_count=10)
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

brain = st.session_state.brain


def collect_brain_events() -> None:
    st.session_state.activity_log.extend(brain.drain_events())
    st.session_state.activity_log = st.session_state.activity_log[-100:]


controls, visualization = st.columns([1, 2])

with controls:
    st.subheader("Steuerung")
    if st.button("Zufälligen Reiz einspeisen", type="primary", use_container_width=True):
        selected_neuron = random.randrange(len(brain.neurons))
        brain.stimulate(selected_neuron, sim_frequency=sim_frequency)
        collect_brain_events()

    feedback_columns = st.columns(2)
    if feedback_columns[0].button("🟢 BELOHNEN", use_container_width=True):
        brain.apply_reward()
        st.session_state.activity_log.append(
            "Feedback: Belohnung registriert (Lernregel derzeit vorbereitet)."
        )
    if feedback_columns[1].button("🔴 BESTRAFEN", use_container_width=True):
        brain.apply_punishment()
        st.session_state.activity_log.append(
            "Feedback: Bestrafung registriert (Lernregel derzeit vorbereitet)."
        )

    if st.button("Gehirn zurücksetzen", use_container_width=True):
        st.session_state.brain = SpikingNeuralNetwork(neuron_count=10)
        st.session_state.activity_log = ["Gehirn mit 10 neuen Neuronen initialisiert."]
        st.rerun()

with visualization:
    st.subheader("Neuronale Ladung")
    charges = {
        f"Neuron {neuron.neuron_id}": neuron.charge for neuron in brain.neurons
    }
    st.bar_chart(charges, y_label="Ladung", x_label="Neuron")
    st.caption(f"{len(brain.neurons)} Neuronen · {len(brain.synapses)} Synapsen")

st.subheader("Aktivitäten-Protokoll")
if st.session_state.activity_log:
    st.text("\n".join(reversed(st.session_state.activity_log)))
else:
    st.caption("Noch keine Aktivität.")