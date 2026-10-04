# bio-tabula-rasa

Ein ereignisgesteuertes Spiking Neural Network, das über benutzergesteuertes
Feedback und begrenzte STDP lernt.

## Starten

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

`brain.py` enthält die threadsichere SNN-Engine, `translator.py` die
zeitgetaktete Text-/Spike-Übersetzung und `app.py` das Streamlit-Dashboard.
Der Zustand wird atomar in `my_biological_brain.json` neben `app.py`
gespeichert und beim nächsten Start wieder geladen.
