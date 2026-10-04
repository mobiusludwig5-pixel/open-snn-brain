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
Der Zustand wird atomar in `my_biological_brain.json` gespeichert und beim
nächsten Start wieder geladen. In einem Desktop-Build liegt die Datei dauerhaft
im Benutzerprofil (`%LOCALAPPDATA%/Bio-Tabula-Rasa` unter Windows,
`~/Library/Application Support/Bio-Tabula-Rasa` unter macOS und
`$XDG_STATE_HOME/bio-tabula-rasa` bzw. `~/.local/state/bio-tabula-rasa` unter
Linux), nicht im temporären PyInstaller-Verzeichnis.

## Native Desktop-App bauen

PyInstaller kann keine Builds für andere Betriebssysteme erzeugen. Baue die App
auf dem jeweiligen Zielsystem. Installiere einmalig die Build-Abhängigkeiten und
starte anschließend:

```bash
python -m pip install -r requirements-build.txt
python build_app.py
```

Das Skript erstellt standardmäßig ein einzelnes GUI-Binary in `dist/`
(`.exe` unter Windows). Mit `python build_app.py --onedir` wird stattdessen ein
Anwendungsordner erstellt; `--console` lässt für die Fehlersuche ein Terminal
sichtbar. Das Programmfenster benötigt die native WebView-Laufzeit des Systems
(WebView2 unter Windows; WebKitGTK unter Linux; WKWebView unter macOS). Unter
Windows ist gegebenenfalls die Microsoft Edge WebView2 Runtime nachzuinstallieren.
Die Python-Installation, mit der gebaut wird, muss eine Shared Library
(`libpython`/`Py_ENABLE_SHARED=1`) bereitstellen; statische Python-Builds kann
PyInstaller nicht verwenden.
