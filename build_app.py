"""Build Bio-Tabula-Rasa as a PyInstaller desktop application."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "Bio-Tabula-Rasa"
ASSET_FILES = ("app.py", "brain.py", "translator.py")
DATA_SEPARATOR = os.pathsep

LAUNCHER_SOURCE = r'''
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import webview
from streamlit.web import bootstrap


def _show_error(message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Bio-Tabula-Rasa konnte nicht starten", message)
        root.destroy()
    except Exception:
        logging.exception("Could not show the desktop error dialog")
        print(message, file=sys.stderr)


def _run_streamlit(app_file: Path, port: int, failures: List[Exception]) -> None:
    try:
        bootstrap.run(
            str(app_file),
            False,
            [],
            {
                "server.address": "127.0.0.1",
                "server.port": port,
                "server.headless": True,
                "server.fileWatcherType": "none",
                "browser.gatherUsageStats": False,
            },
        )
    except Exception as exc:
        failures.append(exc)
        logging.exception("Streamlit server stopped unexpectedly")


def _choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(port: int, failures: List[Exception]) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        if failures:
            raise RuntimeError("The embedded Streamlit server failed") from failures[0]
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    if failures:
        raise RuntimeError("The embedded Streamlit server failed") from failures[0]
    raise TimeoutError("The embedded Streamlit server did not become ready in time")


def main() -> None:
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    app_file = bundle_dir / "app.py"
    if not app_file.is_file():
        raise FileNotFoundError(f"Bundled Streamlit app not found: {app_file}")

    port = _choose_port()
    failures: List[Exception] = []
    server_thread = threading.Thread(
        target=_run_streamlit,
        args=(app_file, port, failures),
        name="bio-tabula-rasa-streamlit",
        daemon=True,
    )
    server_thread.start()
    _wait_for_server(port, failures)
    webview.create_window(
        "Bio-Tabula-Rasa Control Panel",
        f"http://127.0.0.1:{port}",
        min_size=(900, 650),
    )
    webview.start(debug=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception("Desktop application startup failed")
        _show_error(
            f"{exc}\n\n"
            "Prüfe, ob die WebView-Laufzeit des Betriebssystems installiert ist."
        )
        raise SystemExit(1) from exc
'''


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the native Bio-Tabula-Rasa desktop application."
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Create an application folder instead of a single-file executable.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Keep a console window open for troubleshooting.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear PyInstaller's build cache before packaging.",
    )
    parser.add_argument(
        "--name",
        default=APP_NAME,
        help=f"Set the executable name (default: {APP_NAME}).",
    )
    return parser.parse_args()


def _ensure_assets_exist() -> None:
    missing = [
        str(PROJECT_ROOT / filename)
        for filename in ASSET_FILES
        if not (PROJECT_ROOT / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required application files are missing:\n  " + "\n  ".join(missing)
        )


def _pyinstaller_dependencies() -> None:
    if sysconfig.get_config_var("Py_ENABLE_SHARED") != 1:
        raise RuntimeError(
            "The selected Python was built without a shared libpython library, "
            "which PyInstaller requires. Use a shared-library Python build "
            "(Py_ENABLE_SHARED=1) and rerun this script."
        )
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError(
            "PyInstaller is not installed. Install the build requirements with:\n"
            "  python -m pip install -r requirements-build.txt"
        )
    if importlib.util.find_spec("webview") is None:
        raise RuntimeError(
            "pywebview is not installed. Install the build requirements with:\n"
            "  python -m pip install -r requirements-build.txt"
        )


def _collect_package(
    package_name: str,
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    from PyInstaller.utils.hooks import collect_all, copy_metadata

    binaries, datas, hidden_imports = collect_all(package_name)
    datas.extend(copy_metadata("pywebview" if package_name == "webview" else package_name))
    return binaries, datas, hidden_imports


def _build_arguments(
    args: argparse.Namespace,
    launcher_path: Path,
    binaries: Iterable[Tuple[str, str]],
    datas: Iterable[Tuple[str, str]],
    hidden_imports: Iterable[str],
) -> List[str]:
    build_dir = PROJECT_ROOT / "build" / "pyinstaller"
    dist_dir = PROJECT_ROOT / "dist"
    arguments = [
        str(launcher_path),
        "--noconfirm",
        "--name",
        args.name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(build_dir),
        "--paths",
        str(PROJECT_ROOT),
        "--windowed" if not args.console else "--console",
        "--onefile" if not args.onedir else "--onedir",
    ]
    if args.clean:
        arguments.append("--clean")

    for filename in ASSET_FILES:
        arguments.extend(
            [
                "--add-data",
                f"{PROJECT_ROOT / filename}{DATA_SEPARATOR}.",
            ]
        )
    for source, destination in binaries:
        arguments.extend(["--add-binary", f"{source}{DATA_SEPARATOR}{destination}"])
    for source, destination in datas:
        arguments.extend(["--add-data", f"{source}{DATA_SEPARATOR}{destination}"])
    for module_name in sorted(set(hidden_imports)):
        arguments.extend(["--hidden-import", module_name])
    arguments.extend(
        [
            "--hidden-import",
            "brain",
            "--hidden-import",
            "translator",
            "--hidden-import",
            "streamlit.runtime.scriptrunner",
            "--hidden-import",
            "streamlit.runtime.scriptrunner_utils.script_run_context",
            "--hidden-import",
            "streamlit.web.bootstrap",
            "--hidden-import",
            "tkinter",
            "--hidden-import",
            "tkinter.messagebox",
        ]
    )
    return arguments


def main() -> int:
    args = _parse_args()
    _ensure_assets_exist()
    _pyinstaller_dependencies()

    from PyInstaller.__main__ import run as pyinstaller_run

    collected_binaries: List[Tuple[str, str]] = []
    collected_datas: List[Tuple[str, str]] = []
    collected_hidden_imports: List[str] = []
    for package_name in ("streamlit", "webview"):
        binaries, datas, hidden_imports = _collect_package(package_name)
        collected_binaries.extend(binaries)
        collected_datas.extend(datas)
        collected_hidden_imports.extend(hidden_imports)

    with tempfile.TemporaryDirectory(prefix="bio-tabula-rasa-") as temp_dir:
        launcher_path = Path(temp_dir) / "desktop_launcher.py"
        launcher_path.write_text(LAUNCHER_SOURCE, encoding="utf-8")
        pyinstaller_args = _build_arguments(
            args,
            launcher_path,
            collected_binaries,
            collected_datas,
            collected_hidden_imports,
        )
        pyinstaller_run(pyinstaller_args)

    output_path = PROJECT_ROOT / "dist" / args.name
    if sys.platform == "win32":
        output_path = output_path.with_suffix(".exe")
    print(f"Desktop build created under: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
