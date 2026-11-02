"""Starting the app with one command, or by double-clicking a file."""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from humanmock import cli
from humanmock.checkpoint import structure_fingerprint

ROOT = Path(__file__).parent.parent


def write_model(folder: Path, structure: str, modified: int) -> Path:
    folder.mkdir(parents=True)
    meta = folder / "meta.json"
    meta.write_text(json.dumps({"structure": structure}), encoding="utf8")
    os.utime(meta, (modified, modified))
    return folder


def test_it_uses_the_newest_model_that_fits_this_body(tmp_path):
    current = structure_fingerprint()
    write_model(tmp_path / "older", current, 1000)
    newer = write_model(tmp_path / "newer", current, 2000)
    write_model(tmp_path / "newest_but_for_another_body", "something-else", 3000)
    assert cli.newest_checkpoint(tmp_path) == newer


def test_it_copes_with_no_models_and_broken_ones(tmp_path):
    assert cli.newest_checkpoint(tmp_path) is None
    assert cli.newest_checkpoint(tmp_path / "missing") is None
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "meta.json").write_text("{not json", encoding="utf8")
    assert cli.newest_checkpoint(tmp_path) is None


def test_it_moves_to_another_port_when_the_usual_one_is_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        busy = taken.getsockname()[1]
        chosen = cli.free_port(busy)
        assert chosen != busy and chosen > 0
    assert cli.free_port(busy) == busy


def test_the_browser_opens_only_once_the_server_answers(monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)

    class Healthy(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    cli.open_when_ready(url, attempts=1)
    assert opened == []

    server = HTTPServer(("127.0.0.1", port), Healthy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cli.open_when_ready(url, attempts=5)
    finally:
        server.shutdown()
        server.server_close()
    assert opened == [url]


@pytest.mark.parametrize("use_browser", [True, False])
def test_start_finds_a_model_and_runs_the_server(tmp_path, monkeypatch, use_browser):
    model = write_model(tmp_path / "models" / "run1", structure_fingerprint(), 1000)
    ran, threads = {}, []
    monkeypatch.setattr(cli, "_run_server", lambda checkpoint, host, port, database: ran.update(checkpoint=checkpoint, host=host, port=port))
    monkeypatch.setattr(cli, "open_when_ready", lambda url: threads.append(url))
    arguments = ["humanmock", "start", "--checkpoints", str(tmp_path / "models"), "--port", "0"]
    if not use_browser:
        arguments.append("--no-browser")
    monkeypatch.setattr("sys.argv", arguments)
    cli.main()
    assert ran["checkpoint"] == str(model)
    assert ran["host"] == "127.0.0.1", "The app must only be reachable from this computer by default."
    assert bool(threads) is use_browser


def test_the_double_click_files_set_up_and_then_start():
    windows = (ROOT / "start.bat").read_text(encoding="utf8")
    unix = (ROOT / "start.sh").read_text(encoding="utf8")
    for script in (windows, unix):
        assert "(3, 11)" in script, "They must check the Python version, and explain if it is too old."
        assert "-e ." in script and "start" in script


def test_the_scripts_keep_the_line_endings_their_systems_need():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf8")
    assert "*.bat text eol=crlf" in attributes
