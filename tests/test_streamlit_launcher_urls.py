import argparse
import re

from helicon.commands import ctfSimulation, procart

LAUNCHERS = [ctfSimulation, procart]

# a raw GitHub URL pinned to a 40-hex commit, not to a moving branch ref
PINNED = re.compile(
    r"https://raw\.githubusercontent\.com/jianglab/[^/]+/[0-9a-f]{40}/\S+\.py"
)


class TestTheRemoteScriptIsPinned:
    def test_the_url_is_pinned_to_a_commit(self):
        for mod in LAUNCHERS:
            assert PINNED.fullmatch(mod.URL), mod.URL

    def test_the_url_shown_is_the_one_streamlit_runs(self, monkeypatch, capsys):
        for mod in LAUNCHERS:
            ran = []
            monkeypatch.setattr("subprocess.call", lambda cmd: ran.append(cmd) or 0)
            mod.main(argparse.Namespace())
            assert capsys.readouterr().out.strip() == mod.URL
            assert mod.URL in ran[0]
