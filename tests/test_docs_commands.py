"""The reference of the command-line programs in the docs is the help the programs print."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestCommandReference:
    def test_the_generated_page_is_up_to_date(self):
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "docs" / "tools" / "gen_commands.py"),
                "--check",
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout + done.stderr[-500:]

    def test_every_program_of_the_command_line_is_in_the_reference(self):
        from helicon import helicon as cli

        pages = ROOT / "docs" / "commands"
        every = (
            cli.cli_commands
            + cli.napari_commands
            + cli.shiny_commands
            + cli.streamlit_commands
        )
        missing = [name for name in every if not (pages / f"{name}.md").exists()]
        assert not missing, missing
