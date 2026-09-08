"""The launcher is intentionally testable without Qt or cryo-EM dependencies.

Run directly with ``python tests/test_relion_launcher.py`` or with pytest.
"""

import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "relion_launcher",
    Path(__file__).resolve().parents[1] / "src/helicon/lib/relion_launcher.py",
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class RelionLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project with spaces & quotes'"
        self.project.mkdir()
        (self.project / "default_pipeline.star").touch()

    def test_project_requires_file_in_exact_folder(self):
        self.assertTrue(launcher.is_relion_project(self.project))
        child = self.project / "Class2D"
        child.mkdir()
        self.assertFalse(launcher.is_relion_project(child))
        marker = self.project / "default_pipeline.star"
        marker.unlink()
        marker.mkdir()
        self.assertFalse(launcher.is_relion_project(self.project))

    def test_isolation_removes_software_state_preserves_display(self):
        original = {
            "HOME": "/home/test", "DISPLAY": "localhost:10.0",
            "XAUTHORITY": "/tmp/auth", "SSH_AUTH_SOCK": "/tmp/agent",
            "CUDA_VISIBLE_DEVICES": "2", "LC_ALL": "C",
            "PATH": "/helicon/bin", "CONDA_PREFIX": "/helicon",
            "PYTHONPATH": "/helicon/python", "LD_LIBRARY_PATH": "/helicon/lib",
            "LD_PRELOAD": "/helicon/preload.so", "LOADEDMODULES": "helicon",
            "BASH_ENV": "/helicon/init.sh", "QT_PLUGIN_PATH": "/helicon/qt",
            "OMPI_MCA_prefix": "/helicon/mpi", "RELION_QSUB_TEMPLATE": "/old",
            "BASH_FUNC_module%%": "() { eval something; }",
        }
        env = launcher.relion_environment(original, isolated=True)
        for key in ("HOME", "DISPLAY", "XAUTHORITY", "SSH_AUTH_SOCK",
                    "CUDA_VISIBLE_DEVICES", "LC_ALL"):
            self.assertEqual(env[key], original[key])
        self.assertEqual(set(env), {
            "HOME", "DISPLAY", "XAUTHORITY", "SSH_AUTH_SOCK",
            "CUDA_VISIBLE_DEVICES", "LC_ALL", "PATH",
        })
        self.assertNotIn("helicon", env["PATH"])
        self.assertEqual(original["PATH"], "/helicon/bin")
        self.assertEqual(launcher.relion_environment(original, isolated=False), original)
        self.assertIsNot(launcher.relion_environment(original, isolated=False), original)

    def posix_os(self):
        return patch.object(launcher, "os", SimpleNamespace(
            name="posix", environ={"PATH": "/helicon/bin", "DISPLAY": ":1"},
            defpath=os.defpath, fdopen=os.fdopen,
        ))

    def test_script_is_single_argument_and_process_is_detached(self):
        script = Path(self.temp.name) / "launch with spaces & quote'.sh"
        script.write_text("exec relion\n")
        with self.posix_os(), patch.object(launcher.shutil, "which", return_value="/bin/bash"), \
                patch.object(launcher.subprocess, "Popen") as popen:
            process, log = launcher.launch_relion(self.project, str(script))
        self.addCleanup(log.unlink)
        self.assertIs(process, popen.return_value)
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["/bin/bash", "--noprofile", "--norc", str(script)])
        self.assertEqual(kwargs["cwd"], self.project.resolve())
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["close_fds"])
        self.assertNotIn("shell", kwargs)
        self.assertNotIn("helicon", kwargs["env"]["PATH"])
        self.assertEqual(kwargs["env"]["DISPLAY"], ":1")
        self.assertEqual(kwargs["stderr"], launcher.subprocess.STDOUT)
        self.assertTrue(log.is_file())

    def test_default_uses_current_path(self):
        with self.posix_os(), patch.object(launcher.shutil, "which", return_value="/helicon/bin/relion") as which, \
                patch.object(launcher.subprocess, "Popen") as popen:
            _, log = launcher.launch_relion(self.project)
        self.addCleanup(log.unlink)
        which.assert_called_once_with("relion", path="/helicon/bin")
        self.assertEqual(popen.call_args.args[0], ["/helicon/bin/relion"])
        self.assertEqual(popen.call_args.kwargs["env"]["PATH"], "/helicon/bin")

    def test_missing_relion_gives_configuration_guidance(self):
        with self.posix_os(), patch.object(launcher.shutil, "which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "Configure RELION"):
                launcher.launch_relion(self.project)

    def test_invalid_project_and_script_do_not_spawn(self):
        with self.posix_os(), patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "default_pipeline.star"):
                launcher.launch_relion(self.project.parent)
            with self.assertRaisesRegex(ValueError, "absolute path"):
                launcher.launch_relion(self.project, "relative.sh")
            popen.assert_not_called()

    def test_native_windows_reports_wsl_requirement(self):
        with patch.object(launcher, "os", SimpleNamespace(name="nt")):
            with self.assertRaisesRegex(OSError, "Linux/WSL"):
                launcher.launch_relion(self.project)

    def test_spawn_failure_cleans_up_log(self):
        log = Path(self.temp.name) / "failed.log"
        fd = os.open(log, os.O_WRONLY | os.O_CREAT)
        with self.posix_os(), patch.object(launcher.shutil, "which", return_value="/usr/bin/relion"), \
                patch.object(launcher.tempfile, "mkstemp", return_value=(fd, str(log))), \
                patch.object(launcher.subprocess, "Popen", side_effect=OSError("failed")):
            with self.assertRaisesRegex(OSError, "failed"):
                launcher.launch_relion(self.project)
        self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
