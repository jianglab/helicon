"""A folder of the visitor's own computer, in a browser-side (Shinylive) copy.

There Python runs in the browser and cannot see the disk, so the "server"
input mode has nothing to list. This gives it something to list: a button asks
the browser for a folder (the File System Access API: Chrome and Edge), and
the folder is attached under :data:`ROOT` for Python to read like any other --
read-only, in place (a file is read when it is opened, not copied when the
folder is attached), and for as long as the page stays open.

The "server" input mode is then the local folder: :func:`helicon.shiny.source_modes`
offers it under the name :data:`LABEL`, :func:`helicon.shiny.file_picker_field`
puts the button above the path field, and :func:`helicon.shiny.file_picker_fill`
runs the dialog, confined to :data:`ROOT`, and attaches what is chosen. A tab
changes nothing but to ask :func:`helicon.webApps.deployment.has_server_files`
where it asked whether it ran on a hosting service.

The page keeps the folder's handle in IndexedDB for the worker Python runs in
to pick up: a handle cannot be sent as an input value.
"""

import logging
from pathlib import Path

from . import deployment

logger = logging.getLogger(__name__)

# What the "server" input mode is called to a visitor of a browser-side copy.
LABEL = "local folder"
# Where the folders chosen are attached, each under its own name.
ROOT = "/local"

_DB = "helicon-local-folders"

_PAGE_JS = (
    """
window.heliconChooseLocalFolder = async function (inputId) {
  if (!window.showDirectoryPicker) {
    alert("This browser cannot give a page access to a local folder. " +
          "Use Chrome or Edge, or give the file by its URL.");
    return;
  }
  let handle;
  try {
    handle = await window.showDirectoryPicker({mode: "read"});
  } catch (e) {
    return;  // the dialog was closed
  }
  const key = handle.name + "-" + Date.now();
  const db = await new Promise((resolve, reject) => {
    const r = indexedDB.open("%s", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("handles");
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
  await new Promise((resolve, reject) => {
    const t = db.transaction("handles", "readwrite");
    t.objectStore("handles").put(handle, key);
    t.oncomplete = resolve;
    t.onerror = () => reject(t.error);
  });
  db.close();
  Shiny.setInputValue(inputId, {key: key, name: handle.name}, {priority: "event"});
};
"""
    % _DB
)

# Run in the worker: every file of the folder becomes a File (a reference, no
# content read) in a WORKERFS mount, which reads a file's bytes when asked.
_ATTACH_JS = (
    """
(async (FS, key, mountpoint) => {
  const db = await new Promise((resolve, reject) => {
    const r = indexedDB.open("%s", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("handles");
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
  const handle = await new Promise((resolve, reject) => {
    const t = db.transaction("handles", "readwrite");
    const r = t.objectStore("handles").get(key);
    r.onsuccess = () => { t.objectStore("handles").delete(key); resolve(r.result); };
    r.onerror = () => reject(r.error);
  });
  db.close();
  if (!handle) throw new Error("the folder chosen was not found");
  const blobs = [];
  async function walk(dir, prefix) {
    for await (const [name, entry] of dir.entries()) {
      if (entry.kind === "file") {
        blobs.push({name: prefix + name, data: await entry.getFile()});
      } else {
        await walk(entry, prefix + name + "/");
      }
    }
  }
  await walk(handle, "");
  try { FS.unmount(mountpoint); } catch (e) {}
  FS.mkdirTree(mountpoint);
  FS.mount(FS.filesystems.WORKERFS, {blobs: blobs}, mountpoint);
  return blobs.length;
})
"""
    % _DB
)


def available() -> bool:
    """Whether a local folder can be offered: only in a browser-side copy.

    Returns
    -------
    bool
    """
    return deployment.is_browser()


def choose_button(id, label="Choose local folder"):
    """The button that asks the browser for a folder.

    Its input (``input[id]``) is set to ``{"key", "name"}`` each time a folder
    is chosen; :func:`local_folder_server` attaches it.

    Parameters
    ----------
    id : str
        The input id, within the calling module.
    label : str, optional

    Returns
    -------
    htmltools.TagList
    """
    from shiny import ui
    from shiny.module import resolve_id

    return ui.TagList(
        ui.tags.script(_PAGE_JS),
        ui.tags.button(
            label,
            type="button",
            class_="btn btn-default helicon-local-folder",
            onclick=f"heliconChooseLocalFolder('{resolve_id(id)}')",
            title="Let this page read a folder of your computer, e.g. a RELION "
            "or cryoSPARC project. Nothing is uploaded.",
            style="width: 100%; margin-bottom: 5px;",
        ),
    )


async def attach(key, name):
    """Attach the folder the page stored under ``key`` at ``ROOT/name``.

    Parameters
    ----------
    key : str
        The key the page stored the folder's handle under.
    name : str
        The folder's name.

    Returns
    -------
    tuple of (Path, int)
        Where the folder is, and how many files it has.
    """
    import pyodide_js
    from pyodide.code import run_js

    folder = Path(ROOT) / (Path(str(name)).name or "folder")
    n = await run_js(_ATTACH_JS)(pyodide_js.FS, str(key), str(folder))
    return folder, int(n)


def local_folder_server(input, id):
    """Attach each folder chosen with :func:`choose_button` of the same id.

    Call it in the server function of the module the button is in.

    Parameters
    ----------
    input : shiny.Inputs
        The calling module's inputs.
    id : str
        The button's input id.

    Returns
    -------
    reactive.Value
        The folder attached last (a Path), None before the first.
    """
    from shiny import reactive, ui

    folder = reactive.value(None)

    @reactive.effect
    @reactive.event(input[id])
    async def _attach_chosen_folder():
        chosen = input[id]()
        if not isinstance(chosen, dict) or not chosen.get("key"):
            return
        name = str(chosen.get("name") or "folder")
        note = f"helicon_local_folder_{id}"
        ui.notification_show(
            f"Reading the folder {name} …", id=note, duration=None, close_button=False
        )
        try:
            path, n = await attach(chosen["key"], name)
        except Exception as e:
            logger.warning("Could not attach the folder %s", name, exc_info=True)
            ui.notification_show(
                f"Could not read the folder {name}: {e}", type="warning", duration=10
            )
            return
        finally:
            ui.notification_remove(note)
        ui.notification_show(
            f"{name}: {n} file{'' if n == 1 else 's'}, at {path}. Browse for a "
            "file in it.",
            duration=8,
        )
        folder.set(path)

    return folder
