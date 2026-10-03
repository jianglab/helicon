"""A file picker for Shiny apps that browses the server's file system.

For an app that runs where the data are (``helicon webApps`` on the user's own
computer or a lab server): a "Browse..." button beside a path field opens a
dialog that lists the server's folders and files, and the file chosen there
comes back as a path for the field.

Usage, inside a module or an app::

    # UI, next to the field
    ui.input_text("url_params", "Class2D parameter file", value=...)
    helicon.shiny.file_picker_button("params_browse")

    # server
    chosen = helicon.shiny.file_picker_server(
        "params_browse",
        patterns=("*.star", "*.cs"),
        title="Select the Class2D parameter file",
        start=lambda: input.url_params(),
    )

    @reactive.effect
    @reactive.event(chosen)
    def _():
        ui.update_text("url_params", value=chosen())

The dialog has a clickable path (and a box to type or paste one), Up, Home and
working-folder buttons, chips for recently used folders, a filter that narrows
the list as you type, a choice between the field's file types and all files,
and folders and files listed with their size and modification time. A click
selects, a double click opens a folder or chooses a file; the arrow keys,
Enter and Backspace (up a folder) work as in a desktop file dialog.
"""

from __future__ import annotations

import fnmatch
import html
import itertools
import os
import re
import time
from pathlib import Path

from shiny import module, reactive, render, ui

__all__ = [
    "SERVER",
    "source_modes",
    "file_picker_button",
    "file_picker_fill",
    "file_picker_field",
    "file_picker_server",
]

# Folders used most recently, newest first, are shared by every picker of one
# browser session (a user tends to go back to the same few folders) but not
# between sessions: on a lab server several people use the same app.
_MAX_RECENT = 6
_RECENT_ATTR = "_helicon_recent_folders"

_FOLDER_SVG = (
    '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">'
    '<path fill="#e3a72f" d="M1.5 3A1.5 1.5 0 0 1 3 1.5h3.2l1.6 1.6H13A1.5 1.5 0 '
    '0 1 14.5 4.6v7.9A1.5 1.5 0 0 1 13 14H3a1.5 1.5 0 0 1-1.5-1.5z"/></svg>'
)
_FILE_SVG = (
    '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">'
    '<path fill="none" stroke="currentColor" stroke-opacity=".55" '
    'd="M3.5 1.5h6l3 3v10h-9z M9.5 1.5v3h3"/></svg>'
)
_UP_SVG = (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">'
    '<path fill="none" stroke="currentColor" stroke-width="1.6" '
    'd="M8 13V3M3.5 7.5 8 3l4.5 4.5"/></svg>'
)
_HOME_SVG = (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">'
    '<path fill="none" stroke="currentColor" stroke-width="1.4" '
    'd="M2.5 7.5 8 2.5l5.5 5M4 6.5v7h3v-4h2v4h3v-7"/></svg>'
)
_WORK_SVG = (
    '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">'
    '<path fill="none" stroke="currentColor" stroke-width="1.4" '
    'd="M2 4.5h4l1.2 1.2H14v7.8H2zM2 4.5V3h4"/><circle cx="8" cy="9.5" r="1.6" '
    'fill="currentColor"/></svg>'
)

_CSS = """
.hfp { --hfp-border: var(--bs-border-color, #dee2e6);
       --hfp-muted: var(--bs-secondary-color, #6c757d);
       --hfp-hover: var(--bs-tertiary-bg, rgba(0,0,0,.04));
       --hfp-sel: var(--bs-primary-bg-subtle, #cfe2ff);
       --hfp-accent: var(--bs-primary, #0d6efd);
       font-size: .9rem; }
.hfp-bar { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; }
.hfp-btn { display: inline-flex; align-items: center; gap: 4px; padding: 3px 8px;
           border: 1px solid var(--hfp-border); border-radius: 6px;
           background: transparent; color: inherit; line-height: 1.2; white-space: nowrap; }
.hfp-btn:hover { background: var(--hfp-hover); }
.hfp-btn:disabled { opacity: .45; }
.hfp-crumbs { flex: 1; min-width: 0; display: flex; flex-wrap: nowrap; overflow-x: auto;
              align-items: center; gap: 1px; padding: 3px 6px; border: 1px solid var(--hfp-border);
              border-radius: 6px; scrollbar-width: thin; }
.hfp-crumb { border: 0; background: none; color: inherit; padding: 1px 4px; border-radius: 4px;
             white-space: nowrap; }
.hfp-crumb:hover { background: var(--hfp-hover); }
.hfp-crumb:last-child { font-weight: 600; }
.hfp-sep { color: var(--hfp-muted); }
.hfp-path { width: 100%; margin-bottom: 6px; font-family: var(--bs-font-monospace, monospace);
            font-size: .85rem; }
.hfp-places { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; align-items: center; }
.hfp-places .hfp-label { color: var(--hfp-muted); margin-right: 2px; }
.hfp-chip { border: 1px solid var(--hfp-border); border-radius: 999px; padding: 1px 9px;
            background: transparent; color: inherit; font-size: .82rem; max-width: 22em;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hfp-chip:hover { background: var(--hfp-hover); }
.hfp-tools { display: flex; gap: 12px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
.hfp-filter { flex: 1; min-width: 12em; }
.hfp-tools .form-group, .hfp-tools .shiny-input-container { margin: 0 !important; width: auto !important; }
.hfp-tools .checkbox { margin: 0; }
.hfp-list { height: clamp(200px, calc(100vh - 560px), 560px); overflow: auto; border: 1px solid var(--hfp-border); border-radius: 6px;
            outline: none; }
.hfp-list:focus-visible { box-shadow: 0 0 0 2px var(--hfp-sel); }
.hfp-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
.hfp-table th { position: sticky; top: 0; z-index: 1; background: var(--bs-body-bg, #fff);
                font-weight: 500; color: var(--hfp-muted); text-align: left; padding: 4px 8px;
                border-bottom: 1px solid var(--hfp-border); font-size: .8rem; }
.hfp-table td { padding: 3px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                cursor: default; user-select: none; }
.hfp-table .hfp-icon { width: 30px; padding-right: 0; }
.hfp-table td.hfp-size, .hfp-table th.hfp-size { width: 6em; text-align: right; }
.hfp-table td.hfp-time, .hfp-table th.hfp-time { width: 10.5em; }
.hfp-table td.hfp-size, .hfp-table td.hfp-time { color: var(--hfp-muted); font-variant-numeric: tabular-nums; }
.hfp-table th.hfp-sort { cursor: pointer; user-select: none; white-space: nowrap; }
.hfp-table th.hfp-sort:hover { color: var(--bs-body-color, #212529); }
.hfp-table th.hfp-sort[aria-sort="ascending"]::after { content: " \\25B2"; font-size: .7em; }
.hfp-table th.hfp-sort[aria-sort="descending"]::after { content: " \\25BC"; font-size: .7em; }
.hfp-row:hover td { background: var(--hfp-hover); }
.hfp-row.hfp-on td { background: var(--hfp-sel); }
.hfp-row.hfp-on td:first-child { box-shadow: inset 3px 0 0 var(--hfp-accent); }
.hfp-row[hidden] { display: none; }
.hfp-note { color: var(--hfp-muted); padding: 10px 12px; }
.hfp-error { color: var(--bs-danger, #dc3545); padding: 10px 12px; }
.hfp-foot { display: flex; align-items: center; gap: 8px; width: 100%; }
.hfp-chosen { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
              color: var(--hfp-muted); font-family: var(--bs-font-monospace, monospace); font-size: .82rem;
              direction: rtl; text-align: left; }
/* A path field laid out like Shiny's own file input: the label above, then the
   Browse button joined on the left of the box. The field's container is
   flattened so its label and box take places in this grid. */
.hfp-field { display: grid; grid-template-columns: auto minmax(0, 1fr);
             grid-template-areas: "lab lab" "btn box"; margin-bottom: 1rem; width: 100%; }
.hfp-field > .shiny-input-container { display: contents; }
.hfp-field > .shiny-input-container > label { grid-area: lab; }
.hfp-field > .shiny-input-container > input { grid-area: box; min-width: 0;
             border-top-left-radius: 0; border-bottom-left-radius: 0; }
.hfp-field .hfp-browse { grid-area: btn; margin: 0; white-space: nowrap;
             border-top-right-radius: 0; border-bottom-right-radius: 0; }
"""

# The folders files were picked from, kept in the browser (localStorage) so a
# new visit starts where the last one ended. They reach the server as one
# page-level input, which every picker reads; the server keeps nothing of
# them between sessions.
_BROWSER_RECENT_INPUT = "helicon_file_picker_recent"
_MEMORY_JS = """
(function () {
  if (window.__heliconPickerMemory) return;
  window.__heliconPickerMemory = true;
  var KEY = 'heliconFilePicker.recent', MAX = %d;
  function load() {
    try {
      var v = JSON.parse(localStorage.getItem(KEY) || '[]');
      return Array.isArray(v) ? v.filter(function (x) { return typeof x === 'string'; }).slice(0, MAX) : [];
    } catch (e) { return []; }
  }
  function send() {
    if (window.Shiny && Shiny.setInputValue) Shiny.setInputValue('%s', load());
  }
  window.__heliconPickerRemember = function (folder) {
    if (!folder) return;
    var v = load().filter(function (x) { return x !== folder; });
    v.unshift(folder);
    try { localStorage.setItem(KEY, JSON.stringify(v.slice(0, MAX))); } catch (e) {}
    send();
  };
  if (window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected && Shiny.shinyapp.isConnected()) send();
  else $(document).one('shiny:connected', send);
})();
"""

# One script for every picker on the page: it works by delegation, so it does
# not matter when a picker's dialog is drawn.
_JS = """
(function () {
  if (window.__heliconFilePicker) return;
  window.__heliconFilePicker = true;
  function root(el) { return el && el.closest ? el.closest('.hfp') : null; }
  function send(r, msg) {
    msg.n = Date.now() + Math.random();
    Shiny.setInputValue(r.dataset.act, msg, {priority: 'event'});
  }
  function rows(r) { return Array.from(r.querySelectorAll('.hfp-row')).filter(function (x) { return !x.hidden; }); }
  function current(r) { return r.querySelector('.hfp-row.hfp-on'); }
  function chooseButton(r) {
    var modal = r.closest('.modal');
    return modal ? modal.querySelector('.hfp-choose') : null;
  }
  function select(r, row, scroll) {
    r.querySelectorAll('.hfp-row.hfp-on').forEach(function (x) { x.classList.remove('hfp-on'); });
    var btn = chooseButton(r), shown = r.closest('.modal') && r.closest('.modal').querySelector('.hfp-chosen');
    if (!row) { if (btn) btn.disabled = true; if (shown) shown.textContent = ''; return; }
    row.classList.add('hfp-on');
    if (scroll) row.scrollIntoView({block: 'nearest'});
    var isFile = row.dataset.kind === 'file';
    if (btn) btn.disabled = !isFile;
    if (shown) shown.textContent = isFile ? (r.dataset.cwd.replace(/\\/$/, '') + '/' + row.dataset.name) : '';
  }
  function activate(r, row) {
    if (!row) return;
    if (row.dataset.kind === 'dir') send(r, {op: 'open', name: row.dataset.name});
    else {
      if (window.__heliconPickerRemember) window.__heliconPickerRemember(r.dataset.cwd);
      send(r, {op: 'choose', name: row.dataset.name});
    }
  }
  // Sorting by a column, in the browser: '..' stays on top and folders before
  // files, as file managers do. The choice is kept for the next listing, and
  // in the browser for the next visit.
  var SORT_KEY = 'heliconFilePicker.sort';
  function sortState() {
    try {
      var v = JSON.parse(localStorage.getItem(SORT_KEY) || 'null');
      if (v && ['name', 'size', 'mtime'].indexOf(v.key) >= 0) return v;
    } catch (e) {}
    return {key: 'name', dir: 1};
  }
  function applySort(r) {
    var body = r.querySelector('.hfp-table tbody');
    if (!body) return;
    var st = sortState(), names = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
    var all = Array.from(body.querySelectorAll('.hfp-row'));
    var up = all.filter(function (x) { return x.dataset.name === '..'; });
    var rest = all.filter(function (x) { return x.dataset.name !== '..'; });
    function num(x, k) { var v = parseFloat(x.dataset[k]); return isNaN(v) ? -Infinity : v; }
    rest.sort(function (a, b) {
      if (a.dataset.kind !== b.dataset.kind) return a.dataset.kind === 'dir' ? -1 : 1;
      var c = 0;
      if (st.key !== 'name') c = num(a, st.key) - num(b, st.key);
      if (c === 0) c = names.compare(a.dataset.name, b.dataset.name);
      return st.dir * c;
    });
    up.concat(rest).forEach(function (x) { body.appendChild(x); });
    r.querySelectorAll('th.hfp-sort').forEach(function (th) {
      th.setAttribute('aria-sort', th.dataset.sort === st.key ? (st.dir > 0 ? 'ascending' : 'descending') : 'none');
    });
  }
  document.addEventListener('click', function (e) {
    var r = root(e.target);
    if (r) {
      var th = e.target.closest('th.hfp-sort');
      if (th) {
        var st = sortState(), key = th.dataset.sort;
        // the same column again turns the order around; a new one starts with
        // names A-Z, and sizes and times largest and newest first
        st = st.key === key ? {key: key, dir: -st.dir} : {key: key, dir: key === 'name' ? 1 : -1};
        try { localStorage.setItem(SORT_KEY, JSON.stringify(st)); } catch (err) {}
        var on = current(r);
        applySort(r);
        if (on) on.scrollIntoView({block: 'nearest'});
        return;
      }
      var row = e.target.closest('.hfp-row');
      if (row) { select(r, row, false); return; }
      var b = e.target.closest('[data-op]');
      if (b) { send(r, {op: b.dataset.op, path: b.dataset.path || ''}); return; }
    }
    var choose = e.target.closest && e.target.closest('.hfp-choose');
    if (choose) {
      var modal = choose.closest('.modal'), rr = modal && modal.querySelector('.hfp');
      if (rr) activate(rr, current(rr));
    }
  });
  document.addEventListener('dblclick', function (e) {
    var r = root(e.target), row = e.target.closest && e.target.closest('.hfp-row');
    if (r && row) { select(r, row, false); activate(r, row); }
  });
  document.addEventListener('keydown', function (e) {
    var r = root(e.target);
    if (!r) return;
    if (e.target.classList.contains('hfp-path')) {
      if (e.key === 'Enter') { e.preventDefault(); send(r, {op: 'goto', path: e.target.value}); }
      return;
    }
    var inFilter = e.target.classList.contains('hfp-filter');
    if (!inFilter && !e.target.classList.contains('hfp-list')) return;
    var list = rows(r), at = list.indexOf(current(r));
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      var next = e.key === 'ArrowDown' ? Math.min(list.length - 1, at + 1) : Math.max(0, at - 1);
      if (list.length) select(r, list[next], true);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      activate(r, current(r) || (list.length === 1 ? list[0] : null));
    } else if (e.key === 'Backspace' && !inFilter) {
      e.preventDefault();
      send(r, {op: 'up'});
    }
  });
  document.addEventListener('input', function (e) {
    if (!e.target.classList || !e.target.classList.contains('hfp-filter')) return;
    var r = root(e.target), q = e.target.value.trim().toLowerCase();
    r.querySelectorAll('.hfp-row').forEach(function (row) {
      row.hidden = q !== '' && row.dataset.name !== '..' && row.dataset.name.toLowerCase().indexOf(q) < 0;
    });
    // the first match is selected, a file before a folder (a file is what is
    // being picked), and Enter takes it
    var on = current(r);
    if (q !== '' && (!on || on.hidden || on.dataset.name === '..')) {
      var shown = rows(r).filter(function (x) { return x.dataset.name !== '..'; });
      var file = shown.filter(function (x) { return x.dataset.kind === 'file'; })[0];
      select(r, file || shown[0] || null, true);
    } else if (on && on.hidden) {
      select(r, null);
    }
  });
  // a freshly drawn listing: keep the filter, select what the server asks for
  $(document).on('shiny:value', function (e) {
    setTimeout(function () {
      document.querySelectorAll('.hfp').forEach(function (r) {
        var list = r.querySelector('.hfp-list');
        if (!list || list.dataset.drawn === list.dataset.stamp) return;
        list.dataset.drawn = list.dataset.stamp;
        applySort(r);
        var f = r.querySelector('.hfp-filter'), filtering = f && f.value.trim() !== '';
        var want = list.dataset.select, row = null;
        if (want) r.querySelectorAll('.hfp-row').forEach(function (x) { if (x.dataset.name === want) row = x; });
        select(r, row, true);
        // a filter typed before this drawing still applies, and still picks its
        // first match unless the server asked for a particular entry
        if (filtering) f.dispatchEvent(new Event('input', {bubbles: true}));
        var path = r.querySelector('.hfp-path');
        if (path) path.value = r.dataset.cwd = list.dataset.cwd;
        if (!f || document.activeElement !== f) list.focus({preventScroll: true});
      });
    }, 0);
  });
})();
"""


@module.ui
def file_picker_button(label="Browse...", tooltip="Pick a file on this computer"):
    """The button that opens a file picker.

    Put it with the path field it fills; wrapping both in
    ``ui.div(field, button, class_="hfp-field")`` (see :func:`file_picker_field`)
    places it on the left of the box, joined to it, as Shiny's file input does.

    Parameters
    ----------
    label : str, optional
        Button text. Defaults to "Browse...".
    tooltip : str, optional
        Hover text.

    Returns
    -------
    htmltools.TagList
    """
    return ui.TagList(
        ui.tags.style(_CSS),
        ui.tags.script(_JS),
        ui.tags.script(_MEMORY_JS % (_MAX_RECENT, _BROWSER_RECENT_INPUT)),
        ui.input_action_button(
            "open",
            label,
            class_="btn-default hfp-browse",
            title=tooltip,
        ),
    )


def _natural_key(name):
    """Sort ``run_it2`` before ``run_it10``, ignoring case."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _size_text(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def _matches(name, patterns):
    low = name.lower()
    return any(fnmatch.fnmatch(low, p.lower()) for p in patterns)


def list_folder(folder, patterns=(), show_hidden=False, max_entries=5000):
    """The folders and the matching files in ``folder``.

    Parameters
    ----------
    folder : str or Path
    patterns : sequence of str, optional
        Shell patterns a file must match (case ignored); all files when empty.
    show_hidden : bool, optional
        Include names starting with a dot.
    max_entries : int, optional
        Largest number of entries returned.

    Returns
    -------
    entries : list of dict
        ``name``, ``kind`` ("dir" or "file"), ``size`` (bytes, files only) and
        ``mtime`` (seconds), folders first, each group in natural order.
    truncated : int
        How many entries were left out over ``max_entries``.

    Raises
    ------
    OSError
        When the folder cannot be read.
    """
    dirs, files = [], []
    with os.scandir(folder) as it:
        for entry in it:
            name = entry.name
            if not show_hidden and name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                continue
            if is_dir:
                dirs.append(entry)
            elif not patterns or _matches(name, patterns):
                files.append(entry)
    dirs.sort(key=lambda e: _natural_key(e.name))
    files.sort(key=lambda e: _natural_key(e.name))
    entries = []
    for e in dirs:
        try:
            mtime = e.stat().st_mtime
        except OSError:
            mtime = None
        entries.append(dict(name=e.name, kind="dir", size=None, mtime=mtime))
    for e in files:
        try:
            st = e.stat()
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size, mtime = None, None
        entries.append(dict(name=e.name, kind="file", size=size, mtime=mtime))
    truncated = max(0, len(entries) - max_entries)
    return entries[:max_entries], truncated


def recent_folders(session):
    """The recently used folders of one browser session, newest first.

    Parameters
    ----------
    session : shiny.Session
        The session, or a module proxy of it (they share the list).

    Returns
    -------
    list of str
        The session's own list, changed in place by :func:`_remember`.
    """
    root = session
    while getattr(root, "_root_session", None) is not None:
        root = root._root_session
    recent = getattr(root, _RECENT_ATTR, None)
    if recent is None:
        recent = []
        setattr(root, _RECENT_ATTR, recent)
    return recent


def _start_folder(start, recent=()):
    """The folder (and file to select) the dialog opens at.

    Parameters
    ----------
    start : callable, str or None
        The path to open at, or a function returning it.
    recent : sequence of str, optional
        The recently used folders, newest first, to fall back on.

    Returns
    -------
    tuple of (Path, str or None)
        The folder, and the name of the file to select in it.
    """
    value = start() if callable(start) else start
    if value and "://" not in str(value):
        p = Path(str(value).strip()).expanduser()
        try:
            if p.is_file():
                return p.absolute().parent, p.name
            if p.is_dir():
                return p.absolute(), None
            if p.parent.is_dir() and str(p.parent) not in ("", "."):
                return p.absolute().parent, None
        except OSError:
            pass
    for folder in recent:
        if Path(folder).is_dir():
            return Path(folder), None
    return Path.cwd(), None


def _remember(folder, recent):
    """Put ``folder`` first in the list ``recent``, without repeats.

    Parameters
    ----------
    folder : str or Path
        The folder just used.
    recent : list of str
        The list to update in place (see :func:`recent_folders`).
    """
    folder = str(folder)
    if folder in recent:
        recent.remove(folder)
    recent.insert(0, folder)
    del recent[_MAX_RECENT:]


def browser_folders(session):
    """The folders this browser picked files from on earlier visits.

    They are kept in the browser's local storage and sent once the page
    connects, as a page-level input every picker reads.

    Parameters
    ----------
    session : shiny.Session
        The session, or a module session of it.

    Returns
    -------
    list of str
        Newest first, at most a handful; empty before the page has sent them.
    """
    root = session.root_scope() if hasattr(session, "root_scope") else session
    with reactive.isolate():
        try:
            if _BROWSER_RECENT_INPUT not in root.input:
                return []
            value = root.input[_BROWSER_RECENT_INPUT]()
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(f) for f in value if isinstance(f, str) and f][:_MAX_RECENT]


def _merged(*lists):
    """The lists one after another, without repeats."""
    out = []
    for items in lists:
        for item in items:
            if item not in out:
                out.append(item)
    return out


def _server_files_refused():
    """True, after telling the user, when the app runs on a hosting service.

    The picker lists the server's own files, which a hosted copy must not
    show; the apps hide the Browse button there, and this is a second check.
    """
    from helicon.webApps import deployment

    if not deployment.is_cloud():
        return False
    ui.notification_show(
        "This copy of Helicon runs on a hosting service: the server's files "
        "cannot be browsed.",
        type="warning",
    )
    return True


@module.server
def file_picker_server(
    input,
    output,
    session,
    patterns=(),
    title="Select a file",
    start=None,
    max_entries=5000,
):
    """Run a file picker opened by :func:`file_picker_button` of the same id.

    Parameters
    ----------
    patterns : sequence of str, optional
        Shell patterns of the files to list (e.g. ``("*.star", "*.cs")``); all
        files when empty. The dialog can show all files anyway.
    title : str, optional
        The dialog's title.
    start : callable or str, optional
        The path to open at -- typically the field's current value, as a
        function returning it. A file is selected in its folder; otherwise the
        dialog opens at the folder used last, or the working folder.
    max_entries : int, optional
        Largest number of entries listed in a folder.

    Returns
    -------
    reactive.Value
        The absolute path of the file chosen, set each time one is chosen.
    """
    patterns = tuple(patterns or ())
    recent = recent_folders(session)
    chosen = reactive.value(None)
    cwd = reactive.value(Path.cwd())
    preselect = reactive.value(None)
    stamp = reactive.value(0)
    # each drawing of the listing is numbered, so the page script runs once on it
    drawn = itertools.count(1)

    def go(folder, select=None):
        try:
            folder = Path(folder).expanduser().absolute()
            if ".." in folder.parts:
                folder = folder.resolve()
            if not folder.is_dir():
                raise NotADirectoryError(str(folder))
        except OSError as e:
            ui.notification_show(f"Cannot open {folder}: {e}", type="warning")
            return
        preselect.set(select)
        cwd.set(folder)
        stamp.set(stamp() + 1)

    @reactive.effect
    @reactive.event(input.open)
    def _open():
        if _server_files_refused():
            return
        # this session's folders first, then those of earlier visits
        folder, name = _start_folder(start, _merged(recent, browser_folders(session)))
        go(folder, name)
        ui.modal_show(
            ui.modal(
                _dialog(),
                title=title,
                size="l",
                easy_close=True,
                footer=ui.div(
                    ui.tags.span(class_="hfp-chosen"),
                    ui.modal_button("Cancel"),
                    ui.tags.button(
                        "Select",
                        type="button",
                        class_="btn btn-primary hfp-choose",
                        disabled=True,
                    ),
                    class_="hfp-foot",
                ),
            )
        )

    def _dialog():
        tools = [
            ui.tags.input(
                type="search",
                class_="form-control form-control-sm hfp-filter",
                placeholder="Filter by name",
                aria_label="Filter by name",
            )
        ]
        if patterns:
            tools.append(
                ui.input_checkbox("all_files", "All files", value=False, width="auto")
            )
        tools.append(ui.input_checkbox("hidden", "Hidden", value=False, width="auto"))
        return ui.div(
            ui.tags.style(_CSS),
            ui.tags.script(_JS),
            ui.output_ui("pathbar"),
            ui.tags.input(
                type="text",
                class_="form-control form-control-sm hfp-path",
                spellcheck="false",
                aria_label="Path: type or paste one, then Enter",
                title="Type or paste a path, then press Enter",
            ),
            ui.output_ui("places"),
            ui.div(*tools, class_="hfp-tools"),
            ui.output_ui("listing"),
            ui.tags.small(
                "Double-click to open a folder or choose a file · ↑↓ "
                "Enter · Backspace: up a folder"
                + (f" · showing {', '.join(patterns)}" if patterns else ""),
                class_="text-muted",
            ),
            class_="hfp",
            data_act=session.ns("act"),
            data_cwd=str(cwd()),
        )

    @reactive.effect
    @reactive.event(input.act)
    def _act():
        if _server_files_refused():
            return
        msg = input.act() or {}
        op = msg.get("op")
        here = cwd()
        if op == "open":
            name = msg.get("name", "")
            if name == "..":
                if here.parent != here:
                    go(here.parent, here.name)
            else:
                go(here / name)
        elif op == "up":
            if here.parent != here:
                go(here.parent, here.name)
        elif op == "home":
            go(Path.home())
        elif op == "work":
            go(Path.cwd())
        elif op == "goto":
            text = (msg.get("path") or "").strip()
            if not text:
                return
            p = Path(os.path.expandvars(text)).expanduser()
            if p.is_file():
                go(p.parent, p.name)
            elif p.is_dir():
                go(p)
            else:
                ui.notification_show(f"No such file or folder: {text}", type="warning")
        elif op == "choose":
            p = here / msg.get("name", "")
            if p.is_file():
                _remember(here, recent)
                chosen.set(str(p))
                ui.modal_remove()

    @render.ui
    def pathbar():
        here = cwd()
        parts = here.parts
        crumbs = []
        for i, part in enumerate(parts):
            target = Path(*parts[: i + 1])
            if i:
                crumbs.append(ui.tags.span("/" if i > 1 else "", class_="hfp-sep"))
            crumbs.append(
                ui.tags.button(
                    part if part != "/" else "/",
                    type="button",
                    class_="hfp-crumb",
                    data_op="goto",
                    data_path=str(target),
                    title=str(target),
                )
            )
        return ui.div(
            ui.tags.button(
                ui.HTML(_UP_SVG),
                "Up",
                type="button",
                class_="hfp-btn",
                data_op="up",
                title="Up a folder (Backspace)",
                disabled=here.parent == here,
            ),
            ui.tags.button(
                ui.HTML(_HOME_SVG),
                type="button",
                class_="hfp-btn",
                data_op="home",
                title=f"Home: {Path.home()}",
            ),
            ui.tags.button(
                ui.HTML(_WORK_SVG),
                type="button",
                class_="hfp-btn",
                data_op="work",
                title=f"Working folder: {Path.cwd()}",
            ),
            ui.div(*crumbs, class_="hfp-crumbs"),
            class_="hfp-bar",
        )

    @render.ui
    def places():
        cwd()
        seen, chips = set(), []
        folders = _merged(recent, browser_folders(session))[:_MAX_RECENT]
        for label, folder in [("Home", Path.home()), ("Working folder", Path.cwd())] + [
            (Path(f).name or f, Path(f)) for f in folders if Path(f).is_dir()
        ]:
            key = str(folder)
            if key in seen:
                continue
            seen.add(key)
            chips.append(
                ui.tags.button(
                    label,
                    type="button",
                    class_="hfp-chip",
                    data_op="goto",
                    data_path=key,
                    title=key,
                )
            )
        return ui.div(
            ui.tags.span("Places", class_="hfp-label"), *chips, class_="hfp-places"
        )

    @render.ui
    def listing():
        here = cwd()
        stamp()
        show_all = bool(input.all_files()) if patterns else True
        try:
            entries, cut = list_folder(
                here,
                () if show_all else patterns,
                show_hidden=bool(input.hidden()),
                max_entries=max_entries,
            )
        except OSError as e:
            return ui.div(
                ui.div(
                    f"Cannot read this folder: {e.strerror or e}", class_="hfp-error"
                ),
                class_="hfp-list",
                tabindex="0",
                data_cwd=str(here),
                data_stamp=str(next(drawn)),
            )
        rows = []
        if here.parent != here:
            rows.append(
                '<tr class="hfp-row" data-kind="dir" data-name="..">'
                f'<td class="hfp-icon">{_UP_SVG}</td><td>..</td><td class="hfp-size"></td>'
                '<td class="hfp-time"></td></tr>'
            )
        for e in entries:
            name = html.escape(e["name"], quote=True)
            icon = _FOLDER_SVG if e["kind"] == "dir" else _FILE_SVG
            size = _size_text(e["size"]) if e["size"] is not None else ""
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(e["mtime"]))
                if e["mtime"]
                else ""
            )
            # the raw numbers, for sorting in the browser
            raw_size = "" if e["size"] is None else str(e["size"])
            raw_time = "" if e["mtime"] is None else f'{e["mtime"]:.0f}'
            rows.append(
                f'<tr class="hfp-row" data-kind="{e["kind"]}" data-name="{name}" title="{name}"'
                f' data-size="{raw_size}" data-mtime="{raw_time}">'
                f'<td class="hfp-icon">{icon}</td><td>{name}</td>'
                f'<td class="hfp-size">{size}</td><td class="hfp-time">{when}</td></tr>'
            )
        notes = []
        if not entries:
            notes.append(
                "No folders or matching files here."
                + (
                    " Tick All files to see the rest."
                    if patterns and not show_all
                    else ""
                )
            )
        if cut:
            notes.append(f"{cut:,} more not listed: type a filter to narrow it down.")
        table = (
            '<table class="hfp-table"><thead><tr><th class="hfp-icon"></th>'
            '<th class="hfp-sort" data-sort="name" title="Sort by name (again: reverse)">Name</th>'
            '<th class="hfp-size hfp-sort" data-sort="size" title="Sort by size (again: reverse)">Size</th>'
            '<th class="hfp-time hfp-sort" data-sort="mtime" title="Sort by time (again: reverse)">Modified</th>'
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        return ui.div(
            ui.HTML(table),
            *[ui.div(n, class_="hfp-note") for n in notes],
            class_="hfp-list",
            tabindex="0",
            data_cwd=str(here),
            data_select=preselect() or "",
            data_stamp=str(next(drawn)),
        )

    return chosen


def file_picker_field(field, picker_id, enabled=True, **button):
    """A path field with its Browse button beside it.

    Parameters
    ----------
    field : htmltools.Tag
        The field, e.g. a ``ui.input_text``.
    picker_id : str
        The id of the picker (see :func:`file_picker_fill`).
    enabled : bool, optional
        False leaves the field on its own -- for an app that runs where the
        user's files are not, such as a hosted one.
    **button
        Passed to :func:`file_picker_button`.

    Returns
    -------
    htmltools.Tag
    """
    if not enabled:
        return field
    return ui.div(field, file_picker_button(picker_id, **button), class_="hfp-field")


def file_picker_fill(
    picker_id, field_id, input, patterns=(), title="Select a file", update=None
):
    """Run the picker of ``picker_id`` for the text field ``field_id``.

    The dialog opens at the file the field names, and the file chosen is put
    in the field. Call it from the server function of the module (or app) the
    field belongs to.

    Parameters
    ----------
    picker_id, field_id : str
        Ids within the calling module.
    input : shiny.Inputs
        The calling module's inputs.
    patterns : sequence of str, optional
        Shell patterns of the files to list.
    title : str, optional
        The dialog's title.
    update : callable, optional
        ``update(field_id, value=path)``; defaults to ``ui.update_text``.

    Returns
    -------
    reactive.Value
        The path chosen last.
    """
    update = update or ui.update_text
    chosen = file_picker_server(
        picker_id, patterns=patterns, title=title, start=lambda: input[field_id]()
    )

    @reactive.effect
    @reactive.event(chosen)
    def _fill_field():
        update(field_id, value=chosen())

    return chosen


# The input mode that takes a file on the computer running the app.
SERVER = "server"


def source_modes(cloud=False, modes=("upload", "url")):
    """The choices of a "how to obtain the input" radio, ``server`` first.

    ``server`` -- a file on the computer running the app, picked with
    :func:`file_picker_field` -- is offered only when the app runs where the
    user's files are (``cloud`` False).

    Parameters
    ----------
    cloud : bool, optional
        Whether the app runs on a hosting service.
    modes : sequence, optional
        The other modes, as values (shown as they are) or (value, label) pairs.

    Returns
    -------
    dict
        Value -> label, for ``ui.input_radio_buttons(choices=...)``.
    """
    choices = {}
    if not cloud:
        choices[SERVER] = ui.span(
            SERVER, title="A file on the computer running Helicon"
        )
    for m in modes:
        value, label = m if isinstance(m, (tuple, list)) else (m, m)
        choices[value] = label
    return choices
