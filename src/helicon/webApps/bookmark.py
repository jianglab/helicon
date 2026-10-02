"""Bookmark URLs shared by every tab of the Helicon web app.

A bookmark is the page URL. It names the tab and, for that tab, only the
parameters whose values differ from their defaults, under short keys::

    ?tab=AbInitio3D&rise=4.8&csym=2&hand=right&merge_counterparts=0

Each tab module declares its parameters once, in ``BOOKMARK_DEFAULTS``::

    BOOKMARK_DEFAULTS = {
        "rise": ("rise", 4.75),               # short key: (input id, default)
        "apix": ("hill_apix", 2.3438, DERIVED),
    }

and everything else is derived from that table: the browser script that
keeps the URL up to date, the restore when a bookmark is opened, and the
URLs the file browser (``helicon display``) launches tabs with.

Two kinds of parameter:

- an ordinary parameter is in the URL whenever its value differs from the
  declared default, which therefore has to be the value the input shows
  when the tab opens on its own example data;
- a ``DERIVED`` parameter is one the tab works out from the loaded data
  (the pixel size read from a file, a radial range measured from a map).
  It is in the URL only once the user has edited it, or when the bookmark
  being viewed set it, so that loading other data does not by itself
  lengthen the URL.

A value is restored by Shiny's own ``restore_input``, which every
``ui.input_*`` (and ``helicon.shiny`` slider) calls when it is built. That
covers inputs a tab renders later, when its data arrives, as well as those
on the page from the start, and the server sees the bookmarked values in
its first batch of inputs, before any effect runs.

Values are written in their plainest form: numbers as numbers, ``1``/``0``
for booleans, strings as they are, and lists (range sliders, checkbox
groups) comma-separated. The declared default's type says how to read a
value back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import parse_qsl

DERIVED = "derived"

# The query keys that are not tab parameters.
TAB_KEY = "tab"
RESERVED_KEYS = frozenset({TAB_KEY, "helicon_token", "helicon_theme"})


@dataclass(frozen=True)
class Entry:
    """One bookmarked parameter of a tab."""

    input_id: str  # the input's id within the tab's module
    default: Any
    derived: bool = False


def tab_entries(tab_module) -> dict[str, Entry]:
    """The bookmarked parameters of a tab module, by short key.

    Parameters
    ----------
    tab_module : module
        A tab module with a ``BOOKMARK_DEFAULTS`` table.

    Returns
    -------
    dict
        Short key -> :class:`Entry`.
    """
    out = {}
    for key, spec in getattr(tab_module, "BOOKMARK_DEFAULTS", {}).items():
        input_id, default, *flags = spec
        out[key] = Entry(input_id, default, DERIVED in flags)
    return out


def encode_value(value: Any) -> str:
    """A parameter value as it is written in the URL.

    Parameters
    ----------
    value : bool, int, float, str, list or tuple

    Returns
    -------
    str
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ",".join(encode_value(v) for v in value)
    return str(value)


def _number(text: str):
    try:
        return int(text)
    except ValueError:
        return float(text)


def decode_value(text: str, default: Any) -> Any:
    """Read a URL value back, as the type of the parameter's default.

    Parameters
    ----------
    text : str
        The value as written by :func:`encode_value` (or as JSON, the form
        of older bookmarks).
    default : Any
        The parameter's declared default, which gives the type.

    Returns
    -------
    Any

    Raises
    ------
    ValueError
        If ``text`` cannot be read as that type.
    """
    if isinstance(default, bool):
        low = text.strip().lower()
        if low in ("1", "true"):
            return True
        if low in ("0", "false"):
            return False
        raise ValueError(f"not a boolean: {text!r}")
    if isinstance(default, (int, float)):
        return _number(text.strip())
    if isinstance(default, (list, tuple)):
        if text.startswith("["):
            return json.loads(text)
        items = [t for t in text.split(",") if t != ""]
        if default and all(
            isinstance(d, (int, float)) and not isinstance(d, bool) for d in default
        ):
            return [_number(t) for t in items]
        return items
    if default is None:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def query(tab: str, params: Mapping[str, Any] | None = None) -> dict[str, str]:
    """The query parameters of a bookmark URL that opens ``tab``.

    Used by the file browser to open a tab on a file; ``params`` uses the
    tab's short keys.

    Parameters
    ----------
    tab : str
        The tab's name in the navbar, e.g. ``"HelicalPitch"``.
    params : mapping, optional
        Short key -> value.

    Returns
    -------
    dict
        Query key -> encoded value, in URL order.
    """
    out = {TAB_KEY: tab}
    for key, value in (params or {}).items():
        out[key] = encode_value(value)
    return out


def _legacy_params(pairs: list[tuple[str, str]]) -> tuple[str | None, dict]:
    """Tab and short-key values of a bookmark in the earlier form,
    ``?_inputs_&helicon_tab="Tab"&_values_&p={...}``."""
    tab, params = None, {}
    for key, value in pairs:
        if key == "helicon_tab":
            try:
                tab = json.loads(value)
            except ValueError:
                tab = value.strip('"')
        elif key == "p":
            try:
                params = json.loads(value)
            except ValueError:
                params = {}
    return tab, params


def parse(query_string: str, tabs: Mapping[str, Any]) -> tuple[str | None, dict]:
    """The tab a bookmark URL opens, and its inputs to restore.

    Parameters
    ----------
    query_string : str
        The URL's query string, with or without the leading ``?``.
    tabs : mapping
        Tab name -> (module namespace, tab module), as in the app's
        ``_TAB_MODULE_MAP``.

    Returns
    -------
    tab : str or None
        The tab named by the URL, if it is one of ``tabs``.
    inputs : dict
        Full input id (``"<namespace>-<input id>"``) -> value. Keys that are
        not parameters of the tab, and values that cannot be read, are
        left out.
    """
    pairs = parse_qsl(query_string.lstrip("?"), keep_blank_values=True)
    if any(key in ("_inputs_", "helicon_tab") for key, _ in pairs):
        tab, raw = _legacy_params(pairs)
        texts = {k: v if isinstance(v, str) else json.dumps(v) for k, v in raw.items()}
        typed = {k: v for k, v in raw.items() if not isinstance(v, str)}
    else:
        tab = next((v for k, v in pairs if k == TAB_KEY), None)
        texts = {k: v for k, v in pairs if k not in RESERVED_KEYS}
        typed = {}
    if tab not in tabs:
        return None, {}
    namespace, module = tabs[tab]
    from . import deployment

    # a bookmark made where the files are, opened on a hosting service: the
    # "server" input mode is not offered there, so the tab keeps its default
    no_server = deployment.is_cloud()
    inputs = {}
    for key, entry in tab_entries(module).items():
        if key in typed:
            value = typed[key]
        elif key in texts:
            try:
                value = decode_value(texts[key], entry.default)
            except (TypeError, ValueError):
                continue
        else:
            continue
        if no_server and value == "server":
            continue
        inputs[f"{namespace}-{entry.input_id}"] = value
    return tab, inputs


# ── Restore, through Shiny's own input restoration ──────────────────
#
# Shiny restores an input from the "restore context" active when the input
# is built: the page's UI is built under one made from the HTTP request, and
# a session keeps one, made from the URL, for the inputs its renders build.
# Shiny fills both only from its own URL format (``_inputs_&<full id>=<JSON>``);
# these two functions fill them from ours. The class is public
# (``shiny.bookmark.RestoreContext``); the input set, the context manager and
# the session's attribute are not, which is why they are confined to here and
# covered by tests/test_bookmark.py.


def ui_restore_context(inputs: Mapping[str, Any]):
    """A context manager under which a UI built shows ``inputs``' values.

    Parameters
    ----------
    inputs : mapping
        Full input id -> value, as returned by :func:`parse`.

    Returns
    -------
    contextlib.AbstractContextManager
    """
    from shiny.bookmark import RestoreContext
    from shiny.bookmark._restore_state import RestoreInputSet, restore_context

    ctx = RestoreContext()
    ctx.active = bool(inputs)
    ctx.input = RestoreInputSet(dict(inputs))
    return restore_context(ctx)


def restore_in_session(session, inputs: Mapping[str, Any]) -> None:
    """Have inputs that ``session`` builds later start from ``inputs``' values.

    Each value is used once, by the first input built with its id, so a
    re-render later shows what the tab's code gives it.

    Parameters
    ----------
    session : shiny.Session
    inputs : mapping
        Full input id -> value, as returned by :func:`parse`.
    """
    if not inputs:
        return
    from shiny.bookmark import RestoreContext
    from shiny.bookmark._restore_state import RestoreInputSet

    ctx = getattr(session.bookmark, "_restore_context", None)
    if ctx is None:
        ctx = RestoreContext()
        session.bookmark._set_restore_context(ctx)
    ctx.active = True
    ctx.input = RestoreInputSet({**ctx.input.as_dict(), **inputs})


def js_table(tabs: Mapping[str, Any]) -> str:
    """The bookmark table for the page script, as a JSON object literal.

    Parameters
    ----------
    tabs : mapping
        Tab name -> (module namespace, tab module).

    Returns
    -------
    str
        ``{tab: {short key: [full input id, default, derived]}}``.
    """
    table = {}
    for tab, (namespace, module) in tabs.items():
        table[tab] = {
            key: [f"{namespace}-{e.input_id}", e.default, e.derived]
            for key, e in tab_entries(module).items()
        }
    return json.dumps(table)
