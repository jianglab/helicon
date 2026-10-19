"""Unit tests for the web app's display-navigation control plane.

The display file browser reuses one browser tab per tool by POSTing new
bookmark params to ``/helicon/navigate``; the open page polls
``/helicon/pending`` and reloads itself.  These tests cover the pure logic
behind those endpoints: the token gate, the pending-navigation store, the
aliveness probe, and the shared query-string encoding.
"""

import time

from helicon.lib.shiny import encode_query_params
from helicon.webApps.app import _AppControl


def test_encode_query_params_bare_keys_for_empty_values():
    assert encode_query_params({"_inputs_": ""}) == "_inputs_"
    assert (
        encode_query_params({"a": "1", "_inputs_": "", "b": "x y"})
        == "a=1&_inputs_&b=x%20y"
    )


def test_encode_query_params_quotes_values():
    assert encode_query_params({"helicon_tab": '"X"'}) == "helicon_tab=%22X%22"


def test_navigate_stores_pending_and_poll_consumes():
    control = _AppControl()
    # Fresh server: seen_tokens is empty so any token is accepted.
    result = control.navigate("tok1", {"helicon_tab": '"X"'})
    assert result["ok"] is True
    assert "alive" in result

    # Before the page registers its token, poll stays quiet but does NOT
    # consume the pending navigation (the early return skips the consume).
    assert control.poll("tok1") == {"pending": False}

    # Once the page loads, its token is registered and the next poll
    # consumes the pending navigation. The token is carried into the new
    # URL so the reloaded page keeps polling under the same token.
    control.register_token("?helicon_token=tok1")
    polled = control.poll("tok1")
    assert polled["pending"] is True
    assert polled["query_string"] == "helicon_tab=%22X%22&helicon_token=tok1"

    # A second poll confirms the navigation was consumed.
    assert control.poll("tok1") == {"pending": False}


def test_token_gate_rejects_unknown_token_once_registered():
    control = _AppControl()
    control.register_token("?helicon_token=tok1")
    assert control.navigate("tok1", {"helicon_tab": '"X"'})["ok"] is True
    assert control.navigate("tok2", {"helicon_tab": '"X"'})["ok"] is False


def test_only_a_launched_server_reaps_itself(monkeypatch):
    """A server the user started must outlive a quiet spell.

    ``helicon display`` launches a server for one viewer and sets the flag, so
    that server may kill itself when the viewer is gone. ``helicon webApps``
    and a hosted deployment must not: no session open does not mean no job
    running, and the URL would stop answering.
    """
    monkeypatch.delenv("HELICON_WEBAPP_SELF_REAP", raising=False)
    control = _AppControl()
    control.start_watchdog()
    assert control._watchdog_started is False

    monkeypatch.setenv("HELICON_WEBAPP_SELF_REAP", "1")
    control.start_watchdog()
    assert control._watchdog_started is True


def test_a_navigation_is_not_delivered_to_another_tokens_poll():
    """Two launchers, one server: each poll sees only its own navigation."""
    control = _AppControl()
    control.register_token("?helicon_token=tok1")
    control.register_token("?helicon_token=tok2")
    assert control.navigate("tok2", {"helicon_tab": '"X"'})["ok"] is True

    # The navigation was for tok2, so tok1's poll must not consume it.
    assert control.poll("tok1") == {"pending": False}
    polled = control.poll("tok2")
    assert polled["pending"] is True
    assert polled["query_string"] == "helicon_tab=%22X%22&helicon_token=tok2"
    assert control.poll("tok2") == {"pending": False}


def test_navigate_rejects_missing_or_non_dict_params():
    control = _AppControl()
    assert control.navigate("tok1", None)["ok"] is False
    assert control.navigate("tok1", {})["ok"] is False


def test_poll_returns_false_for_unknown_token():
    control = _AppControl()
    assert control.poll("nope") == {"pending": False}


def test_is_alive_tracks_sessions_and_polls():
    control = _AppControl()
    control.start_ts = time.monotonic() - 1000  # past the young-server grace
    control.last_poll_ts = time.monotonic() - 1000
    assert control.is_alive() is False

    control.start_session()
    assert control.is_alive() is True
    control.end_session()

    control.poll("tok1")
    assert control.is_alive() is True
