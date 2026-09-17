"""``getLogger`` must not accumulate handlers.

``logging.getLogger(name)`` returns the same logger every time, so adding
handlers on each call multiplies the output -- twice for the second call, three
times for the third. The console handler writes to stdout, which for the web app
is a pipe to whatever launched it; once the multiplied output fills that pipe the
write blocks and the app wedges, with every worker idle and the main thread in
anon_pipe_write. That presented as a search running fine once and hanging partway
through the next.
"""

import logging

import helicon


def test_repeated_calls_do_not_stack_handlers(tmp_path):
    logfile = tmp_path / "repeat.log"
    counts = [
        len(helicon.getLogger(logfile=str(logfile), verbose=1).handlers)
        for _ in range(5)
    ]
    assert counts == [2] * 5


def test_one_message_is_written_once(tmp_path):
    logfile = tmp_path / "once.log"
    for _ in range(4):
        log = helicon.getLogger(logfile=str(logfile), verbose=1)
    log.info("a distinctive line")
    written = [l for l in logfile.read_text().splitlines() if "a distinctive line" in l]
    assert len(written) == 1


def test_verbosity_of_the_latest_call_wins(tmp_path):
    logfile = tmp_path / "verbosity.log"
    helicon.getLogger(logfile=str(logfile), verbose=0)
    log = helicon.getLogger(logfile=str(logfile), verbose=3)
    console = [h for h in log.handlers if not isinstance(h, logging.FileHandler)]
    assert console and console[0].level == logging.DEBUG


def test_separate_logfiles_keep_separate_handlers(tmp_path):
    a = helicon.getLogger(logfile=str(tmp_path / "a.log"), verbose=1)
    b = helicon.getLogger(logfile=str(tmp_path / "b.log"), verbose=1)
    assert a is not b
    assert len(a.handlers) == 2 and len(b.handlers) == 2


def test_only_handlers_this_function_added_are_removed(tmp_path):
    """A handler someone else attached to the same logger must survive."""
    logfile = tmp_path / "foreign.log"
    log = helicon.getLogger(logfile=str(logfile), verbose=1)
    foreign = logging.NullHandler()
    log.addHandler(foreign)
    helicon.getLogger(logfile=str(logfile), verbose=1)
    assert foreign in log.handlers
