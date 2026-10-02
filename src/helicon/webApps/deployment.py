"""Whether the web app runs on the user's own machine or on a hosting service.

Some features need the user's files on the server: WhereIsMyClass reads the
micrographs, the file browser opens on the server's desktop, and Home starts
helicon subcommands there. On a hosting service the server has none of that,
so those features are turned off.

The answer comes from the server's environment, not from the address the page
was opened at: a lab server reached by its hostname has the lab's files, and a
hosted app opened through a tunnel does not.

``HELICON_DEPLOYMENT=cloud`` or ``HELICON_DEPLOYMENT=local`` decides it
outright (for a host not listed here, or to override one that is). Otherwise
the environment variables the common hosting services set are checked, and
with none of them the app is taken to run locally, as ``helicon webApps`` and
the file browser start it.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlsplit

# (host, variable, value or None for "set to anything"), checked in order.
# Sources: each service's documentation of the variables it sets in a
# running deployment.
_HOSTS = (
    # Posit Connect Cloud (connect.posit.cloud), which hosts helicon now
    ("Posit Connect Cloud", "R_CONFIG_ACTIVE", "connect_cloud"),
    ("Posit Connect Cloud", "QUARTO_PROFILE", "connect_cloud"),
    # shinyapps.io
    ("shinyapps.io", "R_CONFIG_ACTIVE", "shinyapps"),
    # self-hosted Posit Connect
    ("Posit Connect", "POSIT_PRODUCT", "CONNECT"),
    ("Posit Connect", "RSTUDIO_PRODUCT", "CONNECT"),
    # Shiny Server (and the Posit products built on it) sets the port it gives
    # the app; Shiny itself reads this to tell it is behind one
    ("Shiny Server", "SHINY_PORT", None),
    # Hugging Face Spaces
    ("Hugging Face Spaces", "SPACE_ID", None),
    # Google Cloud
    ("Google Cloud Run", "K_SERVICE", None),
    ("Google App Engine", "GAE_SERVICE", None),
    # Amazon Web Services
    ("AWS Lambda", "AWS_LAMBDA_FUNCTION_NAME", None),
    ("AWS ECS", "ECS_CONTAINER_METADATA_URI_V4", None),
    ("AWS ECS", "ECS_CONTAINER_METADATA_URI", None),
    ("AWS App Runner", "AWS_APP_RUNNER_SERVICE_ID", None),
    # Microsoft Azure
    ("Azure App Service", "WEBSITE_SITE_NAME", None),
    ("Azure Container Apps", "CONTAINER_APP_NAME", None),
    # platforms as a service
    ("Heroku", "DYNO", None),
    ("Render", "RENDER_SERVICE_ID", None),
    ("Fly.io", "FLY_APP_NAME", None),
    ("Railway", "RAILWAY_PROJECT_ID", None),
)

_OVERRIDE = "HELICON_DEPLOYMENT"


def hosting_service(environ=None) -> str | None:
    """The hosting service the app runs on, or None.

    Parameters
    ----------
    environ : mapping, optional
        The environment to look in. Defaults to ``os.environ``.

    Returns
    -------
    str or None
        The service's name when one of its variables is set, else None.
    """
    env = os.environ if environ is None else environ
    for host, name, value in _HOSTS:
        got = env.get(name)
        if got is None or got == "":
            continue
        if value is None or got.strip().lower() == value.lower():
            return host
    return None


def is_cloud(environ=None) -> bool:
    """Whether the app runs on a hosting service rather than locally.

    ``HELICON_DEPLOYMENT`` (``cloud`` or ``local``) decides it when set;
    otherwise :func:`hosting_service`.

    Parameters
    ----------
    environ : mapping, optional
        The environment to look in. Defaults to ``os.environ``.

    Returns
    -------
    bool
    """
    env = os.environ if environ is None else environ
    forced = (env.get(_OVERRIDE) or "").strip().lower()
    if forced in ("cloud", "hosted", "remote"):
        return True
    if forced in ("local", "lab"):
        return False
    return hosting_service(env) is not None


_URL_SCHEMES = ("http://", "https://", "ftp://")


def url_allowed(url, environ=None) -> bool:
    """Whether a "url" input may be read: on a hosting service, a URL only.

    The functions that read a URL also read a local path, which on a hosting
    service would let a visitor make the server read its own files.

    Parameters
    ----------
    url : str
    environ : mapping, optional
        The environment to look in. Defaults to ``os.environ``.

    Returns
    -------
    bool
    """
    if not is_cloud(environ):
        return True
    text = str(url).strip()
    if not text.lower().startswith(_URL_SCHEMES):
        return False
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return False
    return host_is_public(host)


def host_is_public(host) -> bool:
    """Whether every address ``host`` resolves to is a public one.

    On a hosting service a URL to ``localhost``, a private network or the
    cloud metadata address (169.254.169.254) would let a visitor make the
    server fetch from inside its own network. A name that does not resolve
    counts as not public (fail closed).

    Parameters
    ----------
    host : str or None
        A host name or a literal IPv4/IPv6 address.

    Returns
    -------
    bool
        True only when the host resolves and none of its addresses is
        loopback, link-local, private, multicast, reserved or unspecified.
    """
    if not host:
        return False
    host = str(host).strip("[]")
    try:
        addrs = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except (OSError, UnicodeError):
            return False
        addrs = []
        for info in infos:
            try:
                addrs.append(ipaddress.ip_address(info[4][0].split("%")[0]))
            except ValueError:
                return False
    if not addrs:
        return False
    for addr in addrs:
        mapped = getattr(addr, "ipv4_mapped", None)
        if mapped is not None:
            addr = mapped
        if (
            addr.is_loopback
            or addr.is_link_local
            or addr.is_private
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
            or not addr.is_global
        ):
            return False
    return True


def refuse_local_path(url) -> bool:
    """Tell the user, and return True, when :func:`url_allowed` says no.

    For a loader of a "url" input: ``if deployment.refuse_local_path(url):
    return``.
    """
    if url_allowed(url):
        return False
    from shiny import ui

    ui.modal_show(
        ui.modal(
            "This copy of Helicon reads files by URL only (http, https or ftp) "
            f"from a public host: {url} is not one.",
            title="Not a URL",
            easy_close=True,
            footer=None,
        )
    )
    return True


def refuse_server_mode() -> bool:
    """Tell the user, and return True, when server files may not be read.

    On a hosting service the server has none of the user's files, and reading
    a path a visitor typed would expose the server's own files. Every loader
    that reads a path on the server (the "server" input mode, companion-file
    lookups, a RELION project folder) starts with ``if
    deployment.refuse_server_mode(): return``, as a server-side check behind
    the UI that already hides those choices.

    Returns
    -------
    bool
        True (after showing a popup) on a hosting service, else False.
    """
    if not is_cloud():
        return False
    from shiny import ui

    ui.modal_show(
        ui.modal(
            "This copy of Helicon runs on a hosting service and cannot read "
            "files on the server. Upload the file or give its URL instead.",
            title="Server files are not available",
            easy_close=True,
            footer=None,
        )
    )
    return True
