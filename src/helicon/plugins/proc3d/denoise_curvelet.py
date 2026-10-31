"""Handler for the denoiseCurvelet option: automatic curvelet + wavelet map denoising."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "denoiseCurvelet"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the denoiseCurvelet option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--denoiseCurvelet",
        nargs="?",
        const="",
        metavar="[strength=<auto|x>][:halfMaps=<half1,half2>][:precision=<single|double>]"
        "[:tileSize=<auto|n>]",
        help="denoise the 3D map automatically: monogenic curvelets (sheets and "
        "lines) and wavelets (spots), with every sub-band's threshold set from its "
        "own noise, averaged. strength scales all thresholds. auto (default): tuned "
        "so that denoised half map 1 best predicts half map 2 when the half maps "
        "are found by their RELION, cryoSPARC or EMDB names next to the input map "
        "(or given as halfMaps) and this is the first option; otherwise 1.5, the "
        "best on amyloid maps (others wanted up to 6); or a number. Tuning takes "
        "4-7 extra denoising runs (one fewer when the map is half map 1). When the "
        "map is a half map or the average of the two, also FSC-weighted or "
        "sharpened (but not masked), every sub-band's noise is measured on their "
        "difference rather than guessed from the map: closer to the truth, with the "
        "best strengths near 1. precision: single (default) gave the same maps "
        "as double, "
        "2-3 times faster in half the memory. tileSize: auto (default) denoises the "
        "whole map when it fits in the free memory, otherwise in the largest tiles "
        "that fit; n uses tiles of n voxels a side, 0 the whole map. CPU count from "
        "--cpu. disabled by default",
        action="append",
        default=None,
    )


def handle(
    data: np.ndarray,
    args: argparse.Namespace,
    index_d: dict,
    param: object,
    apix: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, float, int, int, int]:
    """Handle the denoiseCurvelet option.

    Parameters
    ----------
    data : np.ndarray
        3D volume data.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter string for this option.
    apix : float
        Current pixel size.
    nx : int
        Current X dimension.
    ny : int
        Current Y dimension.
    nz : int
        Current Z dimension.

    Returns
    -------
    tuple[np.ndarray, float, int, int, int]
        (data, apix, nx, ny, nz) after processing.
    """
    if param is not None:
        from helicon.lib.denoise import denoise_map

        _, param_dict = helicon.parse_param_str(param) if param else ({}, {})
        precision = param_dict.get("precision", "single")
        if precision not in ("single", "double"):
            raise HeliconError(
                f"\tERROR: --denoiseCurvelet precision={precision}: "
                "use single or double"
            )
        tile_size = str(param_dict.get("tileSize", "auto"))
        if tile_size != "auto" and not tile_size.isdigit():
            raise HeliconError(
                f"\tERROR: --denoiseCurvelet tileSize={tile_size}: "
                "use auto or a number of voxels"
            )
        n_jobs = getattr(args, "cpu", 0) or 0
        if n_jobs < 1:
            n_jobs = helicon.available_cpu()
        tile_size = None if tile_size == "auto" else int(tile_size)
        strength, strength_why, noise, noise_why, done = _strength(
            data, args, index_d, param_dict, n_jobs, precision, tile_size
        )
        data, info = done or denoise_map(
            data,
            strength=strength,
            workers=n_jobs,
            precision=precision,
            tile_size=tile_size,
            noise=noise,
        )
        del noise
        if getattr(args, "verbose", 0):
            tiles = (
                f"tiles of {info['tile_size']} voxels"
                if info["tile_size"]
                else "the whole map"
            )
            logger.info(
                "\tdenoiseCurvelet: strength %g (%s), %s precision, noise from %s "
                "(%s), %s (%s)",
                info["strength"],
                strength_why,
                info["precision"],
                info["noise_from"],
                noise_why,
                tiles,
                info["tiles_why"],
            )

    index_d[option_name] += 1
    return data, apix, nx, ny, nz


def _strength(data, args, index_d, param_dict, n_jobs, precision, tile_size):
    """The strength to use and why, and the map's noise from its half maps and why.

    The strength is given, tuned on the half maps, or the default. The noise
    map is None when the half maps are not found or do not match the map. When
    the map is half map 1 and was just tuned on, the tuning's run at the chosen
    strength is returned too, so it is not run again.

    Returns
    -------
    tuple of (float, str, numpy.ndarray or None, str, tuple or None)
        The last is ``(denoised, info)`` of that run, or None.
    """
    from helicon.lib.denoise import (
        MAP_STRENGTH,
        find_half_maps,
        map_noise,
        tune_strength,
    )

    strength = str(param_dict.get("strength", "auto"))
    given = None
    if strength != "auto":
        try:
            given = float(strength)
        except ValueError:
            raise HeliconError(
                f"\tERROR: --denoiseCurvelet strength={strength}: use auto or a number"
            ) from None

    def without_halves(why):
        if given is not None:
            return given, "given", None, why, None
        return MAP_STRENGTH, f"the default: {why}", None, why, None

    if "halfMaps" in param_dict:
        halves = str(param_dict["halfMaps"]).split(",")
        if len(halves) != 2:
            raise HeliconError(
                f"\tERROR: --denoiseCurvelet halfMaps={param_dict['halfMaps']}: "
                "give two files, separated by a comma"
            )
    else:
        input_map = getattr(args, "inputMapFile", None)
        halves = find_half_maps(input_map) if input_map else None
        if not halves:
            return without_halves("no half maps found")
    # the half maps are as read: options before this one would have changed the map
    if getattr(args, "all_options", [option_name])[0] != option_name or index_d.get(
        option_name, 0
    ):
        return without_halves("options before it changed the map")
    import mrcfile

    def shape(path):
        with mrcfile.open(path, permissive=True, header_only=True) as mrc:
            return tuple(int(mrc.header[n]) for n in ("nz", "ny", "nx"))

    if any(shape(h) != data.shape for h in halves):
        return without_halves("the half maps are not the map's size")

    def read(path):
        with mrcfile.open(path, permissive=True) as mrc:
            return np.array(mrc.data, np.float32)

    h1, h2 = (read(h) for h in halves)
    # the tuning's run on half map 1 at the chosen strength is this run
    is_half1 = np.array_equal(data, h1)
    try:
        noise, noise_why = map_noise(data, h1, h2)
    except MemoryError:
        noise, noise_why = None, "not enough memory to compare it with the half maps"
    del h1, h2
    if given is not None:
        return given, "given", noise, noise_why, None

    # tuning a large map takes many minutes: keep the answer, keyed by the half
    # maps' paths, sizes and modification times (a changed half map is tuned again)
    @helicon.cache(
        cache_dir=str(helicon.cache_dir / "proc3d"),
        ignore=["n_jobs", "tile_size", "keep"],
        expires_after=90,
    )
    def tuned(
        halves, stamps, precision, noise_from_halves, n_jobs, tile_size, keep=None
    ):
        return tune_strength(
            *(read(h) for h in halves),
            workers=n_jobs,
            precision=precision,
            tile_size=tile_size,
            noise_from_halves=noise_from_halves,
            keep=keep,
        )

    halves = [str(Path(h).resolve()) for h in halves]
    stamps = [(Path(h).stat().st_size, Path(h).stat().st_mtime_ns) for h in halves]
    key = dict(
        halves=halves,
        stamps=stamps,
        precision=precision,
        noise_from_halves=noise is not None,
    )
    cached = tuned.is_cached(n_jobs=n_jobs, tile_size=tile_size, **key)
    if getattr(args, "verbose", 0) and not cached:
        logger.info("\tdenoiseCurvelet: tuning the strength on %s and %s", *halves)
    runs = {}

    def keep(s, score, denoised, info):
        if is_half1 and (not runs or score < runs["score"]):
            runs.update(strength=s, score=score, done=(denoised, info))

    best, scores = tuned(n_jobs=n_jobs, tile_size=tile_size, keep=keep, **key)
    tried = ", ".join(f"{s:g}: {v:.4g}" for s, v in sorted(scores.items()))
    how = "found in the cache" if cached else "tuned on the half maps"
    done = runs["done"] if runs.get("strength") == best else None
    return best, f"{how}; error per strength {tried}", noise, noise_why, done
