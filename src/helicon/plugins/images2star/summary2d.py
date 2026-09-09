"""Handler and PDF gallery helpers for the summary2D images2star option."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
import tempfile
import textwrap

import numpy as np

from helicon.lib.exceptions import (
    HeliconError,
    HeliconFileExistsError,
    HeliconValidationError,
)


option_name = "summary2D"


def add_args(parser):
    parser.add_argument(
        "--summary2D", action="store_true",
        help="write a PDF gallery of the latest iteration of successful RELION Class2D jobs",
    )
    parser.add_argument(
        "--jobs", nargs="+", action="extend", metavar="JOB",
        help=(
            "jobs to summarize: folder names/aliases, numbers, or inclusive numeric "
            "ranges (e.g. --jobs job001 good_classes or --jobs 1,3,5-9). "
            "Default: all jobs; requires --summary2D"
        ),
    )


def check_args(args, parser, all_options):
    """Validate this standalone mode before the CLI loads a particle dataset."""
    if not getattr(args, option_name, False):
        if getattr(args, "jobs", None) is not None:
            raise HeliconValidationError("--jobs requires --summary2D")
        return

    allowed = {
        "summary2D", "jobs", "force", "verbose", "cpu", "ppid",
        "input_imageFiles", "output_starFile",
    }
    # Also inspect parsed values so direct Python callers are validated.
    incompatible = {
        action.dest for action in parser._actions
        if action.option_strings and action.dest not in allowed
        and action.dest != "help"
        and getattr(args, action.dest, action.default) != action.default
    }
    incompatible.update(o for o in all_options if o not in allowed)
    if incompatible:
        raise HeliconValidationError(
            "--summary2D cannot be combined with "
            + ", ".join(f"--{o}" for o in sorted(incompatible))
        )
    if len(args.input_imageFiles) != 1 or not Path(args.input_imageFiles[0]).is_dir():
        raise HeliconValidationError("--summary2D requires one Class2D directory")
    if Path(args.output_starFile).suffix.lower() != ".pdf":
        raise HeliconValidationError("--summary2D requires a .pdf output file")
    if Path(args.output_starFile).exists() and args.force != 1:
        raise HeliconFileExistsError(
            f"the output file ({args.output_starFile}) exists. Use --force=1 to overwrite it"
        )


def handle(data, args, index_d, param):
    """Write the summary without loading or transforming a particle dataset."""
    if not param:
        return data, index_d
    summarize_class2d(
        args.input_imageFiles[0],
        args.output_starFile,
        jobs=args.jobs,
        overwrite=args.force == 1,
        logger=logging.getLogger("helicon.commands.images2star"),
    )
    index_d[option_name] += 1
    return data, index_d


@dataclass
class ClassStack:
    path: Path
    counts: np.ndarray


@dataclass
class Class2DJob:
    path: Path
    aliases: list[str] = field(default_factory=list)
    stacks: list[ClassStack] = field(default_factory=list)


@dataclass
class Class2DSummary:
    jobs: list[Class2DJob] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _natural_key(value):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", str(value))]


def discover_jobs(folder: Path, selectors: list[str] | None = None) -> list[Class2DJob]:
    """Select immediate directories by name, number, or range; collapse aliases.

    Numeric ranges match existing jobNNN directories (including alias targets),
    so gaps are allowed. An explicit name/number or range matching nothing is an
    error. Commas and whitespace both separate selectors.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise HeliconValidationError(f"Not a Class2D directory: {folder}")
    entries = sorted(
        (p for p in folder.iterdir() if p.is_dir() or p.is_symlink()),
        key=lambda p: _natural_key(p.name),
    )
    canonical = {}
    for entry in entries:
        try:
            canonical[entry] = entry.resolve()
        except (OSError, RuntimeError):
            # Preserve broken/cyclic links as candidates for the skip report.
            canonical[entry] = entry.absolute()

    def job_number(entry):
        for name in (canonical[entry].name, entry.name):
            match = re.fullmatch(r"job(\d+)", name)
            if match:
                return int(match[1])
        return None

    selected = entries if selectors is None else []
    names = {p.name for p in entries}
    for selector in selectors or []:
        # A quoted folder name may itself contain spaces or commas.
        tokens = [selector] if selector in names else re.split(r"[,\s]+", selector.strip())
        for token in tokens:
            if not token:
                continue
            interval = re.fullmatch(r"(\d+)-(\d+)", token)
            if token in names:
                matches = [p for p in entries if p.name == token]
            elif interval:
                first, last = map(int, interval.groups())
                if first > last:
                    raise HeliconValidationError(f"Reversed job range: {token}")
                matches = [p for p in entries
                           if (n := job_number(p)) is not None and first <= n <= last]
            elif token.isdigit():
                matches = [p for p in entries if job_number(p) == int(token)]
            else:
                matches = [p for p in entries if p.name == token]
            if not matches:
                raise HeliconValidationError(f"No Class2D jobs match {token!r} in {folder}")
            selected.extend(matches)
    if not selected:
        raise HeliconValidationError(f"No Class2D jobs selected in {folder}")

    jobs = {}
    for entry in selected:
        path = canonical[entry]
        jobs.setdefault(path, Class2DJob(path))
    # Include all discovered aliases, even when selection used a job number.
    for entry in entries:
        job = jobs.get(canonical[entry])
        if job is not None and entry.name != job.path.name:
            job.aliases.append(entry.name)
    return sorted(jobs.values(), key=lambda job: _natural_key(str(job.path)))


def _latest_stacks(folder: Path) -> list[Path]:
    stacks = sorted(folder.glob("*_classes.mrcs"), key=lambda p: _natural_key(p.name))
    numbered = []
    for path in stacks:
        match = re.search(r"_it(\d+)_classes\.mrcs$", path.name)
        if match:
            numbered.append((int(match[1]), path))
    if numbered:
        latest = max(number for number, _ in numbered)
        stacks = [path for number, path in numbered if number == latest]
    # A stack may itself have a symbolic-link alias.
    unique = {}
    for path in sorted(stacks, key=lambda p: p.is_symlink()):
        unique.setdefault(path.resolve(), path)
    if not unique:
        raise ValueError("no *_classes.mrcs outputs")
    return list(unique.values())


def _particle_counts(path: Path, class_count: int) -> np.ndarray:
    import pandas as pd
    import starfile

    tables = starfile.read(path, always_dict=True)
    candidates = [table for table in tables.values()
                  if isinstance(table, pd.DataFrame) and "rlnClassNumber" in table]
    if len(candidates) != 1:
        raise ValueError(f"{path.name}: expected one particle table with rlnClassNumber")
    numbers = pd.to_numeric(candidates[0]["rlnClassNumber"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.all(np.isfinite(numbers) & (numbers == np.floor(numbers))
                  & (numbers >= 1) & (numbers <= class_count)):
        raise ValueError(f"{path.name}: invalid rlnClassNumber (expected 1-{class_count})")
    return np.bincount(numbers.astype(np.int64), minlength=class_count + 1)[1:]


def collect_summary(folder: Path, selectors: list[str] | None = None) -> Class2DSummary:
    """Read exact assignment counts; skip unsuccessful or unusable jobs."""
    import mrcfile

    summary = Class2DSummary()
    for job in discover_jobs(folder, selectors):
        try:
            if not (job.path / "RELION_JOB_EXIT_SUCCESS").is_file():
                raise ValueError("missing RELION_JOB_EXIT_SUCCESS (failed or unfinished)")
            for stack in _latest_stacks(job.path):
                with mrcfile.mmap(stack, mode="r") as mrc:
                    if mrc.data is None or mrc.data.ndim not in (2, 3) or not mrc.data.size:
                        raise ValueError(f"{stack.name}: not a nonempty 2D image stack")
                    count = 1 if mrc.data.ndim == 2 else mrc.data.shape[0]
                metadata = stack.with_name(stack.name.removesuffix("_classes.mrcs") + "_data.star")
                job.stacks.append(ClassStack(stack, _particle_counts(metadata, count)))
            summary.jobs.append(job)
        except (OSError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            label = job.path.name
            if job.aliases:
                label += " (aliases: " + ", ".join(job.aliases) + ")"
            summary.skipped.append((label, str(exc)))
    return summary


def _overview_lines(summary: Class2DSummary, folder: Path) -> list[str]:
    lines = [f"Source: {folder.resolve()}",
             f"Included: {len(summary.jobs)} jobs. Skipped: {len(summary.skipped)} jobs.",
             "Latest iteration only. Classes ordered by original class number.",
             "Particle counts are assignments per job, not unique across jobs.", ""]
    for job in summary.jobs:
        lines.append(job.path.name)
        if job.aliases:
            lines.append("  Aliases: " + ", ".join(job.aliases))
        for stack in job.stacks:
            lines.append(f"  {stack.path.name}: {len(stack.counts)} classes; "
                         f"{int(stack.counts.sum()):,} particles")
    if summary.skipped:
        lines.extend(["", "Skipped jobs"])
        for name, reason in summary.skipped:
            lines.append(f"{name}: {reason}")
    return [wrapped for line in lines
            for wrapped in (textwrap.wrap(line, width=95, subsequent_indent="  ") or [""])]


def _class_limits(image):
    finite = image[np.isfinite(image)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1, 99])
    if low == high:
        low, high = float(finite.min()), float(finite.max())
    if low == high:
        high = low + 1.0
    return low, high


def _render_pdf(summary: Class2DSummary, folder: Path, output: Path) -> None:
    import mrcfile
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    lines = _overview_lines(summary, folder)
    overview_pages = [lines[i:i + 48] for i in range(0, len(lines), 48)]
    total_pages = len(overview_pages) + sum(
        (len(stack.counts) + 19) // 20 for job in summary.jobs for stack in job.stacks
    )
    page = 0
    with PdfPages(output, metadata={"Title": "RELION Class2D summary",
                                    "Creator": "Helicon images2star"}) as pdf:
        def save(figure):
            nonlocal page
            page += 1
            figure.text(0.5, 0.025, f"{page} / {total_pages}", ha="center", fontsize=9)
            pdf.savefig(figure, dpi=200)
            figure.clear()

        for page_lines in overview_pages:
            figure = Figure(figsize=(8.27, 11.69))
            figure.text(0.06, 0.95, "RELION Class2D summary", fontsize=18, weight="bold")
            figure.text(0.06, 0.91, "\n".join(page_lines), fontsize=9,
                        va="top", linespacing=1.45, parse_math=False)
            save(figure)

        for job in summary.jobs:
            for stack in job.stacks:
                with mrcfile.mmap(stack.path, mode="r") as mrc:
                    for first in range(0, len(stack.counts), 20):
                        figure = Figure(figsize=(8.27, 11.69))
                        figure.text(0.06, 0.965, textwrap.shorten(job.path.name, width=65),
                                    fontsize=15, weight="bold", parse_math=False)
                        figure.text(0.06, 0.94,
                                    "\n".join(textwrap.wrap(stack.path.name, width=95)),
                                    fontsize=9, va="top", parse_math=False)
                        axes = figure.subplots(5, 4, squeeze=False)
                        figure.subplots_adjust(left=0.05, right=0.95, bottom=0.06,
                                               top=0.89, wspace=0.18, hspace=0.40)
                        for offset, ax in enumerate(axes.flat):
                            ax.set_axis_off()
                            index = first + offset
                            if index >= len(stack.counts):
                                continue
                            pixels = mrc.data if mrc.data.ndim == 2 else mrc.data[index]
                            low, high = _class_limits(pixels)
                            ax.imshow(pixels, cmap="gray", origin="lower", vmin=low,
                                      vmax=high, interpolation="nearest")
                            label = textwrap.shorten(job.path.name, width=23, placeholder="...")
                            ax.text(0.5, -0.04,
                                    f"{label} / class {index + 1}\n"
                                    f"{int(stack.counts[index]):,} particles",
                                    transform=ax.transAxes, ha="center", va="top",
                                    fontsize=7, parse_math=False)
                        save(figure)


def summarize_class2d(
    folder: str | Path,
    output: str | Path,
    jobs: list[str] | None = None,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> Class2DSummary:
    """Write the summary atomically, returning included and skipped job details."""
    folder, output = Path(folder), Path(output)
    logger = logger or logging.getLogger(__name__)
    if output.suffix.lower() != ".pdf":
        raise HeliconValidationError("Class2D summary output must be a .pdf file")
    if output.exists() and not overwrite:
        raise HeliconFileExistsError(f"{output} exists. Use --force=1 to overwrite it")
    summary = collect_summary(folder, jobs)
    for name, reason in summary.skipped:
        logger.warning("Skipping %s: %s", name, reason)
    if not summary.jobs:
        raise HeliconError("No successful Class2D jobs with readable class stacks and particle metadata")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pdf", delete=False) as temp:
            temp_path = Path(temp.name)
        _render_pdf(summary, folder, temp_path)
        if output.exists() and not overwrite:
            raise HeliconFileExistsError(f"{output} exists. Use --force=1 to overwrite it")
        os.replace(temp_path, output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    logger.info("Summarized %d Class2D jobs in %s (%d skipped)",
                len(summary.jobs), output, len(summary.skipped))
    return summary
