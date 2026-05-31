"""
End-to-end integration test: mock micrograph → CryoSPARC .cs → helicon → RELION 5.0 extraction.

Verifies:
- Y coordinate is NOT inverted (both CS and RELION use top-left origin)
- RELION 5.0 extracts boxes at correct positions
- Cross-correlation between RELION and ground-truth is ~1.0

Skipped if RELION 5.0 or cryosparc-tools are unavailable.
"""

import subprocess
import shutil
from pathlib import Path
import numpy as np
import pytest

N = 4
H, W = 200, 400
BOX = 48
APIX = 1.0

CS_FRACS = np.array([
    [0.15, 0.20],
    [0.35, 0.45],
    [0.65, 0.55],
    [0.85, 0.80],
])

RELION_ENV = "relion-5.0"


def _relion_available():
    try:
        r = subprocess.run(
            ["conda", "run", "-n", RELION_ENV, "which", "relion_preprocess"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _cryosparc_tools_available():
    try:
        from cryosparc.dataset import Dataset  # noqa: F401
        return True
    except ImportError:
        return False


requires_relion = pytest.mark.skipif(
    not _relion_available(),
    reason=f"conda env '{RELION_ENV}' with relion_preprocess not available",
)
requires_cryosparc = pytest.mark.skipif(
    not _cryosparc_tools_available(),
    reason="cryosparc-tools (cryosparc.dataset) not installed",
)


def _place_gaussians(img, centers, sigma=3.0, amp=200.0):
    for i, (xf, yf) in enumerate(centers):
        cx = xf * W
        cy = yf * H
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        img += amp * (i + 1) * g.astype(np.float32)


class TestRelion5Extraction:
    """End-to-end test: CryoSPARC → helicon → RELION 5.0 extraction."""

    @requires_cryosparc
    @requires_relion
    def test_relion5_extraction_cc_1(self, tmp_path):
        """Extract particles via RELION 5.0 and verify CC with ground truth."""
        td = Path(tmp_path)
        mrc = td / "mock.mrc"
        cs = td / "mock.cs"
        mics_star = td / "micrographs.star"
        coord_star = td / "mock_picked.star"
        extract_dir = td / "extracted"
        extract_star = td / "extracted.star"

        # --- Step 1: Create mock micrograph ---
        rng = np.random.RandomState(42)
        img = rng.normal(0, 1, (H, W)).astype(np.float32)
        _place_gaussians(img, CS_FRACS)
        import mrcfile
        mrcfile.new(mrc, overwrite=True).set_data(img)

        # --- Step 2: Create CryoSPARC .cs ---
        from cryosparc.dataset import Dataset as DS
        d = DS(N)
        d.add_fields(
            ["blob/path", "blob/idx", "blob/psize_A",
             "location/center_x_frac", "location/center_y_frac",
             "location/micrograph_shape", "location/micrograph_path",
             "ctf/accel_kv", "ctf/cs_mm"],
            dtypes=["O", "i8", "f8", "f8", "f8", ("f8", (2,)), "O", "f8", "f8"],
        )
        p = str(mrc)
        d["blob/path"] = np.array([p] * N)
        d["blob/idx"] = np.arange(N, dtype=np.int64)
        d["blob/psize_A"] = np.full(N, APIX)
        d["location/center_x_frac"] = CS_FRACS[:, 0]
        d["location/center_y_frac"] = CS_FRACS[:, 1]
        d["location/micrograph_shape"] = np.tile([H, W], (N, 1)).astype(np.float64)
        d["location/micrograph_path"] = np.array([p] * N)
        d["ctf/accel_kv"] = np.full(N, 300.0)
        d["ctf/cs_mm"] = np.full(N, 2.7)
        d.save(str(cs))

        # --- Step 3: helicon conversion ---
        from helicon.lib.io import images2dataframe
        helicon_df = images2dataframe(str(cs), target_convention="relion")
        for i in range(N):
            xf, yf = CS_FRACS[i]
            exp_x = round(xf * W, 2)
            exp_y = round(yf * H, 2)
            assert abs(helicon_df.iloc[i]["rlnCoordinateX"] - exp_x) < 0.01
            assert abs(helicon_df.iloc[i]["rlnCoordinateY"] - exp_y) < 0.01

        # --- Step 4: Write micrograph STAR (relative MRC path) ---
        with open(mics_star, "w") as f:
            f.write(
                "# version 30001\n"
                "\n"
                "data_optics\n"
                "\n"
                "loop_\n"
                "_rlnOpticsGroup #1\n"
                "_rlnVoltage #2\n"
                "_rlnSphericalAberration #3\n"
                "_rlnImagePixelSize #4\n"
                "_rlnMicrographOriginalPixelSize #5\n"
                f"1\t300.0\t2.7\t{APIX}\t{APIX}\n"
                "\n"
                "data_micrographs\n"
                "\n"
                "loop_\n"
                "_rlnMicrographName #1\n"
                "_rlnOpticsGroup #2\n"
                f"{mrc.name}\t1\n"
            )

        # --- Step 5: Write coordinate STAR ---
        with open(coord_star, "w") as f:
            f.write("data_\n\nloop_\n_rlnCoordinateX\n_rlnCoordinateY\n")
            for _, row in helicon_df.iterrows():
                f.write(f"{row['rlnCoordinateX']}\t{row['rlnCoordinateY']}\n")

        # --- Step 6: RELION extraction (run from tmp_path for relative paths) ---
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        # Use relative paths so RELION resolves them under tmp_path
        mics_star_rel = mics_star.relative_to(td)
        coord_dir_rel = Path(".")
        extract_star_rel = extract_star.relative_to(td)
        extract_dir_rel = extract_dir.name
        cmd = [
            "conda", "run", "-n", RELION_ENV,
            "relion_preprocess",
            "--i", str(mics_star_rel),
            "--coord_suffix", "_picked.star",
            "--coord_dir", str(coord_dir_rel),
            "--extract",
            "--extract_size", str(BOX),
            "--part_star", str(extract_star_rel),
            "--part_dir", str(extract_dir_rel),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(td), timeout=120)
        if r.returncode:
            pytest.fail(f"relion_preprocess failed:\n{r.stderr[-800:]}\n{r.stdout[-800:]}")

        import starfile
        sd = starfile.read(str(extract_star))
        if isinstance(sd, dict):
            for k in ("particles", "data_particles"):
                if k in sd:
                    relion_df = sd[k]
                    break
            else:
                relion_df = next(iter(sd.values()))
        else:
            relion_df = sd
        assert len(relion_df) == N, f"Expected {N} particles, got {len(relion_df)}"

        # --- Step 7: Python extraction at same coordinates ---
        with mrcfile.open(mrc) as m:
            micro_img = m.data.astype(np.float32)
        half = BOX // 2
        py_patches = []
        for _, row in helicon_df.iterrows():
            rx = int(round(row["rlnCoordinateX"]))
            ry = int(round(row["rlnCoordinateY"]))
            p = micro_img[ry - half:ry + half, rx - half:rx + half]
            assert p.shape == (BOX, BOX)
            py_patches.append(p)

        # --- Step 8: Cross-correlate ---
        ccs = []
        for _, row in relion_df.iterrows():
            idx_str, relpath = row["rlnImageName"].split("@", 1)
            frame = int(idx_str) - 1
            img_path = td / relpath
            with mrcfile.open(img_path) as m:
                relion_patch = m.data[frame].astype(np.float32)
            cc = np.corrcoef(py_patches[frame].ravel(), relion_patch.ravel())[0, 1]
            ccs.append(cc)

        min_cc = min(ccs)
        assert min_cc > 0.9999, f"Min CC = {min_cc:.6f}, expected > 0.9999"
