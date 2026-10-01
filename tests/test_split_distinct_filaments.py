"""helicon.split_distinct_filaments: separate filaments that share one id.

Only the distance from the helical axis splits a filament; a gap along it,
however long, never does -- class selections leave many of those.
"""

import numpy as np
import pandas as pd
import pytest

import helicon

APIX = 1.0  # Å per micrograph pixel
STEP = 14.0  # inter-box distance, Å


def _filament(x0, y0, angle_deg, n, tube=1, mic="m1", start=0, keep=None, bend=0.0):
    """``n`` segments along a line (or a gentle arc, ``bend`` rad/Å)."""
    s = (np.arange(n) + start) * STEP
    a = np.deg2rad(angle_deg) + bend * s
    if bend:
        x = x0 + np.cumsum(np.r_[0, np.cos(a[1:]) * STEP])
        y = y0 + np.cumsum(np.r_[0, np.sin(a[1:]) * STEP])
    else:
        x, y = x0 + s * np.cos(a), y0 + s * np.sin(a)
    df = pd.DataFrame(
        dict(
            rlnMicrographName=mic,
            rlnHelicalTubeID=tube,
            rlnCoordinateX=x / APIX,
            rlnCoordinateY=y / APIX,
            rlnHelicalTrackLengthAngst=s,
            rlnAnglePsiPrior=-np.rad2deg(a),
        )
    )
    return df if keep is None else df.iloc[keep]


def _split(df, **kw):
    return helicon.split_distinct_filaments(df, apix_micrograph=APIX, **kw)


class TestWhatSplits:
    def test_two_filaments_sharing_an_id_are_separated(self):
        a = _filament(100, 100, 10, 40)
        b = _filament(3000, 1800, -30, 30, start=3)  # same id, another extraction
        df = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=0)
        out = _split(df)
        ids_a = set(out.loc[out.index < 40, "rlnHelicalTubeID"])
        ids_b = set(out.loc[out.index >= 40, "rlnHelicalTubeID"])
        assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b
        assert ids_a == {1}  # the first along the track keeps the id
        assert ids_b == {2}

    def test_parallel_filaments_side_by_side_are_separated(self):
        a = _filament(0, 0, 0, 40)
        b = _filament(0, 120, 0, 40)  # 120 Å apart, same direction
        out = _split(pd.concat([a, b], ignore_index=True))
        assert out["rlnHelicalTubeID"].nunique() == 2

    def test_crossing_filaments_are_not_joined_at_the_crossing(self):
        a = _filament(0, 200, 0, 40)
        b = _filament(280, -100, 90, 40)  # crosses a near its middle
        out = _split(pd.concat([a, b], ignore_index=True))
        assert out["rlnHelicalTubeID"].nunique() == 2
        assert out["rlnHelicalTubeID"].iloc[:40].nunique() == 1


class TestWhatDoesNot:
    def test_gaps_along_the_axis_are_left_alone(self):
        # class selection kept 1 segment in 3, and lost a long stretch
        keep = [i for i in range(120) if i % 3 == 0 and not 40 <= i < 90]
        df = _filament(0, 0, 25, 120, keep=keep)
        assert (_split(df)["rlnHelicalTubeID"] == 1).all()

    def test_a_curved_filament_stays_whole(self):
        df = _filament(0, 0, 0, 150, bend=1 / 3000)  # ~40 degrees over 2 µm
        assert (_split(df)["rlnHelicalTubeID"] == 1).all()

    def test_separate_ids_are_untouched(self):
        a = _filament(0, 0, 0, 20, tube=1)
        b = _filament(0, 500, 0, 20, tube=2)
        df = pd.concat([a, b], ignore_index=True)
        pd.testing.assert_frame_equal(_split(df), df)

    def test_only_the_id_column_changes(self):
        df = pd.concat([_filament(0, 0, 0, 20), _filament(0, 900, 0, 20)])
        out = _split(df)
        others = [c for c in df.columns if c != "rlnHelicalTubeID"]
        pd.testing.assert_frame_equal(out[others], df[others])


class TestIds:
    def test_new_ids_follow_the_largest_of_the_micrograph(self):
        a = _filament(0, 0, 0, 20, tube=3)
        b = _filament(0, 800, 0, 20, tube=3)
        c = _filament(0, 1600, 0, 20, tube=7)
        other = _filament(0, 0, 0, 20, tube=3, mic="m2")
        df = pd.concat([a, b, c, other], ignore_index=True)
        out = _split(df)
        assert sorted(
            out.loc[out.rlnMicrographName == "m1", "rlnHelicalTubeID"].unique()
        ) == [3, 7, 8]
        assert (out.loc[out.rlnMicrographName == "m2", "rlnHelicalTubeID"] == 3).all()


class TestOutput:
    def test_the_optics_table_is_kept(self):
        df = pd.concat([_filament(0, 0, 0, 20), _filament(0, 900, 0, 20)])
        df.attrs["optics"] = pd.DataFrame(dict(rlnOpticsGroup=[1]))
        out = _split(df)
        assert out["rlnHelicalTubeID"].nunique() == 2
        assert "optics" in out.attrs

    def test_helicalpitch_columns_follow_the_new_ids(self):
        from helicon.webApps.lib import helical_pitch_compute as compute

        df = pd.concat([_filament(0, 0, 0, 20), _filament(0, 900, 0, 30)])
        df = compute._annotate_helix_ids(df.reset_index(drop=True))
        out, n0, n1 = compute.split_distinct_filaments(df, 50)
        assert (n0, n1) == (1, 2)
        assert out["helixID"].nunique() == 2
        assert sorted(out.groupby("helixID")["length"].first()) == [266, 406]
        same, *_ = compute.split_distinct_filaments(df, 0)
        assert same is df


class TestInputs:
    def test_pixel_size_is_estimated_from_the_track_length(self):
        df = pd.concat([_filament(0, 0, 0, 30), _filament(0, 400, 0, 30)])
        df["rlnCoordinateX"] *= 2.0  # coordinates in 0.5 Å pixels
        df["rlnCoordinateY"] *= 2.0
        out = helicon.split_distinct_filaments(df)
        assert out["rlnHelicalTubeID"].nunique() == 2

    def test_missing_columns_are_reported(self):
        with pytest.raises(KeyError, match="rlnCoordinateY"):
            helicon.split_distinct_filaments(
                pd.DataFrame(
                    dict(
                        rlnMicrographName=["m"],
                        rlnHelicalTubeID=[1],
                        rlnCoordinateX=[0.0],
                    )
                )
            )
