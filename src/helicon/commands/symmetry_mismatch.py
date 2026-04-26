#!/usr/bin/env python

"""A command line tool that reassigns particles to C1 poses from two jobs of the same set of particles but with different symmetries."""

import argparse
import sys
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import helicon


def main(args):
    helicon.log_command_line()

    cs = None
    project = None
    if args.projectID:
        cs = helicon.connect_cryosparc()
        project = cs.find_project(args.projectID)

    # Load dataset 1
    if args.input1:
        print(f"Loading local file: {args.input1}")
        ds1 = load_cryosparc_cs_file(args.input1, args.pass_through1)
        sym1 = args.sym1
        if not sym1:
            helicon.color_print(
                "ERROR: --sym1 is required when loading dataset 1 from local file.",
                color="red",
            )
            sys.exit(-1)
        job1_id = Path(args.input1).stem
        job1 = None
    else:
        if not project or not args.jobID1:
            helicon.color_print(
                "ERROR: --projectID and --jobID1 are required if --input1 is not provided.",
                color="red",
            )
            sys.exit(-1)
        job1 = project.find_job(args.jobID1)
        sym1 = args.sym1 or get_job_symmetry(job1)
        out_name1 = get_particle_output_name(job1)
        if not out_name1:
            helicon.color_print(
                f"ERROR: Could not find particle output in job {args.jobID1}",
                color="red",
            )
            sys.exit(-1)
        print(f"Loading output: {args.jobID1} ({out_name1})")
        ds1 = job1.load_output(out_name1)
        job1_id = args.jobID1

    if args.verbose > 0:
        print(f"Dataset 1: {len(ds1):,} particles. sym={sym1}")

    # Load dataset 2
    if args.input2:
        print(f"Loading local file: {args.input2}")
        ds2 = load_cryosparc_cs_file(args.input2, args.pass_through2)
        sym2 = args.sym2
        if not sym2:
            helicon.color_print(
                "ERROR: --sym2 is required when loading dataset 2 from local file.",
                color="red",
            )
            sys.exit(-1)
        job2_id = Path(args.input2).stem
        job2 = None
    else:
        if not project or not args.jobID2:
            helicon.color_print(
                "ERROR: --projectID and --jobID2 are required if --input2 is not provided.",
                color="red",
            )
            sys.exit(-1)
        job2 = project.find_job(args.jobID2)
        sym2 = args.sym2 or get_job_symmetry(job2)
        out_name2 = get_particle_output_name(job2)
        if not out_name2:
            helicon.color_print(
                f"ERROR: Could not find particle output in job {args.jobID2}",
                color="red",
            )
            sys.exit(-1)
        print(f"Loading output: {args.jobID2} ({out_name2})")
        ds2 = job2.load_output(out_name2)
        job2_id = args.jobID2

    if args.verbose > 0:
        print(f"Dataset 2: {len(ds2):,} particles. sym={sym2}")

    # Generate pre-matching diagnostic plots (if verbose)
    if args.verbose > 1:
        log_name = (
            f"log_symmetry_mismatch_{args.projectID or 'local'}_{job1_id}_{job2_id}"
        )
        log_dir = Path(log_name)
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            import plotly.express as px
            import pandas as pd
            from collections import Counter

            # Find common micrographs
            uids1 = set(ds1["location/micrograph_uid"])
            uids2 = set(ds2["location/micrograph_uid"])
            common_mic_uids = list(uids1.intersection(uids2))

            if common_mic_uids:
                # Count particles per micrograph efficiently
                counts1 = Counter(ds1["location/micrograph_uid"])
                counts2 = Counter(ds2["location/micrograph_uid"])

                # Find the most populated common micrograph
                best_mic_uid = max(
                    common_mic_uids, key=lambda uid: counts1[uid] + counts2[uid]
                )

                # Extract locations
                mask1 = ds1["location/micrograph_uid"] == best_mic_uid
                mask2 = ds2["location/micrograph_uid"] == best_mic_uid

                df1 = pd.DataFrame(
                    {
                        "x": ds1["location/center_x_frac"][mask1],
                        "y": ds1["location/center_y_frac"][mask1],
                        "Dataset": f"Job {job1_id}",
                    }
                )
                df2 = pd.DataFrame(
                    {
                        "x": ds2["location/center_x_frac"][mask2],
                        "y": ds2["location/center_y_frac"][mask2],
                        "Dataset": f"Job {job2_id}",
                    }
                )
                df = pd.concat([df1, df2])

                fig = px.scatter(
                    df,
                    x="x",
                    y="y",
                    color="Dataset",
                    title=f"Particle Locations for Micrograph {best_mic_uid}",
                    labels={"x": "Center X (frac)", "y": "Center Y (frac)"},
                    opacity=0.5,
                )
                fig.update_yaxes(autorange="reversed")  # Standard for cryo-EM images
                fig.write_image(log_dir / "micrograph_locations_scatter.png")
                print(
                    f"Saved micrograph location plot for {best_mic_uid} ({counts1[best_mic_uid] + counts2[best_mic_uid]} particles)"
                )

        except ImportError:
            pass  # Will be handled by the next plotting block if still missing
        except Exception as e:
            helicon.color_print(
                f"WARNING: Failed to save pre-matching plots: {e}", color="yellow"
            )

    # Find corresponding particles
    if args.verbose > 0:
        print(
            f"Finding particle correspondence (dist_tol={args.dist_tol} Angstrom, axis_tol={args.axis_tol} degrees)..."
        )
    matches = find_particle_correspondence(
        ds1, ds2, dist_tol=args.dist_tol, axis_tol=args.axis_tol, verbose=args.verbose
    )

    if len(matches) == 0:
        helicon.color_print(
            "ERROR: No corresponding particles found between the two jobs.", color="red"
        )
        sys.exit(-1)

    if args.verbose > 0:
        print(f"Found {len(matches):,} corresponding particles.")

    # Filter datasets to keep only matched particles
    matched_uids1 = matches[:, 0]
    matched_uids2 = matches[:, 1]

    # Filter and align datasets based on the match order
    # find_particle_correspondence returns (uid1, uid2) pairs.
    # We ensure ds1[i] corresponds to ds2[i] by taking entries in matched order.

    # For ds1:
    uid_to_idx1 = {uid: i for i, uid in enumerate(ds1["uid"])}
    ds1 = ds1.take([uid_to_idx1[uid] for uid in matched_uids1])

    # For ds2:
    uid_to_idx2 = {uid: i for i, uid in enumerate(ds2["uid"])}
    ds2 = ds2.take([uid_to_idx2[uid] for uid in matched_uids2])

    # Get poses
    pose_attr1 = helicon.first_matched_attr(
        ds1, ["alignments3D_multi/pose", "alignments3D/pose"]
    )
    pose_attr2 = helicon.first_matched_attr(
        ds2, ["alignments3D_multi/pose", "alignments3D/pose"]
    )

    if not pose_attr1 or not pose_attr2:
        helicon.color_print(
            f"ERROR: Could not find pose information in one or both datasets.",
            color="red",
        )
        sys.exit(-1)

    R1 = convert_cryosparc_pose_to_scipy_Rotation(ds1[pose_attr1])
    R2 = convert_cryosparc_pose_to_scipy_Rotation(ds2[pose_attr2])

    euler1 = R1.as_euler("ZXZ", degrees=True)
    euler2 = R2.as_euler("ZXZ", degrees=True)

    rot1 = euler1[:, 0]
    rot2 = euler2[:, 0]

    # Diagnostic Plots (if verbose)
    if args.verbose > 1:
        log_name = (
            f"log_symmetry_mismatch_{args.projectID or 'local'}_{job1_id}_{job2_id}"
        )
        log_dir = Path(log_name)
        # log_dir already created above
        print(f"Saving diagnostic plots to {log_dir}...")

        try:
            import plotly.express as px

            # Histogram of differences
            delta_angles = helicon.angular_difference(rot1, rot2)
            fig_hist = px.histogram(
                x=delta_angles,
                nbins=300,
                title="Rotational Angle Differences",
                labels={"x": f"Differences (°, {sym1} - {sym2})"},
            )
            fig_hist.update_layout(yaxis_title="Frequency")
            fig_hist.write_image(log_dir / "rotation_difference_histogram.png")

            # Scatter plot
            fig_scatter = px.scatter(
                x=rot1,
                y=rot2,
                labels={"x": f"{sym1} Rotation (°)", "y": f"{sym2} Rotation (°)"},
                title=f"Rotation Correlation: {sym1} vs {sym2}",
            )
            fig_scatter.write_image(log_dir / "rotation_correlation_scatter.png")

            # Location Scatter plots
            fig_x = px.scatter(
                x=ds1["location/center_x_frac"],
                y=ds2["location/center_x_frac"],
                labels={"x": "Dataset 1 X (frac)", "y": "Dataset 2 X (frac)"},
                title="Location X Correlation",
            )
            fig_x.write_image(log_dir / "location_x_correlation_scatter.png")

            fig_y = px.scatter(
                x=ds1["location/center_y_frac"],
                y=ds2["location/center_y_frac"],
                labels={"x": "Dataset 1 Y (frac)", "y": "Dataset 2 Y (frac)"},
                title="Location Y Correlation",
            )
            fig_y.write_image(log_dir / "location_y_correlation_scatter.png")

        except ImportError:
            helicon.color_print(
                "WARNING: plotly or kaleido not found. Skipping diagnostic plots.",
                color="yellow",
            )
        except Exception as e:
            helicon.color_print(
                f"WARNING: Failed to save diagnostic plots: {e}", color="yellow"
            )

    # Symmetry expansion integers
    n1 = int(sym1[1:]) if sym1[0].upper() == "C" and sym1[1:].isdigit() else 1
    n2 = int(sym2[1:]) if sym2[0].upper() == "C" and sym2[1:].isdigit() else 1

    estimated_relative_angle, rot1_unfolded, rot2_unfolded = solve_symmetry_mismatch(
        rot1, rot2, n1, n2, num_seed_samples=10, verbose=args.verbose
    )

    if args.verbose > 0:
        print(
            f"Optimal relative rotation between {sym1} and {sym2}: {estimated_relative_angle:.4f}°"
        )

    euler1_to_c1 = euler1.copy()
    euler1_to_c1[:, 0] = rot1_unfolded

    euler2_to_c1 = euler2.copy()
    euler2_to_c1[:, 0] = rot2_unfolded

    # Update datasets
    ds1_out = ds1.copy()
    ds2_out = ds2.copy()

    pose_out1 = convert_euler_angles_to_cryosparc_pose(euler1_to_c1)
    pose_out2 = convert_euler_angles_to_cryosparc_pose(euler2_to_c1)

    if ds1[pose_attr1].ndim == 3:
        ds1_out[pose_attr1] = np.expand_dims(pose_out1, axis=1)
    else:
        ds1_out[pose_attr1] = pose_out1

    if ds2[pose_attr2].ndim == 3:
        ds2_out[pose_attr2] = np.expand_dims(pose_out2, axis=1)
    else:
        ds2_out[pose_attr2] = pose_out2

    # Create External Jobs or save to local files
    items_to_save = [
        {
            "ds": ds1_out,
            "sym": sym1,
            "job_id": job1_id,
            "job": job1,
            "output": args.outputFile1,
            "source_out": out_name1 if job1 else None,
        },
        {
            "ds": ds2_out,
            "sym": sym2,
            "job_id": job2_id,
            "job": job2,
            "output": args.outputFile2,
            "source_out": out_name2 if job2 else None,
        },
    ]

    for item in items_to_save:
        if item["output"]:
            if args.verbose > 0:
                print(
                    f"Saving reassigned dataset ({len(item['ds'])} particles) to {item['output']}..."
                )
            item["ds"].save(item["output"])
            continue

        if project and item["job"]:
            workspace_id = args.workspaceID or item["job"].doc["workspace_uids"][-1]

            import shlex

            command_line = shlex.join(sys.argv)

            title = f"Symmetry Mismatch {item['sym']} (from {job1_id}-{sym1} and {job2_id}-{sym2})"

            if args.verbose > 0:
                print(
                    f"Saving reassigned dataset ({len(item['ds'])} particles) to new job in workspace {workspace_id}..."
                )
            ext_job = project.create_external_job(
                workspace_id, title=title, desc=command_line
            )

            # Connect to input to allow passthroughs
            input_name = "particles"
            ext_job.connect(
                target_input=input_name,
                source_job_uid=item["job_id"],
                source_output=item["source_out"],
                title="Input Particles",
            )

            # Determine output slots
            ds = item["ds"]
            output_slots = []
            if "blob/path" in ds.fields():
                output_slots.append("blob")
            if "ctf/path" in ds.fields():
                output_slots.append("ctf")
            if "alignments3D/pose" in ds.fields():
                output_slots.append("alignments3D")
            elif "alignments3D_multi/pose" in ds.fields():
                output_slots.append("alignments3D_multi")

            ext_job.add_output(
                type="particle",
                name="particles_reassigned",
                title=f'Particles {item["sym"]} reassigned',
                slots=output_slots,
                passthrough=input_name,
            )

            ext_job.start()
            ext_job.save_output("particles_reassigned", ds)
            ext_job.stop()
            if args.verbose > 0:
                print(f"Saved to external job {ext_job.uid} for {item['job_id']}")


def load_cryosparc_cs_file(cs_file, pass_through_file=None):
    from cryosparc.dataset import Dataset

    ds = Dataset.load(cs_file)
    if pass_through_file is not None:
        if Path(pass_through_file).exists():
            pt = Dataset.load(pass_through_file)
            ds = ds.innerjoin(pt)
        else:
            print(f"WARNING: pass through file {pass_through_file} does not exist")
    return ds


def convert_cryosparc_pose_to_scipy_Rotation(poses):
    if len(poses.shape) == 3:
        poses = np.squeeze(poses)
    assert len(poses.shape) == 2, f"ERROR: {poses.shape=} should be 2 dimensions"
    assert poses.shape[1] == 3, f"ERROR: {poses.shape[1]=} should be 3"
    return R.from_rotvec(poses)


def convert_euler_angles_to_cryosparc_pose(eulers, convention="ZXZ"):
    r = R.from_euler(seq=convention, angles=eulers, degrees=True)
    poses = r.as_rotvec(degrees=False)
    return poses


def get_particle_output_name(job):
    """Find the first output group name that contains 'particles'."""
    for group in job.doc.get("output_result_groups", []):
        if (
            "particles" in group.get("name", "").lower()
            or "particle" in group.get("type", "").lower()
        ):
            return group["name"]
    return None


def get_job_symmetry(job):
    """Attempt to detect symmetry from job parameters."""
    try:
        sym = job.doc["params_spec"]["refine_symmetry"]["value"]
    except:
        sym = "C1"
    return sym


def angular_distance(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def relative_angle_range(sym1: int, sym2: int):
    return 360.0 * np.gcd(sym1, sym2) / (sym1 * sym2)


def solve_symmetry_mismatch(rot1, rot2, sym1, sym2, num_seed_samples=10, verbose=0):
    import numpy as np

    period1 = 360.0 / sym1
    period2 = 360.0 / sym2
    max_angle = relative_angle_range(sym1, sym2)

    n_samples = len(rot1)

    # 1. Vectorized Candidate Generation
    # Reshape for broadcasting: rot -> (N, 1, 1), k1 -> (1, sym1, 1), k2 -> (1, 1, sym2)
    r1_arr = np.asarray(rot1).reshape(n_samples, 1, 1)
    r2_arr = np.asarray(rot2).reshape(n_samples, 1, 1)

    k1_arr = np.arange(sym1).reshape(1, sym1, 1)
    k2_arr = np.arange(sym2).reshape(1, 1, sym2)

    unfolded_rot1 = r1_arr + k1_arr * period1
    unfolded_rot2 = r2_arr + k2_arr * period2

    # Calculate candidates and keep in [0, 360)
    cands = np.fmod(unfolded_rot2 - unfolded_rot1 + 360.0, 360.0)

    # Flatten the symmetry dimensions so each sample has sym1*sym2 candidates
    # Shape becomes (N, sym1 * sym2)
    cands_flat = cands.reshape(n_samples, -1)

    # 2. Vectorized Consensus Finding
    # Pool candidates from random samples to avoid failing if a localized block is noisy
    num_seed_samples = min(num_seed_samples, n_samples)
    seed_indices = np.random.choice(n_samples, num_seed_samples, replace=False)
    seed_candidates = cands_flat[seed_indices].flatten()

    best_angle = None
    min_total_error = float("inf")

    for cand in seed_candidates:
        # Calculate distances from all samples to the current candidate
        diffs = angular_distance(cands_flat, cand)
        # Get the minimum distance for each sample
        min_diffs = np.min(diffs, axis=1)
        total_error = np.sum(min_diffs)

        if total_error < min_total_error - 1e-9:
            min_total_error = total_error
            best_angle = cand
        elif np.abs(total_error - min_total_error) <= 1e-9:
            # Tie-breaker: choose the smallest positive candidate in [0, 360)
            if best_angle is None or cand < best_angle:
                best_angle = cand

    # 3. Vectorized Estimate Refinement
    diffs = angular_distance(cands_flat, best_angle)
    best_match_idx = np.argmin(diffs, axis=1)
    # Extract the best matching candidate for each sample
    best_matches = cands_flat[np.arange(n_samples), best_match_idx]

    # Calculate signed difference to align closely with best_angle
    diff_vals = (best_matches - best_angle + 180.0) % 360.0 - 180.0

    estimated_relative_angle = np.fmod(np.mean(best_angle + diff_vals) + 360.0, 360.0)
    estimated_relative_angle = np.fmod(estimated_relative_angle, max_angle)

    # 4. Final choice of k1 and k2 for each sample based on estimated_relative_angle
    final_diffs = angular_distance(cands_flat, estimated_relative_angle)
    final_idx = np.argmin(final_diffs, axis=1)

    # Convert the flattened index back to (k1, k2) choices
    chosen_k1, chosen_k2 = np.unravel_index(final_idx, (sym1, sym2))

    rot1_unfolded = np.fmod(rot1 + chosen_k1 * period1, 360.0)
    rot2_unfolded = np.fmod(rot2 + chosen_k2 * period2, 360.0)

    return estimated_relative_angle, rot1_unfolded, rot2_unfolded


def find_particle_correspondence(ds1, ds2, dist_tol=None, axis_tol=None, verbose=0):
    """
    Find corresponding particles between two datasets based on particle UID (same extraction) or micrograph UID and spatial proximity (different extractions).
    The poses are then used to select the particles with consistent orientation of the particle's two poses (relative rotation is around an axis close to the +Z axis).

    Args:
        ds1: First CryoSPARC Dataset object.
        ds2: Second CryoSPARC Dataset object.
        dist_tol: Optional. Spatial distance tolerance in Angstroms.
        axis_tol: Optional. Tolerance for the angle between the +Z axis and the axis of the relative rotation of a particle's two poses, in degrees.

    Returns:
        numpy.ndarray: Array of shape (N, 2) containing matched UIDs (ds1_uid, ds2_uid).
    """

    # Phase 1: Candidate Identification
    common_uids, idx1_common, idx2_common = np.intersect1d(
        ds1["uid"], ds2["uid"], return_indices=True
    )

    cand_dict = {}  # ds2_idx -> list of ds1_idx
    if len(common_uids) > 0:
        if verbose > 1:
            print(f"Found {len(common_uids)} common particles by UID.")
        for i1, i2 in zip(idx1_common, idx2_common):
            cand_dict[i2] = [i1]
    else:
        # Match by spatial proximity
        if dist_tol is None:
            raise ValueError(
                "dist_tol must be provided when particles do not share UIDs."
            )

        uids1 = np.unique(ds1["location/micrograph_uid"])
        uids2 = np.unique(ds2["location/micrograph_uid"])
        common_mic_uids = np.intersect1d(uids1, uids2)
        print(f"Found {len(common_mic_uids)} common micrographs.")

        for mic_uid in common_mic_uids:
            idx1 = np.where(ds1["location/micrograph_uid"] == mic_uid)[0]
            idx2 = np.where(ds2["location/micrograph_uid"] == mic_uid)[0]
            if len(idx1) == 0 or len(idx2) == 0:
                continue

            pts1 = np.stack(
                [
                    ds1["location/center_x_frac"][idx1],
                    ds1["location/center_y_frac"][idx1],
                ],
                axis=1,
            )
            pts2 = np.stack(
                [
                    ds2["location/center_x_frac"][idx2],
                    ds2["location/center_y_frac"][idx2],
                ],
                axis=1,
            )

            # Scale fractional coordinates to Angstroms
            # micrograph_shape is (height, width)
            if (
                "location/micrograph_psize_A" in ds1.fields()
                and "location/micrograph_shape" in ds1.fields()
            ):
                psize = ds1["location/micrograph_psize_A"][idx1[0]]
                shape = ds1["location/micrograph_shape"][idx1[0]]
                scale = np.array([shape[1] * psize, shape[0] * psize])
                pts1 *= scale
                pts2 *= scale

            tree = KDTree(pts1)
            # Find all neighbors within dist_tol (now in Angstroms)
            neighbor_lists = tree.query_ball_point(pts2, dist_tol)
            for i2_local, neighbors in enumerate(neighbor_lists):
                if neighbors:
                    i2_global = idx2[i2_local]
                    cand_dict[i2_global] = [idx1[n] for n in neighbors]

        if verbose > 1:
            print(
                f"Found {len(cand_dict)} matched pairs of particles using micrograph UIDs and particle locations."
            )

    if not cand_dict:
        return np.array([])

    # Phase 2 & 3: Axis Filtering and Final Selection
    matches = []

    # Prepare rotations if needed
    if axis_tol is None or axis_tol <= 0:
        # No axis filtering: pick the "best" candidate
        # For UID match, there's only one. For spatial, we'll just pick the first one
        for i2, neighbors in cand_dict.items():
            best_match = neighbors[0]  # Just pick first for now
            matches.append((ds1["uid"][best_match], ds2["uid"][i2]))
    else:
        vz_min = np.cos(np.deg2rad(axis_tol))

        def get_rotations(ds, indices):
            for field in ["alignments3D_multi/pose", "alignments3D/pose"]:
                if field in ds.fields():
                    poses = ds[field][indices]
                    if poses.ndim == 3:
                        poses = np.squeeze(poses)
                    return R.from_rotvec(poses)
            return None

        # We only need poses for particles that are candidates
        all_idx1 = sorted(list(set(i1 for i1s in cand_dict.values() for i1 in i1s)))
        all_idx2 = sorted(list(cand_dict.keys()))

        R1_map = {
            idx: rot
            for idx, rot in zip(all_idx1, get_rotations(ds1, all_idx1))
            if rot is not None
        }
        R2_map = {
            idx: rot
            for idx, rot in zip(all_idx2, get_rotations(ds2, all_idx2))
            if rot is not None
        }

        if len(R1_map) < len(all_idx1) or len(R2_map) < len(all_idx2):
            raise ValueError("Pose information missing for some candidate particles")

        for i2, neighbors in cand_dict.items():
            rot2 = R2_map[i2]
            best_match = None
            best_vz = vz_min

            for i1 in neighbors:
                rot1 = R1_map[i1]
                r_diff = rot2 * rot1.inv()
                rotvec = r_diff.as_rotvec()
                angle = np.linalg.norm(rotvec)

                if angle < 1e-6:
                    vz = 1.0
                else:
                    axis = rotvec / angle
                    vz = axis[2]

                if vz > best_vz:
                    best_vz = vz
                    best_match = i1

            if best_match is not None:
                matches.append((ds1["uid"][best_match], ds2["uid"][i2]))

        if verbose > 1:
            print(
                f"Found {len(matches)} matched pairs of particles using relative rotation axis close to +Z axis."
            )

    return np.array(matches)


def add_args(parser):
    parser.add_argument("-p", "--projectID", help="CryoSPARC Project ID (e.g., P407)")
    parser.add_argument(
        "-j1", "--jobID1", help="First input dataset CryoSPARC Job ID (e.g., J100)"
    )
    parser.add_argument(
        "-j2", "--jobID2", help="Second input dataset CryoSPARC Job ID (e.g., J189)"
    )
    parser.add_argument(
        "-i1", "--input1", help="Path to local input .cs file for job 1"
    )
    parser.add_argument(
        "-pt1",
        "--pass_through1",
        help="Path to local pass-through .cs file for job 1 (optional)",
    )
    parser.add_argument(
        "-i2", "--input2", help="Path to local input .cs file for job 2"
    )
    parser.add_argument(
        "-pt2",
        "--pass_through2",
        help="Path to local pass-through .cs file for job 2 (optional)",
    )
    parser.add_argument(
        "-of1", "--outputFile1", help="Path to save reassigned dataset 1 localy (.cs)"
    )
    parser.add_argument(
        "-of2", "--outputFile2", help="Path to save reassigned dataset 2 localy (.cs)"
    )
    parser.add_argument(
        "-s1",
        "--sym1",
        help="Symmetry for job 1 (e.g., C5). Required if --input1 is used.",
    )
    parser.add_argument(
        "-s2",
        "--sym2",
        help="Symmetry for job 2 (e.g., C12). Required if --input2 is used.",
    )
    parser.add_argument("-w", "--workspaceID", help="Output Workspace ID (e.g., W1)")
    parser.add_argument(
        "--dist-tol",
        type=float,
        default=50.0,
        help="Spatial distance tolerance for matching in Angstroms. Default 50.0.",
    )
    parser.add_argument(
        "--axis-tol",
        type=float,
        default=5.0,
        help="Rotation axis tolerance for matching in degrees. Default 5.",
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=2, help="Verbosity level (0-2). Default 2."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reassign particle poses between jobs with different symmetries."
    )
    add_args(parser)
    args = parser.parse_args()
    main(args)
