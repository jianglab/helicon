"""Handler for the extractParticles option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import mrcfile
from cryosparc.dataset import Dataset
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "extractParticles"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the extractParticles option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--extractParticles",
        type=str,
        metavar="box_size=<n>:fft_crop_size=<n>[:recenter=<0|1>][replace_ctf=<0|1>][normalize=<0|1>][fill_mode=<mean|random>][sign=<-1|1>][n_micrographs=<-1|n>][fp16=<0|1>][:micrographs_cs_file=<filename>|micrographs_job_id=<JXXX>][reuse_job_id=<JXXX>]",
        help="split the dataset by micrograph. disabled by default",
        default="",
    )


def handle(
    data,
    args: argparse.Namespace,
    index_d: dict,
    param: object,
    output_title: str,
    output_slots: set,
    exp_group_id_name: str,
    micrograph_name: str,
    original_exp_group_ids: list,
):
    """Handle the extractParticles option.

    Parameters
    ----------
    data : Dataset
        The cryosparc Dataset.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.
    output_title : str
        Title for output filename construction.
    output_slots : set
        Output slot names.
    exp_group_id_name : str
        Name of the exposure group ID column.
    micrograph_name : str
        Name of the micrograph name column.
    original_exp_group_ids : list
        Original exposure group IDs.

    Returns
    -------
    tuple
        (data, output_title, output_slots, index_d) after processing.
    """
    if param:
        if "location/center_x_frac" not in data or "location/center_y_frac" not in data:
            raise HeliconError(
                "--extractParticles option requires location/center_x_frac, location/center_y_frac parameters in the input data"
            )

        param_dict_default = dict(
            box_size=0,
            fft_crop_size=0,
            flip_y=0,
            recenter=1,
            replace_ctf=0,
            normalize=1,
            fill_mode="random",
            sign=-1,
            n_micrographs=-1,
            fp16=1,
            micrographs_cs_file="",
            micrographs_job_id="",
            reuse_job_id="",
            force=0,
            plot_pdf=0,
        )
        _, param_dict = helicon.parse_param_str(param)
        param_dict, param_changed, param_unsuppported = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        if len(param_unsuppported):
            logger.warning("ignoring unknown parameters: %s", param_unsuppported)
        if args.verbose > 2:
            logger.info(f"\tCustom parameters: {param_changed}")
        box_size = int(param_dict["box_size"])
        if box_size <= 0:
            raise HeliconError("box_size (>0) must be specified")
        fft_crop_size = int(param_dict["fft_crop_size"])
        if fft_crop_size <= 0 or fft_crop_size > box_size:
            fft_crop_size = box_size
        recenter = int(param_dict["recenter"]) > 0
        replace_ctf = int(param_dict["replace_ctf"]) > 0
        sign = int(param_dict["sign"])
        flip_y = int(param_dict["flip_y"]) > 0
        fill_mode = param_dict["fill_mode"]
        normalize = int(param_dict["normalize"]) > 0
        fp16 = int(param_dict["fp16"]) > 0
        n_micrographs = int(param_dict["n_micrographs"])
        micrographs_job_id = param_dict["micrographs_job_id"]
        micrographs_cs_file = param_dict["micrographs_cs_file"]
        reuse_job_id = param_dict["reuse_job_id"]
        force = param_dict["force"]
        plot_pdf = int(param_dict["plot_pdf"]) > 0

        output_slots.add("blob")
        output_slots.add("location")

        col_mid = "location/micrograph_uid"
        micrograph_input = ""
        if "location/micrograph_path" not in data:
            if col_mid not in data:
                raise HeliconError(
                    f"{col_mid} must be in the input data when the input data does not have location/micrograph_path parameters"
                )
            if not (micrographs_cs_file or micrographs_job_id):
                raise HeliconError(
                    "micrographs_cs_file or micrographs_job_id must be provided when the input data does not have location/micrograph_path parameters"
                )
        if replace_ctf and not (micrographs_cs_file or micrographs_job_id):
            raise HeliconError(
                "micrographs_cs_file or micrographs_job_id must be provided when replace_ctf is specified"
            )
        if micrographs_cs_file or micrographs_job_id:
            if micrographs_cs_file:
                micrograph_input = micrographs_cs_file
                data_micrographs = Dataset.load(micrographs_cs_file)
                if (
                    "uid" not in data_micrographs
                    or "micrograph_blob/path" not in data_micrographs
                ):
                    raise HeliconError(
                        f"{micrographs_cs_file} does not contain uid and micrograph_blob/path"
                    )
            else:
                micrograph_input = f"{args.projectID}/{micrographs_job_id}"
                micrograph_input_job = cs.find_job(args.projectID, micrographs_job_id)
                input_micrographs_group_name = None
                for g in micrograph_input_job.doc["output_result_groups"]:
                    if g["type"] == "exposure":
                        input_micrographs_group_name = g["name"]
                        break
                if not input_micrographs_group_name:
                    raise HeliconError(
                        f"{micrographs_job_id} does not provide micrographs"
                    )
                data_micrographs = micrograph_input_job.load_output(
                    input_micrographs_group_name
                )
                if (
                    "uid" not in data_micrographs
                    or "micrograph_blob/path" not in data_micrographs
                ):
                    raise HeliconError(
                        f"{micrographs_job_id} result {input_micrographs_group_name} does not contain uid and micrograph_blob/path. Available parameters are: {' '.join(data_micrographs.keys())}"
                    )

            # Check if all micrograph IDs in data exist in data_micrographs
            data_mids = set(data[col_mid])
            micrographs_mids = set(data_micrographs["uid"])
            missing_mids = data_mids - micrographs_mids
            if missing_mids:
                raise HeliconError(
                    f"{len(missing_mids)} micrograph IDs in the input data are not found in the micrographs dataset"
                )

            if "location/micrograph_path" not in data:
                data.add_fields(["location/micrograph_path"], [str])

            cols_ctf = [col for col in data_micrographs if col.split("/")[0] == "ctf"]
            cols_ctf_missing_names = [
                col[0]
                for col in data_micrographs.descr()
                if col[0] in cols_ctf and col[0] not in data
            ]
            cols_ctf_missing_types = [
                col[1]
                for col in data_micrographs.descr()
                if col[0] in cols_ctf and col[0] not in data
            ]
            if len(cols_ctf_missing_names):
                data.add_fields(cols_ctf_missing_names, cols_ctf_missing_types)

            if replace_ctf:
                cols_ctf_to_copy = cols_ctf
            else:
                cols_ctf_to_copy = cols_ctf_missing_names
            if len(cols_ctf_to_copy):
                output_slots.add("ctf")

            for mid in data_mids:
                particle_row_mask = np.where(data[col_mid] == mid)
                micrograph_row_mask = np.where(data_micrographs["uid"] == mid)
                data["location/micrograph_path"][particle_row_mask] = data_micrographs[
                    "micrograph_blob/path"
                ][micrograph_row_mask][0]
                for col in cols_ctf_to_copy:
                    data[col][particle_row_mask] = data_micrographs[col][
                        micrograph_row_mask
                    ][0]

        if flip_y:
            data["location/center_y_frac"] = 1 - data["location/center_y_frac"]

        if recenter and ("alignments3D/shift" in data or "alignments2D/shift" in data):
            if "alignments3D/shift" in data:
                alignment_psize = data["alignments3D/psize_A"]
                alignment_shift = data["alignments3D/shift"]
                output_slots.add("alignments3D")
            elif "alignments2D/shift" in data:
                alignment_psize = data["alignments2D/psize_A"]
                alignment_shift = data["alignments2D/shift"]
                output_slots.add("alignments2D")

            micrograph_psize = data["location/micrograph_psize_A"]
            mic_shape_y, mic_shape_x = data["location/micrograph_shape"].T
            shift_x = alignment_psize * alignment_shift[:, 0] / micrograph_psize
            shift_y = alignment_psize * alignment_shift[:, 1] / micrograph_psize
            new_loc_x = data["location/center_x_frac"] * mic_shape_x - shift_x
            new_loc_y = data["location/center_y_frac"] * mic_shape_y - shift_y

            data["location/center_x_frac"] = new_loc_x / mic_shape_x
            data["location/center_y_frac"] = new_loc_y / mic_shape_y
            if "alignments3D/shift" in data:
                data["alignments3D/shift"][:] = [0, 0]
            else:
                data["alignments2D/shift"][:] = [0, 0]

        if "blob" not in data.prefixes():
            data.add_fields(
                fields=[
                    "blob/path",
                    "blob/idx",
                    "blob/shape",
                    "blob/psize_A",
                    "blob/sign",
                    "blob/import_sig",
                ],
                dtypes=["|O", "<u4", ("<u4", (2,)), "<f4", "<f4", "<u8"],
            )

        reuse_result_folder = None
        if args.projectID and not args.saveLocal:
            output_job = project.create_external_job(
                args.outputWorkspaceID,
                title="Extract Particles",
                desc=f"{' '.join(sys.argv)}",
            )
            for i, jobID in enumerate(args.jobID):
                input_job = cs.find_job(args.projectID, jobID)
                input_group = input_job.doc["output_result_groups"][args.groupIndex[i]]
                input_group_name = input_group["name"]
                output_job.connect(
                    target_input="particles",
                    source_job_uid=jobID,
                    source_output=input_group_name,
                    title="Particles",
                )
            output_job.add_output(
                type="particle",
                name="extracted_particles",
                slots=sorted(list(output_slots)),
                passthrough="particles",
                title="Particles extracted",
            )
            if micrographs_job_id is not None:
                output_job.connect(
                    target_input="micrographs",
                    source_job_uid=micrographs_job_id,
                    source_output=input_micrographs_group_name,
                    title="Micrographs",
                )
                output_job.add_output(
                    type="exposure",
                    name="micrographs",
                    slots=[],
                    passthrough="micrographs",
                    title="Passthrough micrographs",
                )
                # output_job.save_output("micrographs", data_micrographs)
            output_job.mkdir("extract")
            particle_dir = f"{output_job.uid}/extract"
            if reuse_job_id:
                reuse_job = project.find_job(reuse_job_id)
                source_path = Path(reuse_job.dir()) / "extract"
                if source_path.exists() and source_path.is_dir():
                    reuse_result_folder = source_path
            output_job.start(status="running")
        else:
            output_job = None
            particle_dir = "extract"
            Path(particle_dir).mkdir(parents=True, exist_ok=True)

        tasks = []
        mids = np.unique(data[col_mid])
        n_micrographs = len(mids) if n_micrographs <= 0 else n_micrographs
        for mid in mids[:n_micrographs]:
            tasks.append(
                (
                    data,
                    mid,
                    box_size,
                    fft_crop_size,
                    input_project_folder,
                    output_project_folder,
                    particle_dir,
                    sign,
                    fill_mode,
                    normalize,
                    fp16,
                    plot_pdf,
                )
            )

        if args.verbose > 1:
            logger.info(
                f"\tStart extracting {len(data):,} particles from {n_micrographs:,} micrographs"
            )

        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.cpu) as executor:
            n_skipped = 0
            futures = []
            results = []
            for ti, task in enumerate(tasks):
                (
                    data,
                    mid,
                    box_size,
                    fft_crop_size,
                    input_project_folder,
                    output_project_folder,
                    particle_dir,
                    sign,
                    fill_mode,
                    normalize,
                    fp16,
                    plot_pdf,
                ) = task
                subset = data.query({"location/micrograph_uid": mid})

                micrograph_path = subset["location/micrograph_path"][0]
                skip = False
                if not force and reuse_result_folder:
                    source_file = Path(
                        reuse_result_folder / f"{Path(micrograph_path).stem}.mrcs"
                    )
                    if source_file.exists() and source_file.is_file():
                        import mrcfile

                        with mrcfile.open(source_file, header_only=True) as mrc:
                            nx = mrc.header.nx
                            ny = mrc.header.ny
                            nz = mrc.header.nz
                            if nz == len(subset) and ny == nx and ny == fft_crop_size:
                                target_file = Path(
                                    project.dir() / particle_dir / source_file.name
                                )
                                target_file.hardlink_to(source_file)
                                skip = True
                                n_skipped += 1

                if skip:
                    apix = (
                        subset["location/micrograph_psize_A"][0]
                        * box_size
                        / fft_crop_size
                    )
                    result = subset.copy()
                    result["blob/path"] = f"{particle_dir}/{source_file.name}"
                    result["blob/idx"] = np.arange(len(result))
                    result["blob/shape"] = [(fft_crop_size, fft_crop_size)] * len(
                        result
                    )
                    result["blob/psize_A"] = apix
                    result["blob/sign"] = [sign] * len(result)
                    result["blob/import_sig"] = [1] * len(result)
                    results.append(result)
                    if args.verbose > 1:
                        logger.info(
                            f"\t{ti+1}/{len(tasks)}: reuses {str(source_file)}. skipped ({n_skipped})"
                        )
                    continue

                future = executor.submit(
                    extract_one_micrograph,
                    subset,
                    box_size,
                    fft_crop_size,
                    input_project_folder,
                    output_project_folder,
                    particle_dir,
                    sign,
                    fill_mode,
                    normalize,
                    fp16,
                    force,
                    plot_pdf,
                )
                futures.append(future)

            if args.verbose > 1 and n_skipped > 0:
                logger.info(
                    f"\t{len(tasks):,} micrographs: {n_skipped:,} skipped, {len(futures):,} to go"
                )

            if len(futures):
                from tqdm import tqdm

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Extracting",
                    unit="micrograph",
                ):
                    result = future.result()
                    results.append(result)

        from cryosparc.dataset import Dataset

        data = Dataset.append(*results, assert_same_fields=True)
        data = fill_missing_fields(data)

        if args.verbose > 1:
            logger.info(
                f"\t{len(data):,} particles extracted from {n_micrographs:,} micrographs"
            )

        if output_job:
            output_job.save_output("extracted_particles", data)
            output_job.log(
                f"Extracted {len(data):,} particles from {n_micrographs:,} micrographs\nJob completed",
                level="text",
            )
            output_job.stop()
            data = None
        else:
            output_title += (
                f"{'+'+micrograph_input if micrograph_input else ''}->extract particles"
            )
    return data, output_title, output_slots, index_d


def extract_one_micrograph(
    subset: Dataset,
    box_size: int,
    fft_crop_size: int,
    input_project_folder: Path,
    output_project_folder: Path,
    output_particle_foler: str,
    sign: int = -1,
    fill_mode: str = "random",
    normalize: bool = True,
    fp16: bool = False,
    force: bool = False,
    plot_pdf: bool = False,
) -> Dataset:
    """Extract particles from one micrograph.

    Parameters
    ----------
    subset : Dataset
        Particle data for this micrograph.
    box_size : int
        Extraction box size in pixels.
    fft_crop_size : int
        Final box size after FFT cropping.
    input_project_folder : Path
        Root folder for input files.
    output_project_folder : Path
        Root folder for output files.
    output_particle_foler : str
        Subfolder for particle stacks.
    sign : int, optional
        Sign to apply (-1 or 1). Defaults to -1.
    fill_mode : str, optional
        Fill mode for edge particles. Defaults to "random".
    normalize : bool, optional
        Whether to normalize extracted particles. Defaults to True.
    fp16 : bool, optional
        Use float16 output. Defaults to False.
    force : bool, optional
        Force re-extraction. Defaults to False.
    plot_pdf : bool, optional
        Generate PDF with particle locations. Defaults to False.

    Returns
    -------
    Dataset
        Subset with blob metadata added.
    """
    micrograph_path = subset["location/micrograph_path"][0]
    micrograph_file = input_project_folder / subset["location/micrograph_path"][0]

    extracted_particles_filename = (
        f"{output_particle_foler}/{Path(micrograph_path).stem}.mrcs"
    )
    particle_file = output_project_folder / extracted_particles_filename

    apix = subset["location/micrograph_psize_A"][0] * box_size / fft_crop_size

    import mrcfile
    import numpy as np

    skip = False
    if not force and Path(particle_file).exists():
        with mrcfile.open(particle_file, header_only=True) as mrc:
            nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
            if nz == len(subset) and ny == nx and ny == fft_crop_size:
                skip = True

    if not skip:
        mic_shape = subset["location/micrograph_shape"][0]
        location_x = np.rint(subset["location/center_x_frac"] * mic_shape[1]).astype(
            np.int32
        )
        location_y = np.rint(subset["location/center_y_frac"] * mic_shape[0]).astype(
            np.int32
        )

        with mrcfile.mmap(str(micrograph_file), mode="r") as mrc_micrograph:
            micrograph = mrc_micrograph.data

            if plot_pdf:
                micrograph_lp = helicon.low_high_pass_filter(
                    data=micrograph, low_pass_fraction=0.05
                )
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.imshow(micrograph_lp, cmap="gray")
                ax.scatter(
                    location_x,
                    location_y,
                    color="red",
                    s=10,
                    label="Particle Locations",
                )
                ax.set_title("Micrograph with Particle Locations")
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                ax.legend()
                pdf_filename = particle_file.with_suffix(".pdf")
                plt.savefig(pdf_filename, format="pdf")
                plt.close(fig)

            dtype = np.float16 if fp16 else np.float32
            particles = np.zeros(
                (len(subset), fft_crop_size, fft_crop_size), dtype=dtype
            )

            x0_offsets = location_x - box_size // 2
            y0_offsets = location_y - box_size // 2

            clip_buffer = np.zeros((box_size, box_size), dtype=np.float32)

            for i in range(len(subset)):
                x0, y0 = x0_offsets[i], y0_offsets[i]

                x_start, x_end = max(0, x0), min(mic_shape[1], x0 + box_size)
                y_start, y_end = max(0, y0), min(mic_shape[0], y0 + box_size)

                clip = micrograph[y_start:y_end, x_start:x_end]

                if clip.shape != (box_size, box_size):
                    clip_buffer.fill(0)
                    clip_buffer[
                        y_start - y0 : y_end - y0, x_start - x0 : x_end - x0
                    ] = clip.astype(clip_buffer.dtype)
                    clip = clip_buffer

                    if fill_mode is not None:
                        zeros = clip == 0
                        non_zeros = ~zeros
                        if fill_mode == "mean":
                            clip[zeros] = np.mean(clip[non_zeros])
                        elif fill_mode == "random":
                            non_zero_values = clip[non_zeros]
                            clip[zeros] = np.random.normal(
                                loc=np.mean(non_zero_values),
                                scale=np.std(non_zero_values),
                                size=np.count_nonzero(zeros),
                            )

                if fft_crop_size < box_size:
                    clip = helicon.fft_crop(
                        clip, output_size=(fft_crop_size, fft_crop_size)
                    )

                if sign < 0:
                    clip = np.max(clip) + np.min(clip) - clip

                if normalize:
                    std = np.std(clip)
                    if std:
                        mean = np.mean(clip)
                        clip = (clip - mean) / std

                particles[i] = clip.astype(dtype)

        with mrcfile.new(particle_file, overwrite=True) as mrc_output:
            mrc_output.set_data(particles)
            mrc_output.voxel_size = (apix, apix, apix)

    ret = subset.copy()
    ret["blob/path"] = str(extracted_particles_filename)
    ret["blob/idx"] = np.arange(len(ret))
    ret["blob/shape"] = [(fft_crop_size, fft_crop_size)] * len(ret)
    ret["blob/psize_A"] = apix
    ret["blob/sign"] = [sign] * len(ret)
    ret["blob/import_sig"] = [1] * len(ret)
    return ret


def fill_missing_fields(data: Dataset) -> Dataset:
    """Add missing CTF fields with default values (in-place).

    Parameters
    ----------
    data : Dataset
        CryoSPARC dataset to fill.

    Returns
    -------
    Dataset
        The same dataset with missing fields added.
    """
    default_var_type = {
        "ctf/tetra_A": ("<f4", (4,)),
        "ctf/scale_const": "<f4",
        "ctf/trefoil_A": ("<f4", (2,)),
        "ctf/bfactor": "<f4",
        "ctf/scale": "<f4",
        "ctf/tilt_A": ("<f4", (2,)),
        "ctf/anisomag": ("<f4", (4,)),
        "ctf/shift_A": ("<f4", (2,)),
    }
    nonzero_default = {
        "ctf/scale": 1,
    }
    for var in default_var_type:
        prefix = var.split("/")[0]
        if prefix in data.prefixes() and var not in data:
            data.add_fields([var], [default_var_type[var]])
            if var in nonzero_default:
                data[var] = nonzero_default[var]
    return data
