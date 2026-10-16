"""Small RELION and CryoSPARC projects for the tests of the file browser and of the job graph."""

import json

import numpy as np


PIPELINE = """
# version 30001

data_pipeline_general

_rlnPipeLineJobCounter                      5

# version 30001

data_pipeline_processes

loop_
_rlnPipeLineProcessName #1
_rlnPipeLineProcessAlias #2
_rlnPipeLineProcessTypeLabel #3
_rlnPipeLineProcessStatusLabel #4
Import/job001/       None relion.import  Succeeded
Extract/job005/ Extract/first-boxes/ relion.extract  Succeeded
Class2D/job010/ Class2D/looks/ relion.class2d  Succeeded
Class2D/job011/       None relion.class2d  Failed
Select/job012/       None relion.select  Succeeded

# version 30001

data_pipeline_nodes

loop_
_rlnPipeLineNodeName #1
_rlnPipeLineNodeType #2
Import/job001/movies.star 0
Extract/job005/particles.star 2
Class2D/job010/run_it020_data.star 2
Select/job012/particles.star 2

# version 30001

data_pipeline_input_edges

loop_
_rlnPipeLineEdgeFromNode #1
_rlnPipeLineEdgeProcess #2
Import/job001/movies.star Extract/job005/
Extract/job005/particles.star Class2D/job010/
Extract/job005/particles.star Class2D/job011/
Class2D/job010/run_it020_data.star Select/job012/
Extract/job005/particles.star Select/job012/

# version 30001

data_pipeline_output_edges

loop_
_rlnPipeLineEdgeProcess #1
_rlnPipeLineEdgeToNode #2
Import/job001/ Import/job001/movies.star
Extract/job005/ Extract/job005/particles.star
Class2D/job010/ Class2D/job010/run_it020_data.star
Select/job012/ Select/job012/particles.star
"""


def make_pipeline_project(tmp_path):
    """A RELION project that has a pipeline: Import -> Extract -> Class2D (two) -> Select."""
    root = tmp_path / "relion-proj"
    for job in (
        "Import/job001",
        "Extract/job005",
        "Class2D/job010",
        "Class2D/job011",
        "Select/job012",
    ):
        (root / job).mkdir(parents=True)
    (root / "default_pipeline.star").write_text(PIPELINE)
    return root


def make_cs_project(tmp_path):
    """A CryoSPARC project: filament tracing, extraction and two 2D classifications."""
    root = tmp_path / "CS-demo"
    root.mkdir()
    (root / "project.json").write_text(
        json.dumps({"uid": "P1", "title": "Demo project"})
    )

    def job(uid, job_type, status, title="", parents=(), **extra):
        d = root / uid
        d.mkdir()
        record = dict(uid=uid, job_type=job_type, status=status, title=title, **extra)
        # the real files list the inputs as "parents" (and in the connections)
        record["parents"] = list(parents)
        record["input_slot_groups"] = [
            dict(connections=[dict(job_uid=p) for p in parents])
        ]
        (d / "job.json").write_text(json.dumps(record))
        return d

    j5 = job("J5", "filament_tracer_gpu", "completed", "my picks")
    job("J6", "extract_micrographs_multi", "completed", parents=["J5"])
    job(
        "J7", "class_2D_new", "completed", "first look", parents=["J6"],
        created_at={"$date": "2026-10-05T14:50:00.000Z"},
        started_at={"$date": "2026-10-05T14:55:49.123Z"},
        completed_at={"$date": "2026-10-05T16:01:02.000Z"},
    )  # fmt: skip
    job("J12", "class_2D_new", "failed", parents=["J6"])
    job("J13", "some_new_job_type", "running")
    job("J14", "blob_picker_gpu", "building")
    job("J15", "blob_picker_gpu", "queued")
    dtype = [
        ("blob/path", "S20"),
        ("blob/idx", "<u4"),
        ("ctf/df1_A", "<f4"),
        ("uid", "<u8"),
    ]
    with open(j5 / "picked_particles.cs", "wb") as f:
        np.save(f, np.zeros(7, dtype=dtype))
    (root / "J99").mkdir()  # no job.json: not a job
    return root
