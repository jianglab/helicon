import os
from pathlib import Path
import numpy as np
import pandas as pd
import helicon


class EMDB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EMDB, cls).__new__(cls)
        return cls._instance

    def __init__(self, use_curated_helical_parameters=True, cache_dir=None):
        self.emd_ids = []
        self.meta = None

        self.cache_dir = Path(cache_dir) if cache_dir else helicon.cache_dir / "emdb"
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.local_emdb_mirror = (
            Path(os.getenv("EMDB_MIRROR_DIR"))
            if "EMDB_MIRROR_DIR" in os.environ
            else None
        )
        if self.local_emdb_mirror is not None:
            if not (
                self.local_emdb_mirror.exists() and self.local_emdb_mirror.is_dir()
            ):
                self.local_emdb_mirror = None

        self.update_emd_entries(
            use_curated_helical_parameters=use_curated_helical_parameters
        )

    def update_emd_entries(
        self,
        fields=[
            "emdb_id",
            "title",
            "structure_determination_method",
            "resolution",
            "fitted_pdbs",
            "image_reconstruction_helical_delta_z_value",
            "image_reconstruction_helical_delta_phi_value",
            "image_reconstruction_helical_axial_symmetry_details",
        ],
        use_curated_helical_parameters=True,
    ):
        try:
            entries = get_emd_entries(fields=fields)
            if use_curated_helical_parameters:
                entries = update_helical_parameters_from_curated_table(df=entries)
            self.meta = entries.sort_values(by="emd_id", key=lambda x: x.astype(int))
            self.emd_ids = list(self.meta["emd_id"])
        except Exception as e:
            helicon.color_print(e)
            helicon.color_print("WARNING: failed to obtain the list of EMDB entries")

    def get_emdb_map_url(self, emd_id: str):
        emd_id_input = emd_id
        if not isinstance(emd_id, str):
            emd_id = str(emd_id)
        emd_id = emd_id.split(sep="-")[-1].split(sep="_")[-1]
        assert emd_id in self.emd_ids, f"ERROR: {emd_id_input} is not in EMDB"
        # server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
        server = "https://ftp.ebi.ac.uk/pub/databases"  # European Bioinformatics Institute, England
        # server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
        url = f"{server}/emdb/structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
        return url

    def get_emdb_map_file(self, emd_id: str):
        emd_id_input = emd_id
        if not isinstance(emd_id, str):
            emd_id = str(emd_id)
        emd_id = emd_id.split(sep="-")[-1].split(sep="_")[-1]
        assert emd_id in self.emd_ids, f"ERROR: {emd_id_input} is not in EMDB"
        map_file = self.cache_dir / f"emd_{emd_id}.map.gz"
        if map_file.exists() and map_file.stat().st_size:
            return map_file
        if self.local_emdb_mirror:
            map_file_mirror = (
                self.local_emdb_mirror
                / f"structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
            )
            if map_file_mirror.exists() and map_file_mirror.stat().st_size:
                map_file.symlink_to(map_file_mirror)
                map_file_mtime = map_file.stat().st_mtime
                os.utime(
                    map_file, (map_file_mtime, map_file_mtime), follow_symlinks=False
                )
                return map_file
        url = self.get_emdb_map_url(emd_id)
        map_file = helicon.download_file_from_url(
            url, target_file_name=str(map_file), return_filename=True
        )
        if map_file is None:
            raise IOError(f"ERROR: failed to download {emd_id} from EMDB")
        return Path(map_file)

    def download_all_map_files(self, verbose=0):
        for i, emd_id in enumerate(self.emd_ids):
            if verbose:
                print(
                    f"Downloading {i+1}/{len(self)}: {self.get_emdb_map_url(emd_id=emd_id)}"
                )
            self.get_emdb_map_file(emd_id)

    def read_emdb_map(self, emd_id: str):
        map_file = self.get_emdb_map_file(emd_id=emd_id)
        if map_file is None:
            raise IOError(f"ERROR: failed to download {emd_id} from EMDB")

        import mrcfile

        with mrcfile.open(map_file) as mrc:
            data = mrc.data
            apix = float(mrc.voxel_size.x)
            data, _ = helicon.change_map_axes_order(
                data, mrc.header, new_axes=["x", "y", "z"]
            )
        return data, apix

    def get_emdb_xml_url(self, emd_id: str):
        emd_id_input = emd_id
        if not isinstance(emd_id, str):
            emd_id = str(emd_id)
        emd_id = emd_id.split(sep="-")[-1].split(sep="_")[-1]
        assert emd_id in self.emd_ids, f"ERROR: {emd_id_input} is not in EMDB"
        # server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
        server = "https://ftp.ebi.ac.uk/pub/databases"  # European Bioinformatics Institute, England
        # server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
        url = f"{server}/emdb/structures/EMD-{emd_id}/header/emd-{emd_id}.xml"
        return url

    def get_emdb_xml_file(self, emd_id: str):
        emd_id_input = emd_id
        if not isinstance(emd_id, str):
            emd_id = str(emd_id)
        emd_id = emd_id.split(sep="-")[-1].split(sep="_")[-1]
        assert emd_id in self.emd_ids, f"ERROR: {emd_id_input} is not in EMDB"
        xml_file = self.cache_dir / f"emd_{emd_id}.xml"
        if xml_file.exists():
            return xml_file
        if self.local_emdb_mirror:
            xml_file_mirror = (
                self.local_emdb_mirror
                / f"structures/EMD-{emd_id}/header/emd-{emd_id}.xml"
            )
            if xml_file_mirror.exists() and xml_file_mirror.stat().st_size:
                xml_file.symlink_to(xml_file_mirror)
                xml_file_mtime = xml_file.stat().st_mtime
                os.utime(
                    xml_file, (xml_file_mtime, xml_file_mtime), follow_symlinks=False
                )
                return xml_file
        url = self.get_emdb_xml_url(emd_id)
        xml_file = helicon.download_file_from_url(
            url, target_file_name=str(xml_file), return_filename=True
        )
        return Path(xml_file)

    def download_all_xml_files(self, verbose=0):
        for i, emd_id in enumerate(self.emd_ids):
            if verbose:
                print(
                    f"Downloading {i+1}/{len(self)}: {self.get_emdb_xml_url(emd_id=emd_id)}"
                )
            self.get_emdb_xml_file(emd_id)

    def read_emdb_xml(self, emd_id: str):
        xml_file = self.get_emdb_xml_file(emd_id=emd_id)
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_file)
        root = tree.getroot()

        data = helicon.DotDict()

        def parse_element(element, data):
            for child in element:
                if len(child) == 0:
                    data[child.tag] = child.text
                else:
                    data[child.tag] = helicon.DotDict()
                    parse_element(child, data[child.tag])

        parse_element(root, data)
        return data

    def get_info(self, emd_id, return_xml_content=False):
        """Return metadata for the specified EMDB entry as a dictionary with dot notation access.

        Args:
            emd_id (str): EMDB ID
            return_xml_content (bool, optional): Whether to also return parsed XML content. Defaults to False.

        Returns:
            DotDict or tuple: Metadata for the entry with dot notation access. If return_xml_content=True, returns tuple of (dot_dict, xml_content).
        """

        if not isinstance(emd_id, str):
            emd_id = str(emd_id)
        emd_id = emd_id.split(sep="-")[-1].split(sep="_")[-1]
        assert emd_id in self.emd_ids, f"ERROR: {emd_id} is not in EMDB"

        row = self.meta.loc[self.meta["emd_id"] == emd_id].iloc[0]
        info = helicon.DotDict(row.to_dict())
        try:
            pitch = round(
                helicon.twist2pitch(
                    info.twist, info.rise, return_pitch_for_4p75Angstrom_rise=True
                )
            )
        except:
            pitch = np.nan
        info.pitch = pitch

        if return_xml_content:
            xml_content = self.read_emdb_xml(emd_id)
            return info, xml_content

        return info

    def helical_structure_ids(self):
        ids = self.meta.loc[self.meta["method"] == "helical", "emd_id"]
        return list(ids)

    def amyloid_atlas_ids(self):
        df = get_amyloid_atlas()
        ids = [
            id
            for id in df["emd_id"].str.split("-", expand=True).iloc[:, -1]
            if id in self.emd_ids
        ]
        return ids

    def __len__(self):
        return len(self.emd_ids)

    def __getitem__(self, i):
        assert (
            0 <= i < len(self.emd_ids)
        ), f"ERROR: i must be in range [0, {len(self.emd_ids)}). You have specifed {i=}"
        return self.read_emdb_map(self.emd_ids[i])

    def __call__(self, emd_id: str):
        return self.read_emdb_map(emd_id=emd_id)

    def __iter__(self):
        for emd_id in self.emd_ids:
            yield self.read_emdb_map(emd_id)


################################################################################


@helicon.cache(cache_dir=str(helicon.cache_dir), expires_after=7, verbose=0)  # 7 days
def get_emd_entries(fields):
    url = f'https://www.ebi.ac.uk/emdb/api/search/current_status:"REL"?rows=1000000&wt=csv&download=true&fl={",".join(fields)}'
    entries = pd.read_csv(url)
    entries["emd_id"] = entries["emdb_id"].str.split("-", expand=True).iloc[:, 1]
    entries = entries.rename(
        columns={
            "structure_determination_method": "method",
            "fitted_pdbs": "pdb",
            "image_reconstruction_helical_delta_z_value": "rise",
            "image_reconstruction_helical_delta_phi_value": "twist",
            "image_reconstruction_helical_axial_symmetry_details": "csym",
        }
    )
    return entries


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "emdb"), expires_after=30, verbose=0
)  # 30 days
def get_amyloid_atlas(url="https://people.mbi.ucla.edu/sawaya/amyloidatlas"):
    replaced_pdb_ids = {"7z40": "8ade"}

    df = pd.read_html(url, header=0, flavor="html5lib")[0]

    mask = df["PDB ID"].isin(replaced_pdb_ids)
    df.loc[mask, "PDB ID"] = df.loc[mask, "PDB ID"].str.lower().map(replaced_pdb_ids)

    df = df[df["Method"].str.lower() == "cryoem"].copy()

    emdb = EMDB()
    assert emdb.meta is not None, f"Failed to get the list of EMDB entries"

    pdb2emd_mapping = {}
    for index, row in emdb.meta.iterrows():
        for pdb_id in str(row["pdb"]).lower().split(","):
            if pdb_id:
                pdb2emd_mapping[pdb_id] = row["emd_id"]
    df["emd_id"] = df["PDB ID"].map(pdb2emd_mapping)

    df["sample"] = df["Protein"] + " - " + df["Fibril Origins"]
    cols = dict(
        emd_id="emd_id",
        resolution="Resol- ution (Å)",
        pdb_id="PDB ID",
        sample="sample",
        others=["Residues Ordered", "Reference"],
    )
    df = df[helicon.flatten(cols.values())]
    df = df.rename(columns={cols[k]: k for k in cols if k != "others"})
    df = df.drop_duplicates(subset=["emd_id", "pdb_id"])
    df = df.reset_index()

    return df


def update_helical_parameters_from_curated_table(
    df,
    url="https://raw.githubusercontent.com/jianglab/EMDB_helical_parameter_curation/refs/heads/main/EMDB_validation.csv",
):
    columns = df.columns
    df_curated = pd.read_csv(url)
    df_curated = df_curated[df_curated["emdb_id"].isin(df["emdb_id"])]
    df_curated = df_curated.rename(
        columns={
            "twist_validated (°)": "twist",
            "rise_validated (Å)": "rise",
            "csym_validated": "csym",
        }
    )
    df_curated = df_curated[["emdb_id", "twist", "rise", "csym"]]
    df_updated = df.merge(
        df_curated, on="emdb_id", how="left", suffixes=("", "_curated")
    )
    df_updated["twist"] = df_updated["twist_curated"].combine_first(df_updated["twist"])
    df_updated["rise"] = df_updated["rise_curated"].combine_first(df_updated["rise"])
    df_updated["csym"] = df_updated["csym_curated"].combine_first(df_updated["csym"])
    df_updated["twist"] = pd.to_numeric(df_updated["twist"], errors="coerce").round(3)
    df_updated["rise"] = pd.to_numeric(df_updated["rise"], errors="coerce").round(3)
    df_updated = df_updated[columns]
    return df_updated
