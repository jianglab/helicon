import os
from pathlib import Path
import pandas as pd
from persist_cache import cache
import helicon


class EMDB:
    def __init__(self, cache_dir=None):
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

        self.update_emd_entries()

    def update_emd_entries(
        self,
        fields=[
            "emdb_id",
            "title",
            "structure_determination_method",
            "resolution",
            "image_reconstruction_helical_delta_z_value",
            "image_reconstruction_helical_delta_phi_value",
            "image_reconstruction_helical_axial_symmetry_details",
        ],
    ):
        @cache(
            name="emdb_entries", dir=str(self.cache_dir), expiry=7 * 24 * 60 * 60
        )  # 7 days
        def cached_update_emd_entries(fields):
            url = f'https://www.ebi.ac.uk/emdb/api/search/current_status:"REL"?rows=1000000&wt=csv&download=true&fl={",".join(fields)}'
            entries = pd.read_csv(url)
            entries["emd_id"] = (
                entries["emdb_id"].str.split("-", expand=True).iloc[:, 1]
            )
            entries = entries.rename(
                columns={
                    "structure_determination_method": "method",
                    "image_reconstruction_helical_delta_z_value": "rise",
                    "image_reconstruction_helical_delta_phi_value": "twist",
                    "image_reconstruction_helical_axial_symmetry_details": "csym",
                }
            )
            return entries

        try:
            entries = cached_update_emd_entries(fields=fields)
            self.meta = entries
            self.emd_ids = sorted(entries["emd_id"])
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
        if map_file.exists():
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
        map_file = helicon.download_url(url, target_file_name=str(map_file))
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
        import mrcfile

        with mrcfile.open(map_file) as mrc:
            data = mrc.data
            apix = float(mrc.voxel_size.x)
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
        xml_file = helicon.download_url(url, target_file_name=str(xml_file))
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

        class DotDict(dict):
            def __getattr__(self, name):
                return self[name]

            def __setattr__(self, name, value):
                self[name] = value

        data = DotDict()

        def parse_element(element, data):
            for child in element:
                if len(child) == 0:
                    data[child.tag] = child.text
                else:
                    data[child.tag] = DotDict()
                    parse_element(child, data[child.tag])

        parse_element(root, data)
        return data

    def helical_structure_ids(self):
        ids = self.meta.loc[self.meta["method"] == "helical", "emd_id"]
        return list(ids)

    def amyloid_atlas_ids(self):
        df = get_amyloid_atlas()
        ids = [
            id
            for id in df["emd_id"].str.split("-", expand=True).iloc[:, 1]
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


@cache(name="pdb_id_2_emd_id", dir=str(helicon.cache_dir / "emdb"))
def pdb_id_2_emd_id(pdb_id):
    import requests

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url).json()
    emd_id = list(response["rcsb_entry_container_identifiers"]["emdb_ids"])[0]
    print(f"{pdb_id=} -> {emd_id=}")
    return emd_id


@cache(
    name="get_amyloid_atlas",
    dir=str(helicon.cache_dir / "emdb"),
    expiry=30 * 24 * 60 * 60,
)  # 30 days
def get_amyloid_atlas():
    url = "https://people.mbi.ucla.edu/sawaya/amyloidatlas/"
    replaced_pdb_ids = {"7z40": "8ade"}

    df = pd.read_html(url, header=0, flavor="html5lib")[0]

    mask = df["PDB ID"].isin(replaced_pdb_ids)
    df.loc[mask, "PDB ID"] = df.loc[mask, "PDB ID"].str.lower().map(replaced_pdb_ids)

    df = df[df["Method"].str.lower() == "cryoem"].copy()
    df["emd_id"] = df["PDB ID"].apply(pdb_id_2_emd_id)

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
