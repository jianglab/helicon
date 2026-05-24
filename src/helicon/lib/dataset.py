from __future__ import annotations

import logging, os
from pathlib import Path
from typing import Any, Iterator
import numpy as np
import pandas as pd
import helicon
from .exceptions import HeliconIOError

logger = logging.getLogger(__name__)

__all__ = [
    "EMDB",
    "get_amyloid_atlas",
    "get_emd_entries",
    "update_helical_parameters_from_curated_table",
]


class EMDB:
    """Singleton interface for accessing EMDB entries, maps, and metadata.

    Provides methods to download, cache, and read EMDB map files and XML metadata,
    with support for a local mirror directory.
    """

    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> EMDB:
        """Ensure only one instance of EMDB exists (singleton pattern).

        Returns
        -------
        EMDB
            The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super(EMDB, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        use_curated_helical_parameters: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the EMDB interface, loading the entry list.

        Parameters
        ----------
        use_curated_helical_parameters : bool, optional
            Whether to use curated helical parameters from the Jiang lab
            validation table. Defaults to True.
        cache_dir : str or Path, optional
            Custom cache directory path. Defaults to ``helicon.cache_dir / "emdb"``.
        """
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
        fields: list[str] | None = None,
        use_curated_helical_parameters: bool = True,
    ) -> None:
        """Fetch and update the list of EMDB entries from the EMDB API.

        Parameters
        ----------
        fields : list of str, optional
            List of metadata field names to retrieve from EMDB. Defaults to
            standard fields including helical parameters.
        use_curated_helical_parameters : bool, optional
            Whether to override helical parameters with values from the
            curated validation table. Defaults to True.
        """
        if fields is None:
            fields = [
                "emdb_id",
                "title",
                "structure_determination_method",
                "resolution",
                "fitted_pdbs",
                "image_reconstruction_helical_delta_z_value",
                "image_reconstruction_helical_delta_phi_value",
                "image_reconstruction_helical_axial_symmetry_details",
            ]
        try:
            entries = get_emd_entries(fields=fields)
            if use_curated_helical_parameters:
                entries = update_helical_parameters_from_curated_table(df=entries)
            self.meta = entries.sort_values(by="emd_id", key=lambda x: x.astype(int))
            self.emd_ids = list(self.meta["emd_id"])
        except Exception:
            logger.warning("Failed to obtain the list of EMDB entries", exc_info=True)

    def _validate_emd_id(self, emd_id: str):
        """Validate and normalize an EMDB ID string.

        Strips prefixes like ``EMD-`` or ``emd_`` to return the numeric ID.

        Parameters
        ----------
        emd_id : str
            Raw EMDB ID (e.g. ``EMD-1234`` or ``1234``).

        Returns
        -------
        str
            Normalized numeric EMDB ID string.

        Raises
        ------
        AssertionError
            If the ID is not found in the entry list.
        """
        emd_id_input = emd_id
        if not isinstance(emd_id, str):
            emd_id = str(emd_id)
        emd_id = emd_id.split(sep="-")[-1].split(sep="_")[-1]
        assert emd_id in self.emd_ids, f"ERROR: {emd_id_input} is not in EMDB"
        return emd_id

    def _get_emdb_file(
        self, emd_id: str, cache_filename: str, mirror_relpath: str, url_method: Any
    ) -> Path:
        """Retrieve an EMDB file from cache, local mirror, or remote URL.

        Checks local cache first, then a local mirror directory if configured,
        and finally downloads from the remote EMDB server.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.
        cache_filename : str
            Name for the cached file.
        mirror_relpath : str
            Relative path within the local mirror directory.
        url_method : callable
            Function that returns the download URL for the file.

        Returns
        -------
        Path
            Path to the retrieved (and cached) file.
        """
        emd_id = self._validate_emd_id(emd_id)
        target_file = self.cache_dir / cache_filename
        if target_file.exists() and target_file.stat().st_size:
            return target_file

        if self.local_emdb_mirror:
            mirror_file = self.local_emdb_mirror / mirror_relpath
            if not (mirror_file.exists() and mirror_file.stat().st_size):
                if os.access(self.local_emdb_mirror, os.W_OK):
                    url = url_method(emd_id)
                    mirror_file.parent.mkdir(parents=True, exist_ok=True)
                    helicon.download_file_from_url(
                        url, target_file_name=str(mirror_file)
                    )

            if mirror_file.exists() and mirror_file.stat().st_size:
                target_file.unlink(missing_ok=True)
                target_file.symlink_to(mirror_file)
                mtime = target_file.stat().st_mtime
                os.utime(target_file, (mtime, mtime), follow_symlinks=False)
                return target_file

        url = url_method(emd_id)
        downloaded = helicon.download_file_from_url(
            url, target_file_name=str(target_file), return_filename=True
        )
        if downloaded is None:
            raise HeliconIOError(f"failed to download {emd_id} from EMDB")
        return Path(downloaded)

    def get_emdb_map_url(self, emd_id: str):
        """Return the download URL for an EMDB map file.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        str
            URL to the compressed (``.map.gz``) map file on the EMDB server.
        """
        emd_id = self._validate_emd_id(emd_id)
        # server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
        server = "https://ftp.ebi.ac.uk/pub/databases"  # European Bioinformatics Institute, England
        # server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
        url = f"{server}/emdb/structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
        return url

    def get_emdb_map_file(self, emd_id: str):
        """Retrieve the map file for an EMDB entry (from cache, mirror, or remote).

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        Path
            Path to the downloaded or cached map file.
        """
        emd_id = self._validate_emd_id(emd_id)
        return self._get_emdb_file(
            emd_id,
            cache_filename=f"emd_{emd_id}.map.gz",
            mirror_relpath=f"structures/EMD-{emd_id}/map/emd_{emd_id}.map.gz",
            url_method=self.get_emdb_map_url,
        )

    def download_all_map_files(
        self, verbose: int = 0, max_workers: int | None = None
    ) -> None:
        """Download map files for all EMDB entries in parallel.

        Parameters
        ----------
        verbose : int, optional
            If > 0, print progress for each download. Defaults to 0.
        max_workers : int, optional
            Maximum number of parallel downloads. Defaults to None (auto, up to 8).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def download_one(emd_id):
            self.get_emdb_map_file(emd_id)
            return emd_id

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(download_one, e): e for e in self.emd_ids}
            for i, fut in enumerate(as_completed(futures), 1):
                if verbose:
                    emd_id = futures[fut]
                    logger.info(
                        f"Downloaded {i}/{len(self)}: {self.get_emdb_map_url(emd_id=emd_id)}"
                    )
                fut.result()

    def read_emdb_map(self, emd_id: str):
        """Read an EMDB map file and return the data array and voxel size.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        tuple
            (data, apix) where data is the 3D map array and apix is the
            voxel size in Angstroms.
        """
        map_file = self.get_emdb_map_file(emd_id=emd_id)
        import mrcfile

        with mrcfile.open(map_file) as mrc:
            data = mrc.data
            apix = float(mrc.voxel_size.x)
            data, _ = helicon.change_map_axes_order(
                data, mrc.header, new_axes=["x", "y", "z"]
            )
        return data, apix

    def get_emdb_xml_url(self, emd_id: str):
        """Return the download URL for an EMDB XML metadata file.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        str
            URL to the XML header file on the EMDB server.
        """
        emd_id = self._validate_emd_id(emd_id)
        # server = "https://files.wwpdb.org/pub"    # Rutgers University, USA
        server = "https://ftp.ebi.ac.uk/pub/databases"  # European Bioinformatics Institute, England
        # server = "http://ftp.pdbj.org/pub" # Osaka University, Japan
        url = f"{server}/emdb/structures/EMD-{emd_id}/header/emd-{emd_id}.xml"
        return url

    def get_emdb_xml_file(self, emd_id: str):
        """Retrieve the XML metadata file for an EMDB entry.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        Path
            Path to the downloaded or cached XML file.
        """
        emd_id = self._validate_emd_id(emd_id)
        return self._get_emdb_file(
            emd_id,
            cache_filename=f"emd_{emd_id}.xml",
            mirror_relpath=f"structures/EMD-{emd_id}/header/emd-{emd_id}.xml",
            url_method=self.get_emdb_xml_url,
        )

    def download_all_xml_files(
        self, verbose: int = 0, max_workers: int | None = None
    ) -> None:
        """Download XML metadata files for all EMDB entries in parallel.

        Parameters
        ----------
        verbose : int, optional
            If > 0, print progress for each download. Defaults to 0.
        max_workers : int, optional
            Maximum number of parallel downloads. Defaults to None (auto, up to 8).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def download_one(emd_id):
            self.get_emdb_xml_file(emd_id)
            return emd_id

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(download_one, e): e for e in self.emd_ids}
            for i, fut in enumerate(as_completed(futures), 1):
                if verbose:
                    emd_id = futures[fut]
                    logger.info(
                        f"Downloaded {i}/{len(self)}: {self.get_emdb_xml_url(emd_id=emd_id)}"
                    )
                fut.result()

    def read_emdb_xml(self, emd_id: str):
        """Parse an EMDB XML metadata file into a DotDict.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        helicon.DotDict
            Nested dictionary with dot notation access to all XML elements.
        """
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

    def get_info(self, emd_id: str, return_xml_content: bool = False) -> Any:
        """Return metadata for the specified EMDB entry as a DotDict.

        Parameters
        ----------
        emd_id : str
            EMDB ID.
        return_xml_content : bool, optional
            Whether to also return parsed XML content. Defaults to False.

        Returns
        -------
        DotDict or tuple
            Metadata with dot notation access. If *return_xml_content* is True,
            returns (dot_dict, xml_content).
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
        except Exception:
            pitch = np.nan
        info.pitch = pitch

        if return_xml_content:
            xml_content = self.read_emdb_xml(emd_id)
            return info, xml_content

        return info

    def helical_structure_ids(self) -> list[str]:
        """Return a list of EMDB IDs for helical structures.

        Returns
        -------
        list of str
            EMDB IDs of entries whose method is ``'helical'``.
        """
        ids = self.meta.loc[self.meta["method"] == "helical", "emd_id"]
        return list(ids)

    def amyloid_atlas_ids(self) -> list[str]:
        """Return a list of EMDB IDs present in the Amyloid Atlas.

        Returns
        -------
        list of str
            EMDB IDs of entries also found in the Amyloid Atlas.
        """
        df = get_amyloid_atlas()
        ids = [
            id
            for id in df["emd_id"].str.split("-", expand=True).iloc[:, -1]
            if id in self.emd_ids
        ]
        return ids

    def __len__(self) -> int:
        """Return the number of EMDB entries.

        Returns
        -------
        int
            Number of entries in the metadata table.
        """
        return len(self.emd_ids)

    def __getitem__(self, i: int) -> tuple[np.ndarray, float]:
        """Return the map data and pixel size for the i-th EMDB entry.

        Parameters
        ----------
        i : int
            Index into the sorted entry list.

        Returns
        -------
        tuple
            (data, apix) for the entry.
        """
        assert (
            0 <= i < len(self.emd_ids)
        ), f"ERROR: i must be in range [0, {len(self.emd_ids)}). You have specifed {i=}"
        return self.read_emdb_map(self.emd_ids[i])

    def __call__(self, emd_id: str):
        """Read the map data for a given EMDB ID.

        Parameters
        ----------
        emd_id : str
            EMDB entry ID.

        Returns
        -------
        tuple
            (data, apix) for the entry.
        """
        return self.read_emdb_map(emd_id=emd_id)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """Iterate over all EMDB entries, yielding (data, apix) for each.

        Yields
        ------
        tuple
            (data, apix) for each entry in the sorted list.
        """
        for emd_id in self.emd_ids:
            yield self.read_emdb_map(emd_id)


################################################################################


@helicon.cache(cache_dir=str(helicon.cache_dir), expires_after=7, verbose=0)  # 7 days
def get_emd_entries(fields: list[str]) -> pd.DataFrame:
    """Fetch EMDB entry metadata from the EMDB API.

    Parameters
    ----------
    fields : list of str
        List of metadata field names to retrieve from EMDB.

    Returns
    -------
    pd.DataFrame
        DataFrame of EMDB entries with standardized column names
        (method, pdb, rise, twist, csym).
    """
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
def get_amyloid_atlas(
    url: str = "https://people.mbi.ucla.edu/sawaya/amyloidatlas",
) -> pd.DataFrame:
    """Fetch and process the Amyloid Atlas table, mapping PDB IDs to EMDB entries.

    Scrapes the Amyloid Atlas web page, filters for cryo-EM entries, and maps
    PDB IDs to corresponding EMDB IDs.

    Parameters
    ----------
    url : str, optional
        URL of the Amyloid Atlas web page. Defaults to
        https://people.mbi.ucla.edu/sawaya/amyloidatlas.

    Returns
    -------
    pd.DataFrame
        Table of Amyloid Atlas entries with emd_id, resolution, pdb_id, and
        sample columns.
    """
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
    df: pd.DataFrame,
    url: str = "https://raw.githubusercontent.com/jianglab/EMDB_helical_parameter_curation/refs/heads/main/EMDB_validation.csv",
) -> pd.DataFrame:
    """Update helical parameters (twist, rise, csym) from a curated validation table.

    Merges a curated CSV table into the DataFrame, overwriting the existing
    helical parameters where curated data exists.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with emdb_id, twist, rise, and csym columns.
    url : str, optional
        URL of the curated validation CSV file. Defaults to the Jiang lab
        EMDB helix parameter curation repository.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with curated helical parameters where available.
    """
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
