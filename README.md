
# Helicon

A collection of tools for cryo-EM analysis of helical structures

## Publication
  Li, D., Zhang, X., Jiang, W., 2025. Helicon: Helical parameter determination and 3D reconstruction from one image. Journal of Structural Biology 217, 108256. [doi.org/10.1016/j.jsb.2025.108256](https://doi.org/10.1016/j.jsb.2025.108256)


## Installation
Run this command in a terminal:  
```
pip install "helicon[all] @ git+https://github.com/jianglab/helicon"
```

## Usage
Run this command in a terminal and follow the help message:
```
helicon --help
```

## Programs included in helicon

### Command-line programs
- **cryosparc**:           A command-line tool that interacts with a CryoSPARC server and performs image analysis tasks
- **images2star**:         A command-line tool that analyzes/transforms dataset(s) and saves the dataset in a RELION star file
- **proc3d**:              A command-line tool that analyzes/transforms 3D maps
- **trueFSC**:             A command-line tool to compute True FSC curve with optimal mask and phase randomization

### GUI apps
- **display**:             A file browser for viewing image, map, star, bild, eps, pdf, html, and text files

### Shiny web apps
- **webApps**:             A Helicon Web app with seven analytical tools as tabs:
   - *WhereIsMyClass*:     Map 2D classes to helical tube/filament images
   - *HelicalProjection*:  Compare 2D images with helical structure projections in EMDB
   - *HILL*:               Helical indexing using Fourier-Bessel layer lines (power spectra/phase difference)
   - *HelicalPitch*:       Determine helical pitch/twist using 2D Classification info
   - *Denovo3D*:           De novo helical indexing and 3D reconstruction from a single 2D image
   - *HelicalLattice*:     Interconversion of 2D Lattice ⇔ Helical Lattice
   - *HI3D*:               Helical indexing via cylindrical projection of a 3D map

### Streamlit web apps
- **ctfSimulation**:       A Web app that simulates 1D/2D TEM contrast transfer function (CTF)
- **map2seq**:             A Web app that identifies the best protein sequence explaining a 3D density map
- **procart**:             A Web app that plots cartoon illustration of the residue properties of amyloid atomic models


## Documentation
- [Helicon@readthedocs](https://helicon.readthedocs.io): for users
- [Helicon@DeepWiki](https://deepwiki.com/jianglab/helicon): for developers
