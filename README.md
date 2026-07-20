
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
- **denovo3D**:            A Web app that performs de novo helical indexing and 3D reconstruction from a single 2D image
- **helicalPitch**:        A Web app that helps you determine helical pitch/twist using 2D Classification info
- **helicalProjection**:   A Web app that helps you compare 2D images with helical structure projections
- **whereIsMyClass**:      A Web app that maps 2D classes to helical tube/filament images

### Streamlit web apps
- **ctfSimulation**:       A Web app that simulates 1D/2D TEM contrast transfer function (CTF)
- **helicalLattice**:      A Web app that illustrates the interconversion of 2D Lattice ⇔ Helical Lattice
- **hi3d**:                A Web app for helical indexing using the cylindrical projection of a 3D map
- **hill**:                A Web app for helical indexing using Fourier layer lines of 2D images
- **map2seq**:             A Web app that identifies the best protein sequence explaining a 3D density map
- **procart**:             A Web app that plots cartoon illustration of the residue properties of amyloid atomic models


## Documentation
- [Helicon@readthedocs](https://helicon.readthedocs.io): for users
- [Helicon@DeepWiki](https://deepwiki.com/jianglab/helicon): for developers
