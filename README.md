
# Helicon

A collection of tools for cryo-EM analysis of helical structures.

## Publication
  Li, D., Zhang, X., and Jiang, W. (2025). Helicon: Helical indexing and 3D reconstruction from one image. bioRxiv, 2025.04.05.647385. [doi:10.1101/2025.04.05.647385](https://doi.org/10.1101/2025.04.05.647385).
  


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

## Commands included in helicon
- **cryosparc**:           A command line tool that interacts with a CryoSPARC server and performs image analysis tasks
- **ctfSimulation**:       A Web app that simulates 1D/2D TEM contrast transfer function (CTF)
- **denovo3D**:            A Web app for de novo helical indexing and 3D reconstruction from a single 2D image
- **denovo3DBatch**:       A command line tool for de novo helical indexing and 3D reconstruction from a single 2D image
- **helicalLattice**:      A Web app that illustrates the interconversion of 2D Lattice ⇔ Helical Lattice
- **helicalPitch**:        A Web app that helps you determine helical pitch/twist using 2D Classification info
- **helicalProjection**:   A Web app that helps you compare 2D images with helical structure projections
- **hi3d**:                A Web app for helical indexing using the cylindrical projection of a 3D map
- **hill**:                A Web app for helical indexing using Fourier layer lines of 2D images
- **images2star**:         A command line tool that analyzes/transforms dataset(s) and saves the dataset in RELION star file
- **map2seq**:             A Web app that identifies the best protein sequence explaining a 3D density map
- **proc3d**:              A command line tool that anaylzes/transforms 3D maps
- **procart**:             A Web app that plots cartoon illustration of the residue properties of amyloid atomic models
- **whereIsMyClass**:      A Web app that maps 2D classes to helical tube/filament images

