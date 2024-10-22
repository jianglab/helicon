
# Helicon

A collection of tools for cryo-EM analysis of helical structures.

## Installation
Run this command in a terminal:  
```
pip install "helicon[all] @ git+https://github.com/jianglab/helicon"
```
## Functions

### helicon linear regression
Reconstrct the helical 3D volume from a single 2D image

### hill
Interactive webapp for helical indexing in fourier space

### FiT
Previously named HLM (https://github.com/smallelephant9516/HLM), use the language model to classify different types of filaments after 2D classification

### helicalPitch
Estimating the Pitch of the helical structure (especially the low twist amyloid) from distance of the same 2D class assigned to the same filaments. 

### images2star
Operation on the .star meta files

### procart
Estimating the amino acid sequence from the 3D density of the amyloid structure. 

## Usage
Run this command in a terminal and follow the help message:
```
helicon --help
```

## Disclaimer
Helicon is at a very early stage of development, and we are releasing it to promote open science. However, it might be very unstable/buggy, and the support will be very limited.