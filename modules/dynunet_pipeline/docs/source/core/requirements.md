## Requirements

### Operating system
The tool has been tested with Python 3.8 on Ubuntu 20.04 and Windows 10.

### Python libraries
The following Python libraries are required:

* PyTorch (Installation instructions: [https://pytorch.org/](https://pytorch.org/))
* PyTorch-Ignite (install with: `pip install pytorch-ignite`)
* MONAI (install with: `pip install monai`)
* NumpPy (install with `pip install numpy`)
* Nibabel (install with `pip install nibabel`)
* tqdm (install with `pip install tqdm`)

### HD-BET
HD-BET is used for brain extraction if the model was trained on skull-stripped images. To install HD-BET
type these commands in the command line:

    git clone https://github.com/MIC-DKFZ/HD-BET.git
    cd HD-BET 
    pip install -e .
    cd ..

The first time HD-BET is run after installation, it should automatically download model weights from zenodo.org. If 
there is a problem and the error "0.model not found", you can download it manually [here](https://zenodo.org/record/2540695/#.Y_27oYDP18I)
and place the model weights in the location requested in the error message. 

### ANTs 
Furthermore, preprocessing involves image registration which relies on routines available in 
[Advanced Normalization Tools (ANTs)](http://stnava.github.io/ANTs/). The two required pre-compiled binaries 
for Linux/Windows/MacOS are included in the downloaded `data` folder (next section).
These have to be made accessible from the command line via the commands "antsRegistration" and
"antsApplyTransforms" from the command line.

On Linux, this can be achieved by creating two new text files under `/usr/bin` (or any other path that is part of the
PATH environment variable) named "antsRegistration" and "antsApplyTransforms" with the following contents:
    
    `#! /bin/sh
    exec <...>/antsRegistration.glnxa64 "$@"`

and
    
    `#! /bin/sh
    exec <...>/antsApplyTransforms.glnxa64 "$@"`

where `<...>` is the absolute path to folder containing the downloaded Linux binaries.


