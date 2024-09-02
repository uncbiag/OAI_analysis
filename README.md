
# OAI Analysis Pipeline

This repository contains the full code to process OAI DESS images. I.e., segmentation, thickness measurement, atlas-registration, 3D to 2D mapping.
We are currently in the process of refactoring it. The new (not entirely completed yet) codebase can be found in the [OAI Analysis 2 repository](https://github.com/uncbiag/OAI_analysis_2). 

## Knee cartilage analysis from OAI image data
The analysis interfaces given in [oai_image_analysis.py](./oai_image_analysis.py) include

1. Preprocess image, e.g. Normalized intensties, flip left/right knees to the same orientation.
2. Segment knee cartilage using the trained CNN model 
3. Extract the surface mesh and compute the thickness at each vertex
4. Register the image to the atlas 
5. Use the inverse transformation to warped the surface mesh with thickness map to the atlas space.
6. Map the thickness on the warped mesh to the atlas mesh
7. Project thickness from 3D surface to a 2D grid

The atlas in registration has been built by:
1. Build an atlas from a set of images with manual segmentations
2. Extract surface mesh from the atlas segmentation (average of the registered image used for building atlas)

See [pipelines.py](./pipelines.py) for how to config and run a analysis pipeline or part of it.

The atlas is [given](./atlas/atlas_60_LEFT_baseline_NMI).

## Dependencies:
0. Python >=3.6
1. [PyMesh](https://github.com/PyMesh/PyMesh): 
    built from source, tested with commit [e3c777a66c92f97dcfea610f66bbffa60701cd5f](https://github.com/PyMesh/PyMesh/tree/e3c777a66c92f97dcfea610f66bbffa60701cd5f) 
2. [NiftyReg](https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg) 
    built from source, tested with commit [4e4525b84223c182b988afaa85e32ac027774c42](https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg/tree/4e4525b84223c182b988afaa85e32ac027774c42)
   Or [EasyReg](github.com/uncbiag/easyreg) branch oai_analysis
3. [surface-distance](https://github.com/deepmind/surface-distance)
    built from source, tested with commit [f850c1640cd26c8cf6fa6095e7464db695406fd5](https://github.com/deepmind/surface-distance/tree/f850c1640cd26c8cf6fa6095e7464db695406fd5). Only needed for evaluating atlas.
4. [requirement.txt](./requirement.txt) gives other requirements can be installed from pip or conda.


## Network Version Use

The network version refers to paper "Networks for Joint Affine and Non-parametric Image Registration" (https://arxiv.org/pdf/1903.08811.pdf) \

*  Set env, let's first cd into OAI_analysis repository, then run the following commands
```
git clone https://github.com/uncbiag/easyreg.git
cd easyreg
pip install -r requirements.txt
# Download the pretrained model (in mermaid directory) (a seven-step affine network with a three-step vSVF model)
cd demo && mkdir pretrained && cd pretrained
gdown https://drive.google.com/uc?id=1f7pWcwGPvr28u4rr3dAL_4aD98B7PNDX
cd ../..
git clone https://github.com/uncbiag/mermaid.git
cd mermaid
python setup.py develop
```  


Overall the use is the same as the optimization version.\
(Make sure the path settings in pipline is correct) \
Several extra settings need to be done in main function in pipeline.py
```
use_nifti=False
avsm_directory = 'The PATH OF EASYREG DIRECTORY'
```
