



# Knee cartilage analysis from OAI image data
The analysis interfaces given in [oai_image_analysis.py](./oai_image_analysis.py) include

1. Preprocess image, e.g. Normalized intensties, flip left/right knees to the same orientation.
2. Segmenting knee cartilage using the trained CNN model 
3. Extract the surface mesh and compute the thickness at each vertex
4. Regiter the image to the atlas 
5. Used the inverse transformation to warped the surface mesh with thickness map to the atlas space.
6. Map the thickness on the warped mesh to the atlas mesh
7. Project thickness from 3D surface to a 2D grid (TODO) 

The atlas in registration has been built by:
1. Build an atlas from a set of images with manual segmentations
2. Extract surface mesh from the atlas segmentation (average of the registered image used for building atlas)

See [pipelines.py](./pipelines.py) for how to config and run a analysis pipeline or part of it.

The atlas is [given](./atlas/atlas_60_LEFT_baseline_NMI).

# Dependencies:
0. Python >=3.6
1. [PyMesh](https://github.com/PyMesh/PyMesh): 
    built from source, tested with commit [e3c777a66c92f97dcfea610f66bbffa60701cd5f](https://github.com/PyMesh/PyMesh/tree/e3c777a66c92f97dcfea610f66bbffa60701cd5f) 
2. [NiftyReg](https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg) 
    built from source, tested with commit [4e4525b84223c182b988afaa85e32ac027774c42](https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg/tree/4e4525b84223c182b988afaa85e32ac027774c42)
3. [surface-distance](https://github.com/deepmind/surface-distance)
    built from source, tested with commit [f850c1640cd26c8cf6fa6095e7464db695406fd5](https://github.com/deepmind/surface-distance/tree/f850c1640cd26c8cf6fa6095e7464db695406fd5). Only needed for evaluate atlas.
4.[requirement.txt](./requirement.txt) gives other requirements can be installed from pip or conda.  
