# OAI data cartilage analysis pipeline
1. Build an atlas from a set of images with manual segmentations
2. Extract surface mesh from the atlas segmentation (average of the registered image used for building atlas)
3. For each new images, do:
    1. segment the image with trained segmentation model
    2. extract the surface mesh and compute the thickness at each vertex
    3. regiter the image to the atlas 
    4. used the inverse transformation to warped the surface mesh with thickness map to the atlas space.
