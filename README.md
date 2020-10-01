About:
The aim of this project is to optimize and to openly provide osteoarthritis community a new technology to rapidly and automatically measure cartilage thickness, appearance and changes on magnetic resonance images (MRI) of the knee for huge image databases. This will allow assessment of trajectories of cartilage loss over time and associations with clinical outcomes on an unprecedented scale; future work will focus on incorporating additional disease markers, ranging from MRI-derived biomarkers for bone and synovial lesions, to biochemical biomarkers, to genetic information. novel, efficiently parallelized codes and tools for pairwise, group-wise and longitudinal analysis of high-resolution MR images suitable for large image repositories with the potential to produce highly reproducible diagnostic and monitoring capabilities by exploiting massively parallel computing on an HPC cluster and cloud facilities.

Installation:

Dependency python(>=3.6) packges:

pip3.6 install --user SimpleITK
pip3.6 install --user mpi4py
pip3.6 install --user tensorboardX
pip3.6 install --user matplotlib
pip3.6 install --user scikit-image
pip3.6 install --user sklearn
pip3.6 install --user pyparsing
pip3.6 install --user itk
pip3.6 install --user pynrrd
pip3.6 install --user blosc
pip3.6 install --user progressbar
pip3.6 install --user future
pip3.6 install --user ants
pip3.6  install --user nibabel

Other dependency packages:

yum install eigen3-devel gmp-devel gmpxx4ldbl mpfr-devel boost-devel boost-thread-devel tbb-devel python3-devel scotch-devel cmake3

OAI installation:

mkdir oai (this is the top-level directory where OAI analysis repository and other dependency packages should be installed)

NiftyReg or Easyreg:

Niftyreg installation:

mkdir niftyreg
cd niftyreg
git clone git://git.code.sf.net/p/niftyreg/git niftyreg-git
mkdir build
mkdir install
cd build
cmake3 -D CMAKE_C_COMPILER='path to c compiler' -D CMAKE_CXX_COMPILER='path to g++' -D CMAKE_INSTALL_PREFIX='path to the install dir' ..
make install

Or

Easyreg installation (requires python >=3.7):

cd oai
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

PyMesh installation (requires python >= 3.6):

cd oai
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
export PYMESH_PATH=`pwd`
pip install --user -r $PYMESH_PATH/python/requirements.txt
#edit setup.py: #!/usr/bin/env python -> #!/usr/bin/env python3.6
python3.6 ./setup.py build
cd $PYMESH_PATH/third_party
mkdir build
cd build
cmake3 -D CMAKE_C_COMPILER='path to c compiler' -D CMAKE_CXX_COMPILER='path to c++ compiler' ..
make
make tests
make install
cd ../..
mkdir build
cd build
cmake3 -D CMAKE_C_COMPILER='path to c compiler' -D CMAKE_CXX_COMPILER='path to c++ compiler' CMAKE_BUILD_TYPE=Release ..
make
cd tools/Tetrahedralization
cmake3 -E cmake_link_script CMakeFiles/lib_Tetrahedralization.dir/link.txt --verbose=1 >& jk.sh
#edit jk.sh, append /usr/lib64/libscotch.so /usr/lib64/libscotcherr.so /usr/lib64/libscotcherrexit.so /usr/lib64/libscotchmetis.so
sh jk.sh
cd ../..
make
make tests
cd ..
python3.6 ./setup.py install --user
python3.6 -c "import pymesh; pymesh.test()"

Surface-distance package:

cd oai
git clone https://github.com/deepmind/surface-distance
pip3.6 install surface-distance/

OAI_analysis package:

cd oai
git clone https://github.com/uncbiag/OAI_analysis.git
pip install --user -r requirement.txt

Getting the Data:

Download the data from https://nda.nih.gov/oai/accessing_images.html

Running Experiments:

configuration:
There are various inputs such as data locations, nifty-reg installation path,
easyreg/avsm path, and whether to use nifty or easyreg. All these can be
configured with config.yml.

# Knee cartilage analysis from OAI image data
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

The atlas is [given](./atlas/atlas_60_LEFT_baseline_NMI).

Single Image Analysis:
comment demo_analyze_cohort () in pipelines.py.

Cohort Analysis:
comment demo_analyze_single_image () in pipelines.py
