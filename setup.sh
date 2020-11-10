#!/bin/bash

mkdir build
cd build
echo 'downloading python 3.6.8'
wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tar.xz &>/dev/null
echo 'downloaded python 3.6.8'
echo 'extracting'
tar -xvf Python-3.6.8.tar.xz &>/dev/null
echo 'extraction complete'
echo 'building'
cd Python-3.6.8
./configure --prefix=/shared --enable-optimizations &>/dev/null
make -j &>/dev/null
echo 'build complete'
make install &>/dev/null

export PATH=/shared/bin:$PATH &>/dev/null

echo 'installing dependency modules'
python3.6 -m pip install SimpleITK==1.2.4 &>/dev/null

python3.6 -m pip install mpi4py tensorboardX matplotlib scikit-image sklearn pymesh pyparsing itk pynrrd blosc progressbar future ants nibabel &>/dev/null

python3.6 -m pip install pandas torch torchvision &>/dev/null

sudo yum install eigen3-devel gmp-devel gmpxx4ldbl mpfr-devel boost-devel boost-thread-devel tbb-devel python3-devel scotch-devel cmake3 &>/dev/null

sudo yum install libmpc-devel &>/dev/null
sudo yum install gcc-c++
echo 'dependency module installation complete'

cd ..

echo 'building gcc-5.4.0'
wget ftp://ftp.mirrorservice.org/sites/sourceware.org/pub/gcc/releases/gcc-5.4.0/gcc-5.4.0.tar.gz &>/dev/null
tar -xvf gcc-5.4.0.tar.gz &>/dev/null
cd gcc-5.4.0 &>/dev/null
./configure --enable-languages=c,c++ --disable-multilib --prefix=/shared/ &>/dev/null
make &>/dev/null
make install &>/dev/null

echo 'gcc-5.4.0 complete'

sudo ln -sf /usr/bin/cmake3 /usr/bin/cmake &>/dev/null

cd ..

git clone https://github.com/PyMesh/PyMesh.git &>/dev/null
git checkout e3c777a66c92f97dcfea610f66bbffa60701cd5f -b test &>/dev/null
git submodule update --init &>/dev/null
pip3.6 install -r python/requirements.txt &>/dev/null
#edit setup.py: #!/usr/bin/env python3.6
cd third_party
mkdir build
cd build
cmake3 -D CMAKE_C_COMPILER=/shared/bin/gcc -D CMAKE_CXX_COMPILER=/shared/bin/g++ .. &>/dev/null
make &>/dev/null
make install &>/dev/null
cd ../../
mkdir build
cd build
cmake3 -D CMAKE_C_COMPILER=/shared/bin/gcc -D CMAKE_CXX_COMPILER=/shared/bin/g++ CMAKE_BUILD_TYPE=Release .. &>/dev/null
make &>/dev/null
cd tools/Tetrahedralization
cmake3 -E cmake_link_script CMakeFiles/lib_Tetrahedralization.dir/link.txt --verbose=1 &> jk.sh
sh jk.sh &>/dev/null
cd ../../
make &>/dev/null
cd ..
python3.6 ./setup.py install &>/dev/null
export LD_LIBRARY_PATH=/shared/lib64

cd ../
git clone https://github.com/deepmind/surface-distance &>/dev/null
pip3.6 install surface-distance/

#TODO: add file editings for PyMesh
#TODO: niftyreg
#TODO: data download
