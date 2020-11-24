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

sudo yum -y install eigen3-devel gmp-devel gmpxx4ldbl mpfr-devel boost-devel boost-thread-devel tbb-devel python3-devel scotch-devel cmake3 &>/dev/null


#installing the above packages on the compute nodes
sinfo_out=$(sinfo list | awk 'NF>1{print $NF}' | sed '1d' )

compute_prefix=$(echo $sinfo_out | cut -d[ -f1)

compute_range=$( echo "$sinfo_out" | cut -d "[" -f2- | cut -d "]" -f1 )

compute_start=$(echo "$compute_range" | cut -d\- -f1)

compute_end=$(echo "$compute_range" | cut -d\- -f2)

echo ${compute_prefix}

for ((i=compute_start;i<=compute_end;i++)); do
 echo "installing packages on ${compute_prefix}${i}"
 ssh ${compute_prefix}${i} sudo yum -y install eigen3-devel gmp-devel gmpxx4ldbl mpfr-devel boost-devel boost-thread-devel tbb-devel python3-devel scotch-devel cmake3 &>/dev/null
done

sudo yum -y install libmpc-devel &>/dev/null
sudo yum -y install gcc-c++
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

echo 'installing PyMesh'

git clone https://github.com/PyMesh/PyMesh.git &>/dev/null
cd PyMesh &>/dev/null
git checkout e3c777a66c92f97dcfea610f66bbffa60701cd5f -b test &>/dev/null
git submodule update --init &>/dev/null
pip3.6 install -r python/requirements.txt &>/dev/null
#edit setup.py: #!/usr/bin/env python3.6
sed -i 's/#!\/usr\/bin\/env\ python/#!\/usr\/bin\/env\ python3.6/g' setup.py
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

echo 'PyMesh installation complete'

echo 'Installing surface-distance'

cd ../
git clone https://github.com/deepmind/surface-distance &>/dev/null
pip3.6 install surface-distance/

echo 'surface-distance installation complete'

echo 'installing niftyreg'
git clone git://git.code.sf.net/p/niftyreg/git niftyreg-git
cd niftyreg-git
mkdir build
mkdir install
cd build
cmake3 -D CMAKE_C_COMPILER=/shared/bin/gcc -D CMAKE_CXX_COMPILER=/shared/bin/g++ -D CMAKE_INSTALL_PREFIX=../install/
make install

echo 'niftyreg installation complete'
