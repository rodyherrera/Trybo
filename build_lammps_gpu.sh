#!/bin/bash

echo "Starting LAMMPS installation with GPU support"

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing basic dependencies..."
sudo apt install -y build-essential git cmake ccache wget \
    libfftw3-dev libpng-dev libjpeg-dev libhdf5-serial-dev \
    python3-dev python3-pip libopenmpi-dev openmpi-bin mpi-default-bin

echo "Installing CUDA Toolkit..."
if ! dpkg -l | grep -q nvidia-cuda-toolkit; then
    sudo apt install -y nvidia-cuda-toolkit
else
    echo "CUDA Toolkit is already installed"
fi

if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA was not installed correctly. Verify that your GPU is compatible."
    exit 1
fi

echo "CUDA installed successfully: $(nvcc --version | head -n1)"

echo "Cloning LAMMPS from GitHub..."
if [ ! -d "lammps" ]; then
    git clone --depth=1 https://github.com/lammps/lammps.git
else
    echo "LAMMPS directory already exists. Updating..."
    cd lammps
    git pull
    cd ..
fi

echo "Configuring build with GPU support..."
cd lammps
mkdir build
cd build

cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPKG_GPU=on \
    -DGPU_API=cuda \
    -DGPU_ARCH=sm_86 \
    -DBUILD_OMP=on \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DPKG_MOLECULE=on \
    -DPKG_KSPACE=on \
    -DPKG_MANYBODY=on \
    -DPKG_RIGID=on \
    -DPKG_MISC=on \
    -DPKG_USER-MISC=on

echo "Compiling LAMMPS (this may take several minutes)..."
make -j$(nproc)

echo "Installing LAMMPS..."
sudo make install

if command -v lmp &> /dev/null; then
    echo "LAMMPS with GPU support installed successfully!"
    echo "Installed version: $(lmp -v 2>&1 | grep 'LAMMPS version')"
    echo "Compiled packages:"
    lmp -h 2>&1 | grep -A 20 'Installed packages:'
else
    echo "Error: LAMMPS installation failed."
    exit 1
fi

echo "IMPORTANT: This script configured LAMMPS with GPU architecture 'sm_86' by default (for NVIDIA RTX 3060 Laptop GPU)."
echo "If you have a different GPU, modify the -DGPU_ARCH=sm_86 line with your specific architecture."
echo "To determine your GPU architecture, run: nvidia-smi --query-gpu=name,compute_cap --format=csv"
echo ""
echo "To use LAMMPS with GPU, include 'package gpu' in your input scripts."

echo "Installation completed."