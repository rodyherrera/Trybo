#!/bin/bash

echo "Trybo (May 5 2025 - Development - 1.0)"
echo "Your suite of atomic-level analyses."

if [ ! -f "./lammps/build/lmp" ]; then
    echo "Error: LAMMPS executable not found at ./lammps/build/lmp"
    echo "Building LAMMPS automatically..."
    
    # Run the build script
    bash toolchain/build_lammps_gpu.sh
    
    # Check if the build was successful
    if [ $? -eq 0 ] && [ -f "./lammps/build/lmp" ]; then
        echo "LAMMPS built successfully. Continuing..."
    else
        echo "LAMMPS build failed. Please check the build output."
        exit 1
    fi
fi

# Check if input file exists
if [ ! -f "./src/trybo_wear_simulation.lammps" ]; then
    echo "Error: Input file not found at ./src/trybo_wear_simulation.lammps"
    exit 1
fi

# Get processor information
TOTAL_PROCS=$(nproc)
if [ -f /proc/cpuinfo ]; then
    # Count physical cores only (not hyperthreaded logical processors)
    PHYSICAL_CORES=$(grep "cpu cores" /proc/cpuinfo | head -1 | awk '{print $4}')
    if [ -z "$PHYSICAL_CORES" ]; then
        # Fallback if cpu cores info is not available
        PHYSICAL_CORES=$((TOTAL_PROCS / 2))
    fi
else
    # Estimate physical cores (assuming hyperthreading)
    PHYSICAL_CORES=$((TOTAL_PROCS / 2))
fi

# Use one less than physical cores to leave one for the system
USE_PROCS=$((PHYSICAL_CORES - 1))

# Ensure we use at least 1 processor
if [ $USE_PROCS -lt 1 ]; then
    USE_PROCS=1
fi

echo "System has $TOTAL_PROCS logical processors ($PHYSICAL_CORES physical cores)"
echo "Using $USE_PROCS processors for simulation"
echo "Starting simulation..."

# Run LAMMPS with MPI - using the --use-hwthread-cpus option to ensure proper counting
mpirun --use-hwthread-cpus -np $USE_PROCS ./lammps/build/lmp -in ./src/trybo_wear_simulation.lammps

# Alternative command with oversubscribe if the above fails
if [ $? -ne 0 ]; then
    echo "First MPI attempt failed, trying with --oversubscribe option..."
    mpirun --oversubscribe -np $USE_PROCS ./lammps/build/lmp -in ./src/trybo_wear_simulation.lammps
fi

if [ $? -eq 0 ]; then
    echo "Simulation completed successfully!"
else
    echo "Simulation failed with exit code $?"
    exit 1
fi

exit 0