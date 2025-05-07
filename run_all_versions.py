#!/usr/bin/env python3

import os
import sys
import subprocess
import time
import re
from pathlib import Path

def parse_timing(output):
    timing = {
        'gaussian_blur': 0,
        'unsharp_masking': 0,
        'clahe': 0,
        'total': 0
    }
    
    # Extract timing information using regex
    gaussian_match = re.search(r'Gaussian blur completed in (\d+) ms', output)
    unsharp_match = re.search(r'Unsharp masking completed in (\d+) ms', output)
    clahe_match = re.search(r'CLAHE completed in (\d+) ms', output)
    total_match = re.search(r'Total Time:\s+(\d+) ms', output)
    
    if gaussian_match:
        timing['gaussian_blur'] = int(gaussian_match.group(1))
    if unsharp_match:
        timing['unsharp_masking'] = int(unsharp_match.group(1))
    if clahe_match:
        timing['clahe'] = int(clahe_match.group(1))
    if total_match:
        timing['total'] = int(total_match.group(1))
    
    return timing

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        return True, parse_timing(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(e.stderr)
        return False, None

def print_timing_summary(timings):
    print("\nExecution Time Summary (in milliseconds):")
    print("="*80)
    print(f"{'Version':<10} {'Gaussian Blur':<15} {'Unsharp Masking':<15} {'CLAHE':<15} {'Total':<15}")
    print("-"*80)
    
    for version, timing in timings.items():
        if timing:
            print(f"{version:<10} {timing['gaussian_blur']:<15} {timing['unsharp_masking']:<15} "
                  f"{timing['clahe']:<15} {timing['total']:<15}")
    
    print("="*80)
    
    # Calculate speedups relative to sequential
    if 'sequential' in timings and timings['sequential']:
        seq_total = timings['sequential']['total']
        print("\nSpeedup relative to Sequential Version:")
        print("-"*80)
        for version, timing in timings.items():
            if version != 'sequential' and timing:
                speedup = seq_total / timing['total']
                print(f"{version:<10}: {speedup:.2f}x faster")

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_all_versions.py <input_image>")
        print("Example: python run_all_versions.py input/sample1.bmp")
        sys.exit(1)

    input_image = sys.argv[1]
    if not os.path.exists(input_image):
        print(f"Error: Input file '{input_image}' does not exist")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Default parameters
    N = "5"        # kernel size
    sigma = "1.0"  # blur strength
    alpha = "1.5"  # sharpening strength
    output_base = Path(input_image).stem

    # Get the input filename without extension
    input_base = Path(input_image).stem

    # Dictionary to store timing information
    timings = {}

    # Run sequential version
    seq_cmd = ["./filter_seq", input_image, "unsharp_clahe", N, sigma, alpha, output_base]
    success, timing = run_command(seq_cmd, "Sequential Version")
    if success:
        timings['sequential'] = timing

    # Run OpenMP version
    omp_cmd = ["./filter_omp", input_image, "unsharp_clahe", N, sigma, alpha, output_base]
    success, timing = run_command(omp_cmd, "OpenMP Version")
    if success:
        timings['openmp'] = timing

    # Run MPI version (with 4 processes)
    mpi_cmd = ["mpiexec", "-n", "1", "./filter_mpi", input_image, "unsharp_clahe", N, sigma, alpha, output_base]
    success, timing = run_command(mpi_cmd, "MPI Version")
    if success:
        timings['mpi'] = timing

    # Run CUDA version
    cuda_cmd = ["./filter_cuda", input_image, "unsharp_clahe", N, sigma, alpha, output_base]
    success, timing = run_command(cuda_cmd, "CUDA Version")
    if success:
        timings['cuda'] = timing

    print("\nAll versions completed!")

    # Print timing summary
    print_timing_summary(timings)

if __name__ == "__main__":
    main() 