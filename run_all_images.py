#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
import time
import json
import re

def extract_timing_data(output):
    """Extract timing data from the execution summary."""
    timing_data = {}
    
    # Regular expression to match the timing table
    pattern = r'Version\s+Gaussian Blur\s+Unsharp Masking\s+CLAHE\s+Total\s+\n-+\n(.*?)\n=+'
    match = re.search(pattern, output, re.DOTALL)
    
    if match:
        # Split the matched content into lines and process each line
        lines = match.group(1).strip().split('\n')
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    version = parts[0]
                    timing_data[version] = {
                        'gaussian_blur': int(parts[1]),
                        'unsharp_masking': int(parts[2]),
                        'clahe': int(parts[3]),
                        'total': int(parts[4])
                    }
    
    return timing_data

def run_for_all_images():
    # Get the input directory
    input_dir = Path("input")
    if not input_dir.exists():
        print("Error: 'input' directory does not exist")
        return

    # Get all image files in the input directory
    image_extensions = ('.bmp',)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print("No image files found in the input directory")
        return

    print(f"Found {len(image_files)} image files to process")
    
    # Create a results directory for the summary
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get current timestamp for the results files
    results_file = results_dir / f"performance_results.txt"
    json_file = results_dir / f"performance_data.json"
    
    # Dictionary to store all timing data
    all_timing_data = {
        'images': {}
    }
    
    # Process each image
    with open(results_file, 'w') as f:
        f.write("Performance Results Summary\n")
        f.write("="*80 + "\n\n")
        
        for image_file in image_files:
            print(f"\nProcessing {image_file.name}...")
            f.write(f"Image: {image_file.name}\n")
            f.write("-"*80 + "\n")
            
            # Run the script for this image
            cmd = ["python", "run_all_versions.py", str(image_file)]
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout)
                
                # Write the output to the results file
                f.write(result.stdout)
                f.write("\n" + "="*80 + "\n\n")
                
                # Extract timing data and add to the dictionary
                timing_data = extract_timing_data(result.stdout)
                if timing_data:
                    all_timing_data['images'][image_file.name] = timing_data
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Error processing {image_file.name}: {e.stderr}"
                print(error_msg)
                f.write(error_msg + "\n\n")
                continue
            
            # Add a small delay between processing images
            time.sleep(1)
    
    # Save the timing data to JSON file
    with open(json_file, 'w') as f:
        json.dump(all_timing_data, f, indent=4)
    
    print(f"\nProcessing complete!")
    print(f"Detailed results have been saved to {results_file}")
    print(f"Timing data for plotting has been saved to {json_file}")

if __name__ == "__main__":
    run_for_all_images() 