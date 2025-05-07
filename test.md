# Image Filter Program Documentation

## Project Overview
This is an image processing program written in C++ that provides various image filtering capabilities for BMP images. The program is designed to enhance image quality through different filtering techniques.

## Project Structure
- `filter.cpp`: Main program implementation
- `bmplib.cpp` & `bmplib.h`: BMP image handling library
- `CMakeLists.txt`: Build configuration
- `input/`: Directory for input images
- `output/`: Directory for processed images
- `build/`: Build artifacts directory

## Features
1. **Unsharp Masking Filter**
   - Enhances image details through sharpening
   - Configurable kernel size, sigma, and sharpening factor
   - Supports both RGB and grayscale images

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Improves local contrast in images
   - Configurable tile size and clip limit
   - Works with any image size

3. **Combined Unsharp Masking + CLAHE**
   - Applies both filters in sequence
   - Enhanced detail and local contrast
   - Fixed CLAHE parameters for optimal results

## Technical Details
- Written in C++
- Uses CMake for build management
- Supports BMP image format
- Handles both RGB and grayscale images
- Memory-efficient processing
- Robust error handling

## Usage
The program is executed from the command line with the following format:
```
filter.exe input.bmp [filter_type] [parameters] output.bmp
```

### Example Commands:
1. Unsharp Masking:
   ```
   filter.exe input/sample.bmp unsharp 5 1.0 1.5 output.bmp
   ```

2. CLAHE:
   ```
   filter.exe input/sample.bmp clahe 8 2.0 0 output.bmp
   ```

3. Combined Filter:
   ```
   filter.exe input/sample.bmp unsharp_clahe 5 1.0 1.5 output.bmp
   ```

## Requirements
- C++ compiler with C++11 support
- CMake 3.0 or higher
- Windows operating system (based on .exe extension)

## Build Instructions
1. Create a build directory:
   ```
   mkdir build
   cd build
   ```

2. Generate build files:
   ```
   cmake ..
   ```

3. Build the project:
   ```
   cmake --build .
   ```

## Output
The program generates processed images in the `output` directory, including:
- Final processed images
- Intermediate results (for certain filters)
- Error logs (if any)

## Error Handling
The program includes comprehensive error handling for:
- File I/O operations
- Parameter validation
- Memory allocation
- Image processing operations
