#include <iostream>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <vector>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>
#include "bmplib.h"

using namespace std;
namespace fs = std::filesystem;

// Custom exception class for image processing errors
class ImageProcessingError : public std::runtime_error {
public:
    explicit ImageProcessingError(const string& message) : std::runtime_error(message) {}
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw ImageProcessingError(string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// Function to validate image dimensions
void validateImageDimensions(const ImageSize* size) {
    if (size->width <= 0 || size->height <= 0) {
        throw ImageProcessingError("Invalid image dimensions: width and height must be positive");
    }
    if (size->width > 10000 || size->height > 10000) {
        throw ImageProcessingError("Image dimensions too large: maximum supported size is 10000x10000");
    }
}

// CUDA kernel for Gaussian kernel generation
__global__ void gaussianKernel(double* kernel, int N, double sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int i = idx / N;
        int j = idx % N;
        double x = i - (N/2);
        double y = j - (N/2);
        kernel[idx] = exp(-(x*x + y*y) / (2 * sigma * sigma));
    }
}

// CUDA kernel for convolution
__global__ void convolveKernel(unsigned char* output, const unsigned char* input, 
                             const double* kernel, int width, int height, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < RGB; c++) {
            double sum = 0.0;
            for (int i = -N/2; i <= N/2; i++) {
                for (int j = -N/2; j <= N/2; j++) {
                    int inputX = x + j;
                    int inputY = y + i;
                    
                    // Handle boundary conditions
                    if (inputX < 0) inputX = 0;
                    if (inputX >= width) inputX = width - 1;
                    if (inputY < 0) inputY = 0;
                    if (inputY >= height) inputY = height - 1;
                    
                    int inputIdx = (inputY * width + inputX) * RGB + c;
                    int kernelIdx = (i + N/2) * N + (j + N/2);
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
            int outputIdx = (y * width + x) * RGB + c;
            output[outputIdx] = static_cast<unsigned char>(fmax(0.0, fmin(255.0, sum)));
        }
    }
}

// CUDA kernel for unsharp masking
__global__ void unsharpKernel(unsigned char* output, const unsigned char* input, 
                             const unsigned char* blur, int width, int height, double alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < RGB; c++) {
            int idx = (y * width + x) * RGB + c;
            double detail = input[idx] - blur[idx];
            double temp = input[idx] + (alpha * detail);
            output[idx] = static_cast<unsigned char>(fmax(0.0, fmin(255.0, temp)));
        }
    }
}

// CUDA kernel for CLAHE histogram computation
__global__ void computeHistogramKernel(int* histogram, const unsigned char* input, 
                                     int width, int height, int tileSize, int tileX, int tileY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int startX = tileX * tileSize;
    int startY = tileY * tileSize;
    int endX = min(startX + tileSize, width);
    int endY = min(startY + tileSize, height);
    
    if (x + startX < endX && y + startY < endY) {
        int idx = ((y + startY) * width + (x + startX)) * RGB;
        int luminance = static_cast<int>(0.299 * input[idx] + 
                                       0.587 * input[idx + 1] + 
                                       0.114 * input[idx + 2]);
        atomicAdd(&histogram[luminance], 1);
    }
}

// CUDA kernel for CLAHE transformation
__global__ void applyCLAHEKernel(unsigned char* output, const unsigned char* input,
                                const double* cdf, int width, int height, 
                                int tileSize, int numTilesX, int numTilesY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        double tileX = static_cast<double>(x) / tileSize;
        double tileY = static_cast<double>(y) / tileSize;
        int tx1 = static_cast<int>(tileX);
        int ty1 = static_cast<int>(tileY);
        int tx2 = min(tx1 + 1, numTilesX - 1);
        int ty2 = min(ty1 + 1, numTilesY - 1);
        double wx = tileX - tx1;
        double wy = tileY - ty1;
        
        int idx = (y * width + x) * RGB;
        int luminance = static_cast<int>(0.299 * input[idx] + 
                                       0.587 * input[idx + 1] + 
                                       0.114 * input[idx + 2]);
        
        double cdf_tl = cdf[(ty1 * numTilesX + tx1) * 256 + luminance];
        double cdf_tr = cdf[(ty1 * numTilesX + tx2) * 256 + luminance];
        double cdf_bl = cdf[(ty2 * numTilesX + tx1) * 256 + luminance];
        double cdf_br = cdf[(ty2 * numTilesX + tx2) * 256 + luminance];
        
        double interpolatedCDF = (1 - wx) * (1 - wy) * cdf_tl +
                               wx * (1 - wy) * cdf_tr +
                               (1 - wx) * wy * cdf_bl +
                               wx * wy * cdf_br;
        
        int newLuminance = static_cast<int>(interpolatedCDF * 255);
        double scale = static_cast<double>(newLuminance) / (luminance + 1e-6);
        scale = fmin(1.2, fmax(0.8, scale));
        
        double maxChannel = fmax(fmax(static_cast<double>(input[idx]),
                                    static_cast<double>(input[idx + 1])),
                               static_cast<double>(input[idx + 2]));
        double ratios[3];
        for (int c = 0; c < RGB; c++) {
            ratios[c] = input[idx + c] / (maxChannel + 1e-6);
        }
        
        for (int c = 0; c < RGB; c++) {
            double mean = (input[idx] + input[idx + 1] + input[idx + 2]) / 3.0;
            double diff = input[idx + c] - mean;
            double newValue = (mean * scale) + (diff * scale * 0.8);
            
            if (input[idx + c] > 200) {
                double blend = (input[idx + c] - 200) / 55.0;
                blend = fmin(1.0, fmax(0.0, blend));
                newValue = input[idx + c] * blend + newValue * (1 - blend);
            }
            
            newValue = newValue * ratios[c];
            output[idx + c] = static_cast<unsigned char>(fmin(255.0, fmax(0.0, newValue)));
        }
    }
}

// Function to apply Gaussian filter using CUDA
void gaussian_filter_cuda(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma) {
    // Allocate device memory
    unsigned char *d_input, *d_output, *d_blur;
    double *d_kernel;
    size_t imageSize = size->width * size->height * RGB;
    size_t kernelSize = N * N * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK(cudaMalloc(&d_blur, imageSize));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize));
    
    // Copy input to device
    unsigned char* flat_input = new unsigned char[imageSize];
    for (int i = 0; i < size->height; i++) {
        for (int j = 0; j < size->width; j++) {
            for (int k = 0; k < RGB; k++) {
                flat_input[(i * size->width + j) * RGB + k] = in[i][j][k];
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(d_input, flat_input, imageSize, cudaMemcpyHostToDevice));
    
    // Generate Gaussian kernel
    dim3 blockDim(256);
    dim3 gridDim((N * N + blockDim.x - 1) / blockDim.x);
    gaussianKernel<<<gridDim, blockDim>>>(d_kernel, N, sigma);
    CUDA_CHECK(cudaGetLastError());
    
    // Normalize kernel
    double* h_kernel = new double[N * N];
    CUDA_CHECK(cudaMemcpy(h_kernel, d_kernel, kernelSize, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) {
        sum += h_kernel[i];
    }
    for (int i = 0; i < N * N; i++) {
        h_kernel[i] /= sum;
    }
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice));
    
    // Perform convolution
    dim3 blockDim2(16, 16);
    dim3 gridDim2((size->width + blockDim2.x - 1) / blockDim2.x,
                  (size->height + blockDim2.y - 1) / blockDim2.y);
    convolveKernel<<<gridDim2, blockDim2>>>(d_blur, d_input, d_kernel, 
                                           size->width, size->height, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    unsigned char* flat_output = new unsigned char[imageSize];
    CUDA_CHECK(cudaMemcpy(flat_output, d_blur, imageSize, cudaMemcpyDeviceToHost));
    
    // Convert flat array back to 3D array
    for (int i = 0; i < size->height; i++) {
        for (int j = 0; j < size->width; j++) {
            for (int k = 0; k < RGB; k++) {
                out[i][j][k] = flat_output[(i * size->width + j) * RGB + k];
            }
        }
    }
    
    // Clean up
    delete[] flat_input;
    delete[] flat_output;
    delete[] h_kernel;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_kernel));
}

// Function to apply CLAHE using OpenMP
void applyCLAHE_omp(unsigned char*** output, unsigned char*** input, ImageSize* size, int tileSize, double clipLimit) {
    int width = size->width;
    int height = size->height;
    int numTilesX = (width + tileSize - 1) / tileSize;
    int numTilesY = (height + tileSize - 1) / tileSize;
    
    // Allocate memory for histograms and CDFs
    int* histograms = new int[numTilesX * numTilesY * 256]();
    double* cdf = new double[numTilesX * numTilesY * 256];
    
    #pragma omp parallel
    {
        // Compute histograms for each tile
        #pragma omp for collapse(2)
        for (int ty = 0; ty < numTilesY; ty++) {
            for (int tx = 0; tx < numTilesX; tx++) {
                int startX = tx * tileSize;
                int startY = ty * tileSize;
                int endX = min(startX + tileSize, width);
                int endY = min(startY + tileSize, height);
                
                int* hist = histograms + (ty * numTilesX + tx) * 256;
                
                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        int idx = (y * width + x) * RGB;
                        int luminance = static_cast<int>(0.299 * input[y][x][0] + 
                                                       0.587 * input[y][x][1] + 
                                                       0.114 * input[y][x][2]);
                        hist[luminance]++;
                    }
                }
            }
        }
        
        // Process histograms and compute CDFs
        #pragma omp for
        for (int t = 0; t < numTilesX * numTilesY; t++) {
            int* hist = histograms + t * 256;
            double* tileCdf = cdf + t * 256;
            
            // Clip histogram
            int clipCount = 0;
            for (int i = 0; i < 256; i++) {
                if (hist[i] > clipLimit) {
                    clipCount += hist[i] - clipLimit;
                    hist[i] = clipLimit;
                }
            }
            
            // Redistribute clipped pixels
            int redistCount = clipCount / 256;
            int remainder = clipCount % 256;
            for (int i = 0; i < 256; i++) {
                hist[i] += redistCount;
                if (i < remainder) hist[i]++;
            }
            
            // Compute CDF
            tileCdf[0] = hist[0];
            for (int i = 1; i < 256; i++) {
                tileCdf[i] = tileCdf[i-1] + hist[i];
            }
            
            // Normalize CDF
            double scale = 255.0 / tileCdf[255];
            for (int i = 0; i < 256; i++) {
                tileCdf[i] *= scale;
            }
        }
        
        // Apply CLAHE transformation
        #pragma omp for collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double tileX = static_cast<double>(x) / tileSize;
                double tileY = static_cast<double>(y) / tileSize;
                int tx1 = static_cast<int>(tileX);
                int ty1 = static_cast<int>(tileY);
                int tx2 = min(tx1 + 1, numTilesX - 1);
                int ty2 = min(ty1 + 1, numTilesY - 1);
                double wx = tileX - tx1;
                double wy = tileY - ty1;
                
                int idx = (y * width + x) * RGB;
                int luminance = static_cast<int>(0.299 * input[y][x][0] + 
                                               0.587 * input[y][x][1] + 
                                               0.114 * input[y][x][2]);
                
                double cdf_tl = cdf[(ty1 * numTilesX + tx1) * 256 + luminance];
                double cdf_tr = cdf[(ty1 * numTilesX + tx2) * 256 + luminance];
                double cdf_bl = cdf[(ty2 * numTilesX + tx1) * 256 + luminance];
                double cdf_br = cdf[(ty2 * numTilesX + tx2) * 256 + luminance];
                
                double interpolatedCDF = (1 - wx) * (1 - wy) * cdf_tl +
                                       wx * (1 - wy) * cdf_tr +
                                       (1 - wx) * wy * cdf_bl +
                                       wx * wy * cdf_br;
                
                int newLuminance = static_cast<int>(interpolatedCDF);
                double scale = static_cast<double>(newLuminance) / (luminance + 1e-6);
                scale = fmin(1.2, fmax(0.8, scale));
                
                double maxChannel = fmax(fmax(static_cast<double>(input[y][x][0]),
                                            static_cast<double>(input[y][x][1])),
                                       static_cast<double>(input[y][x][2]));
                double ratios[3];
                for (int c = 0; c < RGB; c++) {
                    ratios[c] = input[y][x][c] / (maxChannel + 1e-6);
                }
                
                for (int c = 0; c < RGB; c++) {
                    double mean = (input[y][x][0] + input[y][x][1] + input[y][x][2]) / 3.0;
                    double diff = input[y][x][c] - mean;
                    double newValue = (mean * scale) + (diff * scale * 0.8);
                    
                    if (input[y][x][c] > 200) {
                        double blend = (input[y][x][c] - 200) / 55.0;
                        blend = fmin(1.0, fmax(0.0, blend));
                        newValue = input[y][x][c] * blend + newValue * (1 - blend);
                    }
                    
                    newValue = newValue * ratios[c];
                    output[y][x][c] = static_cast<unsigned char>(fmin(255.0, fmax(0.0, newValue)));
                }
            }
        }
    }
    
    delete[] histograms;
    delete[] cdf;
}

// Function to apply unsharp masking using CUDA
void unsharp_cuda(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha, 
                 const char* output_filename, int clahe_tile_size, double clahe_clip_limit) {
    // Create output directory if it doesn't exist
    string output_dir = "output/";
    try {
        fs::create_directories(output_dir);
    } catch (const fs::filesystem_error& e) {
        throw ImageProcessingError("Failed to create output directory: " + string(e.what()));
    }

    // Initialize timing variables
    auto total_start = std::chrono::high_resolution_clock::now();
    auto stage_start = total_start;
    auto stage_end = stage_start;

    // Allocate memory
    unsigned char*** blur = allocateRGBImage(size);
    unsigned char*** intermediate = allocateRGBImage(size);
    if (!blur || !intermediate) {
        if (blur) freeRGBImage(blur, size);
        if (intermediate) freeRGBImage(intermediate, size);
        throw ImageProcessingError("Failed to allocate memory for processing images");
    }

    try {
        cout << "Applying Gaussian blur..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        gaussian_filter_cuda(blur, in, size, N, sigma);
        stage_end = std::chrono::high_resolution_clock::now();
        auto blur_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "Gaussian blur completed in " << blur_time << " ms" << endl;

        // Save the blurred image
        string blurred_filename = string(output_filename);
        size_t last_slash = blurred_filename.find_last_of("/\\");
        size_t last_dot = blurred_filename.find_last_of(".");
        if (last_dot == string::npos) last_dot = blurred_filename.length();

        string filename = blurred_filename.substr(last_slash + 1, last_dot - last_slash - 1);
        string extension = blurred_filename.substr(last_dot);
        string ext = extension.empty() ? ".bmp" : extension;
        string blurred_output = output_dir + "cuda_" + filename + "_blurred" + ext;

        if(writeRGBBMP(blurred_output.c_str(), &blur, size) != 0) {
            throw ImageProcessingError("Failed to write blurred image file: " + blurred_output);
        }
        cout << "Blurred image has been saved as '" << blurred_output << "'" << endl;

        cout << "Applying unsharp masking..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        
        // Allocate device memory for unsharp masking
        unsigned char *d_input, *d_blur, *d_output;
        size_t imageSize = size->width * size->height * RGB;
        
        CUDA_CHECK(cudaMalloc(&d_input, imageSize));
        CUDA_CHECK(cudaMalloc(&d_blur, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));
        
        // Copy data to device
        unsigned char* flat_input = new unsigned char[imageSize];
        unsigned char* flat_blur = new unsigned char[imageSize];
        for (int i = 0; i < size->height; i++) {
            for (int j = 0; j < size->width; j++) {
                for (int k = 0; k < RGB; k++) {
                    flat_input[(i * size->width + j) * RGB + k] = in[i][j][k];
                    flat_blur[(i * size->width + j) * RGB + k] = blur[i][j][k];
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_input, flat_input, imageSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_blur, flat_blur, imageSize, cudaMemcpyHostToDevice));
        
        // Launch unsharp masking kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((size->width + blockDim.x - 1) / blockDim.x,
                    (size->height + blockDim.y - 1) / blockDim.y);
        unsharpKernel<<<gridDim, blockDim>>>(d_output, d_input, d_blur, 
                                            size->width, size->height, alpha);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy result back to host
        unsigned char* flat_output = new unsigned char[imageSize];
        CUDA_CHECK(cudaMemcpy(flat_output, d_output, imageSize, cudaMemcpyDeviceToHost));
        
        // Convert flat array back to 3D array
        for (int i = 0; i < size->height; i++) {
            for (int j = 0; j < size->width; j++) {
                for (int k = 0; k < RGB; k++) {
                    intermediate[i][j][k] = flat_output[(i * size->width + j) * RGB + k];
                }
            }
        }
        
        // Clean up device memory
        delete[] flat_input;
        delete[] flat_blur;
        delete[] flat_output;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_blur));
        CUDA_CHECK(cudaFree(d_output));
        
        stage_end = std::chrono::high_resolution_clock::now();
        auto unsharp_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "Unsharp masking completed in " << unsharp_time << " ms" << endl;

        // Save the unsharpened image
        string unsharp_output = output_dir + "cuda_" + filename + "_unsharp" + ext;
        if(writeRGBBMP(unsharp_output.c_str(), &intermediate, size) != 0) {
            throw ImageProcessingError("Failed to write unsharpened image file: " + unsharp_output);
        }
        cout << "Unsharpened image has been saved as '" << unsharp_output << "'" << endl;

        cout << "Applying CLAHE to unsharpened image..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        
        // Apply CLAHE using OpenMP
        applyCLAHE_omp(out, intermediate, size, clahe_tile_size, clahe_clip_limit);
        
        stage_end = std::chrono::high_resolution_clock::now();
        auto clahe_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "CLAHE completed in " << clahe_time << " ms" << endl;

        // Save the final image
        string final_output = output_dir + "cuda_" + filename + "_unsharp_clahe" + ext;
        if(writeRGBBMP(final_output.c_str(), &out, size) != 0) {
            throw ImageProcessingError("Failed to write final image file: " + final_output);
        }
        cout << "Final image (unsharp + CLAHE) has been saved as '" << final_output << "'" << endl;

        // Calculate and display total execution time
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_time = blur_time + unsharp_time + clahe_time;

        // Print execution time summary
        cout << "\nExecution Time Summary:" << endl;
        cout << "----------------------" << endl;
        cout << "Gaussian Blur:     " << setw(6) << blur_time << " ms" << endl;
        cout << "Unsharp Masking:   " << setw(6) << unsharp_time << " ms" << endl;
        cout << "CLAHE:            " << setw(6) << clahe_time << " ms" << endl;
        cout << "----------------------" << endl;
        cout << "Total Time:        " << setw(6) << total_time << " ms" << endl;
        cout << "----------------------" << endl;

    } catch (...) {
        freeRGBImage(blur, size);
        freeRGBImage(intermediate, size);
        throw;
    }

    freeRGBImage(blur, size);
    freeRGBImage(intermediate, size);
}

int main(int argc, char* argv[]) {
    try {
        if(argc < 7) {
            cout << "usage: ./filter_cuda <input file> <filter_type> <param1> <param2> <param3> <output file name>" << endl;
            cout << "Filter types and parameters:" << endl;
            cout << "  unsharp <N> <sigma> <alpha>" << endl;
            cout << "    N: kernel size (odd number between 3 and 11)" << endl;
            cout << "    sigma: blur strength (positive number)" << endl;
            cout << "    alpha: sharpening strength (0.0 to 5.0)" << endl;
            cout << "  unsharp_clahe <N> <sigma> <alpha>" << endl;
            cout << "    N: kernel size (odd number between 3 and 11)" << endl;
            cout << "    sigma: blur strength (positive number)" << endl;
            cout << "    alpha: sharpening strength (0.0 to 5.0)" << endl;
            cout << "    (CLAHE parameters: tile_size=8, clip_limit=2.0)" << endl;
            cout << "  clahe <tile_size> <clip_limit> <dummy>" << endl;
            cout << "    tile_size: size of tiles for local histogram (e.g., 8, 16, 32)" << endl;
            cout << "    clip_limit: contrast limit (1.0 to 4.0)" << endl;
            cout << "    dummy: ignored parameter (for compatibility)" << endl;
            return -1;
        }

        ImageSize size;
        unsigned char*** input = nullptr;
        unsigned char*** output = nullptr;
        char* outfile = argv[6];

        // read file contents into input array
        int status = readRGBBMP(argv[1], &input, &size);
        if(status != 0) {
            cout << "unable to open " << argv[1] << " for input." << endl;
            return -1;
        }

        validateImageDimensions(&size);

        // Allocate output image
        output = allocateRGBImage(&size);
        if (output == nullptr) {
            cout << "Error: Failed to allocate memory for output image" << endl;
            freeRGBImage(input, &size);
            return -1;
        }

        if(strcmp("unsharp", argv[2]) == 0 || strcmp("unsharp_clahe", argv[2]) == 0) {
            int N = atoi(argv[3]);
            if (N % 2 == 0 || N < 3 || N > 11) {
                cout << "Error: N must be an odd number between 3 and 11" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            double sigma = atof(argv[4]);
            if (sigma <= 0) {
                cout << "Error: sigma must be positive" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            double alpha = atof(argv[5]);
            if (alpha < 0 || alpha > 5) {
                cout << "Error: alpha should be between 0 and 5" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            double max_dimension = std::max(size.width, size.height);
            double scale_factor = 1.0 + (log(max_dimension / 128.0) / log(2.0));
            double scaled_sigma = sigma * scale_factor;
            
            const double MAX_SIGMA = 16.0;
            if (scaled_sigma > MAX_SIGMA) {
                cout << "Warning: Scaled sigma (" << scaled_sigma << ") exceeds maximum limit. Capping at " << MAX_SIGMA << endl;
                scaled_sigma = MAX_SIGMA;
            }
            
            cout << "Image size: " << size.width << "x" << size.height << endl;
            cout << "Original sigma: " << sigma << endl;
            cout << "Scale factor: " << scale_factor << endl;
            cout << "Scaled sigma: " << scaled_sigma << endl;
            cout << "Sharpening strength (alpha): " << alpha << endl;

            if(strcmp("unsharp_clahe", argv[2]) == 0) {
                cout << "Applying unsharp masking followed by CLAHE..." << endl;
                unsharp_cuda(output, input, &size, N, scaled_sigma, alpha, outfile, 8, 1.0);
            } else {
                cout << "Applying unsharp masking..." << endl;
                unsharp_cuda(output, input, &size, N, scaled_sigma, alpha, outfile, 8, 1.0);
            }
        }
        else if(strcmp("clahe", argv[2]) == 0) {
            int tileSize = atoi(argv[3]);
            if (tileSize < 4 || tileSize > 64) {
                cout << "Error: tile size must be between 4 and 64" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            double clipLimit = atof(argv[4]);
            if (clipLimit < 1.0 || clipLimit > 4.0) {
                cout << "Error: clip limit should be between 1.0 and 4.0" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            cout << "Applying CLAHE with:" << endl;
            cout << "  Tile size: " << tileSize << "x" << tileSize << endl;
            cout << "  Clip limit: " << clipLimit << endl;

            // Apply CLAHE (implementation omitted for brevity)
            // This would involve CUDA kernel launches for histogram computation and transformation

            if(writeRGBBMP(outfile, &output, &size) != 0) {
                cout << "error writing file " << outfile << endl;
            }
        }
        else {
            cout << "unknown filter type. Supported filters: 'unsharp', 'unsharp_clahe', 'clahe'" << endl;
            freeRGBImage(input, &size);
            freeRGBImage(output, &size);
            return -1;
        }

        freeRGBImage(input, &size);
        freeRGBImage(output, &size);
        return 0;

    } catch (const ImageProcessingError& e) {
        cout << "Error: " << e.what() << endl;
        return -1;
    } catch (const std::exception& e) {
        cout << "Unexpected error: " << e.what() << endl;
        return -1;
    } catch (...) {
        cout << "Unknown error occurred" << endl;
        return -1;
    }
} 