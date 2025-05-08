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
#include <omp.h>
#include "bmplib.h"
#include <thread>

using namespace std;
namespace fs = std::filesystem;

// Custom exception class for image processing errors
class ImageProcessingError : public std::runtime_error {
public:
    explicit ImageProcessingError(const string& message) : std::runtime_error(message) {}
};

// Function to validate image dimensions
void validateImageDimensions(const ImageSize* size) {
    if (size->width <= 0 || size->height <= 0) {
        throw ImageProcessingError("Invalid image dimensions: width and height must be positive");
    }
    if (size->width > 10000 || size->height > 10000) {
        throw ImageProcessingError("Image dimensions too large: maximum supported size is 10000x10000");
    }
}

// Function to safely allocate 3D image array
unsigned char*** allocateImage(const ImageSize* size) {
    try {
        unsigned char*** image = new unsigned char**[size->height];
        for (int i = 0; i < size->height; i++) {
            image[i] = new unsigned char*[size->width];
            for (int j = 0; j < size->width; j++) {
                image[i][j] = new unsigned char[RGB];
            }
        }
        return image;
    } catch (const std::bad_alloc& e) {
        throw ImageProcessingError("Failed to allocate memory for image: " + string(e.what()));
    }
}

// Function to safely deallocate 3D image array
void deallocateImage(unsigned char*** image, const ImageSize* size) {
    if (image) {
        for (int i = 0; i < size->height; i++) {
            if (image[i]) {
                for (int j = 0; j < size->width; j++) {
                    delete[] image[i][j];
                }
                delete[] image[i];
            }
        }
        delete[] image;
    }
}

// Function prototypes
void convolve_omp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double kernel[][11]);
void gaussian_omp(double k[][11], int N, double sigma);
void gaussian_filter_omp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma);
void unsharp_omp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha, 
                const char* output_filename, int clahe_tile_size = 8, double clahe_clip_limit = 1.0);
void applyCLAHE_omp(unsigned char*** output, unsigned char*** input, ImageSize* size, int tileSize = 8, double clipLimit = 1.0);

// Parallel convolution implementation
void convolve_omp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double kernel[][11]) {
    // Allocate padded array
    int*** padded = new int**[size->height + N - 1];
    for(int i = 0; i < size->height + N - 1; i++) {
        padded[i] = new int*[size->width + N - 1];
        for(int j = 0; j < size->width + N - 1; j++) {
            padded[i][j] = new int[RGB];
        }
    }

    // Initialize padded array to 0
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < size->height + N - 1; i++) {
        for(int j = 0; j < size->width + N - 1; j++) {
            for(int k = 0; k < RGB; k++) {
                padded[i][j][k] = 0;
            }
        }
    }

    // Copy input into padded array
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < size->height; i++) {
        for(int j = 0; j < size->width; j++) {
            for(int k = 0; k < RGB; k++) {
                padded[i + N/2][j + N/2][k] = in[i][j][k];
            }
        }
    }

    // Perform convolution
    #pragma omp parallel for collapse(2)
    for(int y = N/2; y < size->height + N/2; y++) {
        for(int x = N/2; x < size->width + N/2; x++) {
            for(int k = 0; k < RGB; k++) {
                double sum = 0.0;
                for(int i = -N/2; i <= N/2; i++) {
                    for(int j = -N/2; j <= N/2; j++) {
                        sum += padded[y + i][x + j][k] * kernel[i + N/2][j + N/2];
                    }
                }
                out[y - N/2][x - N/2][k] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, sum)));
            }
        }
    }

    // Clean up padded array
    for(int i = 0; i < size->height + N - 1; i++) {
        for(int j = 0; j < size->width + N - 1; j++) {
            delete[] padded[i][j];
        }
        delete[] padded[i];
    }
    delete[] padded;
}

// Parallel Gaussian kernel generation
void gaussian_omp(double k[][11], int N, double sigma) {
    double sum = 0.0;
    
    // Calculate Gaussian values
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i - (N/2);
            double y = j - (N/2);
            k[i][j] = exp(-(x*x + y*y) / (2 * sigma * sigma));
            sum += k[i][j];
        }
    }
    
    // Normalize kernel values
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            k[i][j] /= sum;
        }
    }
}

// Parallel Gaussian filter
void gaussian_filter_omp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma) {
    double k[11][11];
    // initialize kernel to zero
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            k[i][j] = 0; 
        }
    }
    gaussian_omp(k, N, sigma);
    convolve_omp(out, in, size, N, k);
}

// Parallel unsharp masking implementation
void unsharp_omp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha, 
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

    // Allocate blur image and intermediate image for CLAHE
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
        gaussian_filter_omp(blur, in, size, N, sigma);
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 2);
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
        string blurred_output = output_dir + "omp_" + filename + "_blurred" + ext;

        if(writeRGBBMP(blurred_output.c_str(), &blur, size) != 0) {
            throw ImageProcessingError("Failed to write blurred image file: " + blurred_output);
        }

        cout << "Blurred image has been saved as '" << blurred_output << "'" << endl;

        cout << "Applying unsharp masking..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        
        // Process the image for unsharp masking in parallel
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < size->height; i++) {
            for (int j = 0; j < size->width; j++) {
                for (int k = 0; k < RGB; k++) {
                    double detail = in[i][j][k] - blur[i][j][k];
                    double temp = in[i][j][k] + (alpha * detail);
                    intermediate[i][j][k] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, temp)));
                }
            }
        }
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 2);
        stage_end = std::chrono::high_resolution_clock::now();
        auto unsharp_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "Unsharp masking completed in " << unsharp_time << " ms" << endl;

        // Save the unsharpened image
        string unsharp_output = output_dir + "omp_" + filename + "_unsharp" + ext;
        if(writeRGBBMP(unsharp_output.c_str(), &intermediate, size) != 0) {
            throw ImageProcessingError("Failed to write unsharpened image file: " + unsharp_output);
        }
        cout << "Unsharpened image has been saved as '" << unsharp_output << "'" << endl;

        cout << "Applying CLAHE to unsharpened image..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        applyCLAHE_omp(out, intermediate, size, clahe_tile_size, clahe_clip_limit);
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 2);
        stage_end = std::chrono::high_resolution_clock::now();
        auto clahe_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "CLAHE completed in " << clahe_time << " ms" << endl;

        // Save the final image
        string final_output = output_dir + "omp_" + filename + "_unsharp_clahe" + ext;
        if(writeRGBBMP(final_output.c_str(), &out, size) != 0) {
            throw ImageProcessingError("Failed to write final image file: " + final_output);
        }
        cout << "Final image (unsharp + CLAHE) has been saved as '" << final_output << "'" << endl;

        // Calculate and display total execution time
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

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

// Parallel CLAHE implementation
void applyCLAHE_omp(unsigned char*** output, unsigned char*** input, ImageSize* size, 
                    int tileSize, double clipLimit) {
    int numTilesX = (size->width + tileSize - 1) / tileSize;
    int numTilesY = (size->height + tileSize - 1) / tileSize;
    int pixelsPerTile = tileSize * tileSize;
    int clipLimitPixels = (int)(clipLimit * pixelsPerTile / 256);

    // Pre-allocate CDF array
    double* cdfArray = new double[numTilesY * numTilesX * 256];
    double*** tileCDFs = new double**[numTilesY];
    for (int i = 0; i < numTilesY; i++) {
        tileCDFs[i] = new double*[numTilesX];
        for (int j = 0; j < numTilesX; j++) {
            tileCDFs[i][j] = &cdfArray[(i * numTilesX + j) * 256];
        }
    }

    try {
        // Compute CDFs for all tiles in parallel
        #pragma omp parallel for collapse(2)
        for (int ty = 0; ty < numTilesY; ty++) {
            for (int tx = 0; tx < numTilesX; tx++) {
                int startX = tx * tileSize;
                int startY = ty * tileSize;
                int width = std::min(tileSize, size->width - startX);
                int height = std::min(tileSize, size->height - startY);

                // Local histogram for this tile
                int histogram[256] = {0};
                
                // Compute histogram
                for (int y = startY; y < startY + height; y++) {
                    for (int x = startX; x < startX + width; x++) {
                        int luminance = (int)(0.299 * input[y][x][0] + 
                                            0.587 * input[y][x][1] + 
                                            0.114 * input[y][x][2]);
                        histogram[luminance]++;
                    }
                }

                // Clip histogram
                int excess = 0;
                for (int i = 0; i < 256; i++) {
                    if (histogram[i] > clipLimitPixels) {
                        excess += histogram[i] - clipLimitPixels;
                        histogram[i] = clipLimitPixels;
                    }
                }

                // Redistribute excess
                if (excess > 0) {
                    int increment = excess / 256;
                    int remainder = excess % 256;
                    for (int i = 0; i < 256; i++) {
                        histogram[i] += increment;
                        if (remainder > 0) {
                            histogram[i]++;
                            remainder--;
                        }
                    }
                }

                // Compute CDF
                double sum = 0.0;
                for (int i = 0; i < 256; i++) {
                    sum += (double)histogram[i] / (width * height);
                    tileCDFs[ty][tx][i] = sum;
                }
            }
        }

        // Process pixels in parallel
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < size->height; y++) {
            for (int x = 0; x < size->width; x++) {
                double tileX = (double)x / tileSize;
                double tileY = (double)y / tileSize;
                int tx1 = static_cast<int>(tileX);
                int ty1 = static_cast<int>(tileY);
                int tx2 = std::min(tx1 + 1, numTilesX - 1);
                int ty2 = std::min(ty1 + 1, numTilesY - 1);
                double wx = tileX - tx1;
                double wy = tileY - ty1;

                int originalLuminance = (int)(0.299 * input[y][x][0] + 
                                            0.587 * input[y][x][1] + 
                                            0.114 * input[y][x][2]);

                double cdf_tl = tileCDFs[ty1][tx1][originalLuminance];
                double cdf_tr = tileCDFs[ty1][tx2][originalLuminance];
                double cdf_bl = tileCDFs[ty2][tx1][originalLuminance];
                double cdf_br = tileCDFs[ty2][tx2][originalLuminance];

                double interpolatedCDF = (1 - wx) * (1 - wy) * cdf_tl +
                                       wx * (1 - wy) * cdf_tr +
                                       (1 - wx) * wy * cdf_bl +
                                       wx * wy * cdf_br;

                int newLuminance = static_cast<int>(interpolatedCDF * 255);
                double scale = (double)newLuminance / (originalLuminance + 1e-6);
                scale = std::min(1.2, std::max(0.8, scale));

                double maxChannel = std::max({(double)input[y][x][0], 
                                           (double)input[y][x][1], 
                                           (double)input[y][x][2]});
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
                        blend = std::min(1.0, std::max(0.0, blend));
                        newValue = input[y][x][c] * blend + newValue * (1 - blend);
                    }

                    newValue = newValue * ratios[c];
                    output[y][x][c] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, newValue)));
                }
            }
        }
    } catch (...) {
        delete[] cdfArray;
        delete[] tileCDFs;
        throw;
    }

    delete[] cdfArray;
    delete[] tileCDFs;
}

#ifndef AUTOTEST

int main(int argc, char* argv[]) {
    try {
        if(argc < 7) {
            cout << "usage: ./filter_omp <input file> <filter_type> <param1> <param2> <param3> <output file name>" << endl;
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

        int status = readRGBBMP(argv[1], &input, &size);
        if(status != 0) {
            cout << "unable to open " << argv[1] << " for input." << endl;
            return -1;
        }

        validateImageDimensions(&size);

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
                unsharp_omp(output, input, &size, N, scaled_sigma, alpha, outfile, 8, 2.0);
            } else {
                cout << "Applying unsharp masking..." << endl;
                unsharp_omp(output, input, &size, N, scaled_sigma, alpha, outfile);
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

            applyCLAHE_omp(output, input, &size, tileSize, clipLimit);

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

#endif 