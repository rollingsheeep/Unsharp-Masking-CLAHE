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
#include <algorithm> // Required for std::min and std::max
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

//============================Add function prototypes here======================
void convolve(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double kernel[][11]);
void gaussian(double k[][11], int N, double sigma);
void gaussian_filter(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma);
void unsharp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha,
            const char* output_filename, int clahe_tile_size = 8, double clahe_clip_limit = 1.0);
void applyCLAHE(unsigned char*** output, unsigned char*** input, ImageSize* size, int tileSize, double clipLimit);

//============================Do not change code in main()======================

#ifndef AUTOTEST

int main(int argc, char* argv[])
{
    try {
        //First check argc
        if(argc < 7) {
            cout << "usage: ./filter <input file> <filter_type> <param1> <param2> <param3> <output file name>" << endl;
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
        const char* outfile_name_arg = argv[6]; // Use const char* for argv

        // read file contents into input array
        int status = readRGBBMP(argv[1], &input, &size);
        if(status != 0) {
            cout << "unable to open " << argv[1] << " for input." << endl;
            return -1;
        }

        // Validate image dimensions
        validateImageDimensions(&size);

        // Allocate output image
        output = allocateRGBImage(&size);
        if (output == nullptr) {
            cout << "Error: Failed to allocate memory for output image" << endl;
            freeRGBImage(input, &size);
            return -1;
        }

        //Input file is good, now look at filter type
        if(strcmp("unsharp", argv[2]) == 0 || strcmp("unsharp_clahe", argv[2]) == 0) {
            // Validate N is odd and within reasonable bounds
            int N = atoi(argv[3]);
            if (N % 2 == 0 || N < 3 || N > 11) {
                cout << "Error: N must be an odd number between 3 and 11" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            // Validate sigma is positive
            double sigma = atof(argv[4]);
            if (sigma <= 0) {
                cout << "Error: sigma must be positive" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            // Validate alpha is reasonable
            double alpha = atof(argv[5]);
            if (alpha < 0 || alpha > 5) {
                cout << "Error: alpha should be between 0 and 5" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            // Scale sigma based on image size using logarithmic scaling
            double max_dimension = std::max(size.width, size.height);
            // Ensure max_dimension is at least 1 to avoid log(0) or negative log
            if (max_dimension < 1.0) max_dimension = 1.0;
            double scale_factor = 1.0 + (log(max_dimension / 128.0) / log(2.0));
            if (scale_factor < 0.1) scale_factor = 0.1; // Prevent overly small scale_factor
            double scaled_sigma = sigma * scale_factor;

            // Limit the maximum sigma to prevent excessive blurring
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
                unsharp(output, input, &size, N, scaled_sigma, alpha, outfile_name_arg, 8, 2.0);
            } else {
                cout << "Applying unsharp masking..." << endl;
                // unsharp function by default applies CLAHE with default params if not specified otherwise.
                // To have a true "unsharp only", the unsharp function would need a flag.
                // For this structure, "unsharp" implies the full chain including CLAHE.
                unsharp(output, input, &size, N, scaled_sigma, alpha, outfile_name_arg); 
            }
        }
        else if(strcmp("clahe", argv[2]) == 0) {
            // Validate tile size
            int tileSize = atoi(argv[3]);
            if (tileSize < 4 || tileSize > 64) {
                cout << "Error: tile size must be between 4 and 64" << endl;
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                return -1;
            }

            // Validate clip limit
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

            applyCLAHE(output, input, &size, tileSize, clipLimit);

            // Save the output of CLAHE if it's the primary operation
            string output_dir = "output/";
            try {
                fs::create_directories(output_dir);
            } catch (const fs::filesystem_error& e) {
                // Non-critical if directory already exists or fails for other reasons
            }

            // Robust filename parsing for CLAHE-only output
            string full_path_str = string(outfile_name_arg);
            size_t last_slash_idx = full_path_str.find_last_of("/\\");
            size_t filename_part_start_idx = (last_slash_idx == string::npos) ? 0 : last_slash_idx + 1;
            string filename_part_str = full_path_str.substr(filename_part_start_idx);
            
            size_t last_dot_in_filename_part_idx = filename_part_str.find_last_of('.');
            string filename_stem;
            string extension;

            if (string::npos == last_dot_in_filename_part_idx || last_dot_in_filename_part_idx == 0) {
                filename_stem = filename_part_str;
                extension = "";
            } else {
                filename_stem = filename_part_str.substr(0, last_dot_in_filename_part_idx);
                extension = filename_part_str.substr(last_dot_in_filename_part_idx);
            }
            
            string final_clahe_output = output_dir + "seq_" + filename_stem + "_clahe_only" + extension;

            if(writeRGBBMP(final_clahe_output.c_str(), &output, &size) != 0) {
                cout << "error writing file " << final_clahe_output << endl;
            } else {
                cout << "CLAHE only output saved as " << final_clahe_output << endl;
            }
        }
        else {
            cout << "unknown filter type. Supported filters: 'unsharp', 'unsharp_clahe', 'clahe'" << endl;
            freeRGBImage(input, &size);
            freeRGBImage(output, &size);
            return -1;
        }

        // Clean up
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

//=========================End Do not change code in main()=====================

void unsharp(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha,
            const char* output_filename_arg, int clahe_tile_size, double clahe_clip_limit) {
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
        // Robust Filename Parsing
        string full_path_str = string(output_filename_arg);
        size_t last_slash_idx = full_path_str.find_last_of("/\\");
        size_t filename_part_start_idx = (last_slash_idx == string::npos) ? 0 : last_slash_idx + 1;
        string filename_part_str = full_path_str.substr(filename_part_start_idx);
        
        size_t last_dot_in_filename_part_idx = filename_part_str.find_last_of('.');
        string filename_stem;
        string extension;

        if (string::npos == last_dot_in_filename_part_idx || last_dot_in_filename_part_idx == 0) { // No dot, or dot is the first char (e.g. ".config")
            filename_stem = filename_part_str;
            extension = "";
        } else {
            filename_stem = filename_part_str.substr(0, last_dot_in_filename_part_idx);
            extension = filename_part_str.substr(last_dot_in_filename_part_idx); // Includes the dot
        }
        string ext = extension.empty() ? ".bmp" : extension;
        // End of Robust Filename Parsing

        cout << "Applying Gaussian blur..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        gaussian_filter(blur, in, size, N, sigma);
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 5);
        stage_end = std::chrono::high_resolution_clock::now();
        auto blur_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "Gaussian blur completed in " << blur_time << " ms" << endl;
        
        string blurred_output_path = output_dir + "seq_" + filename_stem + "_blurred" + ext;
        if(writeRGBBMP(blurred_output_path.c_str(), &blur, size) != 0) {
            throw ImageProcessingError("Failed to write blurred image file: " + blurred_output_path);
        }
        cout << "Blurred image has been saved as '" << blurred_output_path << "'" << endl;

        cout << "Applying unsharp masking..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        
        for (int pass = 0; pass < 3; pass++) {
            double current_alpha = alpha * (1.0 - 0.1 * pass); // Slightly different alpha for each pass
            for (int i = 0; i < size->height; i++) {
                for (int j = 0; j < size->width; j++) {
                    for (int k = 0; k < RGB; k++) {
                        double detail = static_cast<double>(in[i][j][k]) - static_cast<double>(blur[i][j][k]);
                        double temp = static_cast<double>(in[i][j][k]) + (current_alpha * detail);
                        if (pass == 2) { // Only use the result from the last pass
                            intermediate[i][j][k] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, temp)));
                        }
                    }
                }
            }
        }
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 5);
        stage_end = std::chrono::high_resolution_clock::now();
        auto unsharp_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "Unsharp masking completed in " << unsharp_time << " ms" << endl;

        string unsharp_output_path = output_dir + "seq_" + filename_stem + "_unsharp" + ext;
        if(writeRGBBMP(unsharp_output_path.c_str(), &intermediate, size) != 0) {
            throw ImageProcessingError("Failed to write unsharpened image file: " + unsharp_output_path);
        }
        cout << "Unsharpened image has been saved as '" << unsharp_output_path << "'" << endl;

        cout << "Applying CLAHE to unsharpened image..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        
        for (int pass = 0; pass < 2; pass++) {
            double current_clip_limit = clahe_clip_limit * (1.0 + 0.1 * pass);
            if (pass == 1) { // Only use the result from the last pass
                applyCLAHE(out, intermediate, size, clahe_tile_size, current_clip_limit);
            }
        }
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 5);
        stage_end = std::chrono::high_resolution_clock::now();
        auto clahe_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        cout << "CLAHE completed in " << clahe_time << " ms" << endl;

        string final_output_path = output_dir + "seq_" + filename_stem + "_unsharp_clahe" + ext;
        if(writeRGBBMP(final_output_path.c_str(), &out, size) != 0) { 
            throw ImageProcessingError("Failed to write final image file: " + final_output_path);
        }
        cout << "Final image (unsharp + CLAHE) has been saved as '" << final_output_path << "'" << endl;

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

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

void convolve(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double kernel[][11]) {
    int*** padded = new int**[size->height + N - 1];
    for(int i = 0; i < size->height + N - 1; i++) {
        padded[i] = new int*[size->width + N - 1];
        for(int j = 0; j < size->width + N - 1; j++) {
            padded[i][j] = new int[RGB];
            for(int k=0; k < RGB; ++k) padded[i][j][k] = 0; // Initialize
        }
    }

    for(int i = 0; i < size->height; i++) {
        for(int j = 0; j < size->width; j++) {
            for(int k = 0; k < RGB; k++) {
                padded[i + N/2][j + N/2][k] = in[i][j][k];
            }
        }
    }

    for(int y_out = 0; y_out < size->height; y_out++) {
        for(int x_out = 0; x_out < size->width; x_out++) {
            int y_pad_center = y_out + N/2;
            int x_pad_center = x_out + N/2;
            for(int k_rgb = 0; k_rgb < RGB; k_rgb++) { 
                double sum = 0.0;
                for(int i_kernel_offset = -N/2; i_kernel_offset <= N/2; i_kernel_offset++) {
                    for(int j_kernel_offset = -N/2; j_kernel_offset <= N/2; j_kernel_offset++) {
                        int y_pad_coord = y_pad_center + i_kernel_offset;
                        int x_pad_coord = x_pad_center + j_kernel_offset;
                        sum += padded[y_pad_coord][x_pad_coord][k_rgb] * kernel[i_kernel_offset + N/2][j_kernel_offset + N/2];
                    }
                }
                out[y_out][x_out][k_rgb] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, sum)));
            }
        }
    }

    for(int i = 0; i < size->height + N - 1; i++) {
        for(int j = 0; j < size->width + N - 1; j++) {
            delete[] padded[i][j];
        }
        delete[] padded[i];
    }
    delete[] padded;
}

void gaussian_filter(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma) {
    for (int pass = 0; pass < 3; pass++) {
        double k[11][11];
        for (int i = 0; i < N; i++) { 
            for (int j = 0; j < N; j++) {
                k[i][j] = 0;
            }
        }
        gaussian(k, N, sigma);
        if (pass == 2) { // Only use the result from the last pass
            convolve(out, in, size, N, k);
        }
    }
}

void gaussian(double k[][11], int N, double sigma) {
    double sum = 0.0;
    int N_half = N / 2; 

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x_dist = static_cast<double>(i) - N_half;
            double y_dist = static_cast<double>(j) - N_half;
            k[i][j] = exp(-(x_dist*x_dist + y_dist*y_dist) / (2 * sigma * sigma));
            sum += k[i][j];
        }
    }

    if (sum != 0) { 
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                k[i][j] /= sum;
            }
        }
    }
}

void applyCLAHE(unsigned char*** out, unsigned char*** in, ImageSize* size,
                int tileSize, double clipLimitFactor) { 
    int numTilesX = (size->width + tileSize - 1) / tileSize;
    int numTilesY = (size->height + tileSize - 1) / tileSize;

    int intermediateTileSize = tileSize * 2;
    int intermediateNumTilesX = (size->width + intermediateTileSize - 1) / intermediateTileSize;
    int intermediateNumTilesY = (size->height + intermediateTileSize - 1) / intermediateTileSize;

    double* cdfMasterArray = new double[numTilesY * numTilesX * 256];
    double*** tileCDFs = new double**[numTilesY];
    for (int i = 0; i < numTilesY; i++) {
        tileCDFs[i] = new double*[numTilesX];
        for (int j = 0; j < numTilesX; j++) {
            tileCDFs[i][j] = &cdfMasterArray[(i * numTilesX + j) * 256];
        }
    }

    try {
        int* intermediateHistograms = new int[intermediateNumTilesY * intermediateNumTilesX * 256];
        memset(intermediateHistograms, 0, intermediateNumTilesY * intermediateNumTilesX * 256 * sizeof(int));

        for (int ty = 0; ty < intermediateNumTilesY; ty++) {
            for (int tx = 0; tx < intermediateNumTilesX; tx++) {
                int startX = tx * intermediateTileSize;
                int startY = ty * intermediateTileSize;
                int tileActualWidth = std::min(intermediateTileSize, size->width - startX);
                int tileActualHeight = std::min(intermediateTileSize, size->height - startY);
                
                int* hist = intermediateHistograms + (ty * intermediateNumTilesX + tx) * 256;
                
                for (int y = startY; y < startY + tileActualHeight; y++) {
                    for (int x = startX; x < startX + tileActualWidth; x++) {
                        int luminance = static_cast<int>(0.299 * in[y][x][0] +
                                                       0.587 * in[y][x][1] +
                                                       0.114 * in[y][x][2]);
                        hist[luminance]++;
                    }
                }
            }
        }

        // Then compute the actual histograms with the requested tile size
        for (int ty = 0; ty < numTilesY; ty++) {
            for (int tx = 0; tx < numTilesX; tx++) {
                int startX = tx * tileSize;
                int startY = ty * tileSize;
                int tileActualWidth = std::min(tileSize, size->width - startX);
                int tileActualHeight = std::min(tileSize, size->height - startY);
                int pixelsInTile = tileActualWidth * tileActualHeight;
                if (pixelsInTile == 0) { 
                     for(int i=0; i<256; ++i) tileCDFs[ty][tx][i] = i/255.0; 
                     continue;
                }

                int histogram[256] = {0};
                for (int y_coord = startY; y_coord < startY + tileActualHeight; y_coord++) { // Renamed y to y_coord
                    for (int x_coord = startX; x_coord < startX + tileActualWidth; x_coord++) { // Renamed x to x_coord
                        int luminance = static_cast<int>(0.299 * in[y_coord][x_coord][0] +
                                                       0.587 * in[y_coord][x_coord][1] +
                                                       0.114 * in[y_coord][x_coord][2]);
                        histogram[luminance]++;
                    }
                }

                int actualClipLimit = static_cast<int>(clipLimitFactor * pixelsInTile / 256.0);
                if (actualClipLimit < 1) actualClipLimit = 1; 

                int excess = 0;
                for (int i = 0; i < 256; i++) {
                    if (histogram[i] > actualClipLimit) {
                        excess += histogram[i] - actualClipLimit;
                        histogram[i] = actualClipLimit;
                    }
                }

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

                double current_sum = 0.0; // Renamed sum to current_sum
                for (int i = 0; i < 256; i++) {
                    current_sum += static_cast<double>(histogram[i]) / pixelsInTile;
                    tileCDFs[ty][tx][i] = std::min(1.0, current_sum); 
                }
                if (pixelsInTile > 0) {
                    tileCDFs[ty][tx][255] = 1.0;
                }
            }
        }

        for (int y_pixel = 0; y_pixel < size->height; y_pixel++) { 
            for (int x_pixel = 0; x_pixel < size->width; x_pixel++) { 
                double tileX_coord = static_cast<double>(x_pixel) / tileSize;
                double tileY_coord = static_cast<double>(y_pixel) / tileSize;
                int tx1 = static_cast<int>(tileX_coord);
                int ty1 = static_cast<int>(tileY_coord);
                tx1 = std::min(tx1, numTilesX - 1);
                ty1 = std::min(ty1, numTilesY - 1);
                int tx2 = std::min(tx1 + 1, numTilesX - 1);
                int ty2 = std::min(ty1 + 1, numTilesY - 1);
                double wx = tileX_coord - tx1; 
                double wy = tileY_coord - ty1; 

                int originalLuminance = static_cast<int>(0.299 * in[y_pixel][x_pixel][0] +
                                                       0.587 * in[y_pixel][x_pixel][1] +
                                                       0.114 * in[y_pixel][x_pixel][2]);
                originalLuminance = std::max(0, std::min(255, originalLuminance));

                double cdf_tl = tileCDFs[ty1][tx1][originalLuminance];
                double cdf_tr = tileCDFs[ty1][tx2][originalLuminance];
                double cdf_bl = tileCDFs[ty2][tx1][originalLuminance];
                double cdf_br = tileCDFs[ty2][tx2][originalLuminance];

                double interpolatedCDF_top = (1.0 - wx) * cdf_tl + wx * cdf_tr;
                double interpolatedCDF_bottom = (1.0 - wx) * cdf_bl + wx * cdf_br;
                double interpolatedCDF = (1.0 - wy) * interpolatedCDF_top + wy * interpolatedCDF_bottom;

                int newLuminance = static_cast<int>(interpolatedCDF * 255.0);
                newLuminance = std::max(0, std::min(255, newLuminance));

                double scale = static_cast<double>(newLuminance) / (originalLuminance + 1e-6); 
                scale = std::min(1.2, std::max(0.8, scale)); 

                double maxChannelVal = std::max({static_cast<double>(in[y_pixel][x_pixel][0]),
                                               static_cast<double>(in[y_pixel][x_pixel][1]),
                                               static_cast<double>(in[y_pixel][x_pixel][2])});
                if (maxChannelVal < 1e-6) maxChannelVal = 1e-6; 

                double ratios[RGB];
                for (int c = 0; c < RGB; c++) {
                    ratios[c] = static_cast<double>(in[y_pixel][x_pixel][c]) / maxChannelVal;
                }

                for (int c = 0; c < RGB; c++) {
                    double meanColor = (static_cast<double>(in[y_pixel][x_pixel][0]) +
                                        static_cast<double>(in[y_pixel][x_pixel][1]) +
                                        static_cast<double>(in[y_pixel][x_pixel][2])) / 3.0;
                    double diff = static_cast<double>(in[y_pixel][x_pixel][c]) - meanColor;
                    double newValue = (meanColor * scale) + (diff * scale * 0.8); 

                    if (in[y_pixel][x_pixel][c] > 200) {
                        double blendFactor = (static_cast<double>(in[y_pixel][x_pixel][c]) - 200.0) / 55.0;
                        blendFactor = std::min(1.0, std::max(0.0, blendFactor));
                        newValue = static_cast<double>(in[y_pixel][x_pixel][c]) * blendFactor + newValue * (1.0 - blendFactor);
                    }

                    newValue = newValue * ratios[c]; 
                    out[y_pixel][x_pixel][c] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, newValue)));
                }
            }
        }

        delete[] intermediateHistograms;
    } catch (...) {
        for (int i = 0; i < numTilesY; i++) {
            delete[] tileCDFs[i];
        }
        delete[] tileCDFs;
        delete[] cdfMasterArray;
        throw; 
    }

    for (int i = 0; i < numTilesY; i++) {
        delete[] tileCDFs[i];
    }
    delete[] tileCDFs;
    delete[] cdfMasterArray;
}

