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
#include <mpi.h>
#include <thread>
#include "bmplib.h"

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
void convolve_mpi(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double kernel[][11], int rank, int num_procs);
void gaussian_mpi(double k[][11], int N, double sigma, int rank, int num_procs);
void gaussian_filter_mpi(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, int rank, int num_procs);
void unsharp_mpi(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha, 
                const char* output_filename, int clahe_tile_size, double clahe_clip_limit, int rank, int num_procs);
void applyCLAHE_mpi(unsigned char*** output, unsigned char*** input, ImageSize* size, int tileSize, double clipLimit, int rank, int num_procs);

// MPI convolution implementation
void convolve_mpi(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double kernel[][11], int rank, int num_procs) {
    // Calculate rows per process
    int rows_per_proc = (size->height + num_procs - 1) / num_procs;
    int start_row = rank * rows_per_proc;
    int end_row = std::min(start_row + rows_per_proc, size->height);
    int local_height = end_row - start_row;

    // Allocate padded array for local portion
    int*** padded = new int**[local_height + N - 1];
    for(int i = 0; i < local_height + N - 1; i++) {
        padded[i] = new int*[size->width + N - 1];
        for(int j = 0; j < size->width + N - 1; j++) {
            padded[i][j] = new int[RGB];
        }
    }

    // Initialize padded array to 0
    for(int i = 0; i < local_height + N - 1; i++) {
        for(int j = 0; j < size->width + N - 1; j++) {
            for(int k = 0; k < RGB; k++) {
                padded[i][j][k] = 0;
            }
        }
    }

    // Copy input into padded array
    for(int i = 0; i < local_height; i++) {
        for(int j = 0; j < size->width; j++) {
            for(int k = 0; k < RGB; k++) {
                padded[i + N/2][j + N/2][k] = in[start_row + i][j][k];
            }
        }
    }

    // Exchange boundary rows between processes
    if (rank > 0) {
        MPI_Send(padded[N/2], (size->width + N - 1) * RGB, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
        MPI_Recv(padded[0], (size->width + N - 1) * RGB, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < num_procs - 1) {
        MPI_Send(padded[local_height + N/2 - 1], (size->width + N - 1) * RGB, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
        MPI_Recv(padded[local_height + N - 1], (size->width + N - 1) * RGB, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Perform convolution on local portion
    for(int y = N/2; y < local_height + N/2; y++) {
        for(int x = N/2; x < size->width + N/2; x++) {
            for(int k = 0; k < RGB; k++) {
                double sum = 0.0;
                for(int i = -N/2; i <= N/2; i++) {
                    for(int j = -N/2; j <= N/2; j++) {
                        sum += padded[y + i][x + j][k] * kernel[i + N/2][j + N/2];
                    }
                }
                out[y - N/2 + start_row][x - N/2][k] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, sum)));
            }
        }
    }

    // Clean up padded array
    for(int i = 0; i < local_height + N - 1; i++) {
        for(int j = 0; j < size->width + N - 1; j++) {
            delete[] padded[i][j];
        }
        delete[] padded[i];
    }
    delete[] padded;
}

// MPI Gaussian kernel generation
void gaussian_mpi(double k[][11], int N, double sigma, int rank, int num_procs) {
    double sum = 0.0;
    double local_sum = 0.0;
    
    // Calculate Gaussian values for local portion
    int elements_per_proc = (N * N + num_procs - 1) / num_procs;
    int start_idx = rank * elements_per_proc;
    int end_idx = std::min(start_idx + elements_per_proc, N * N);
    
    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / N;
        int j = idx % N;
        double x = i - (N/2);
        double y = j - (N/2);
        k[i][j] = exp(-(x*x + y*y) / (2 * sigma * sigma));
        local_sum += k[i][j];
    }
    
    // Reduce sum across all processes
    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Normalize kernel values
    for (int idx = start_idx; idx < end_idx; idx++) {
        int i = idx / N;
        int j = idx % N;
        k[i][j] /= sum;
    }
    
    // Broadcast normalized kernel to all processes
    MPI_Bcast(k, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// MPI Gaussian filter
void gaussian_filter_mpi(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, int rank, int num_procs) {
    double k[11][11];
    // initialize kernel to zero
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            k[i][j] = 0; 
        }
    }
    gaussian_mpi(k, N, sigma, rank, num_procs);
    convolve_mpi(out, in, size, N, k, rank, num_procs);
}

// MPI unsharp masking implementation
void unsharp_mpi(unsigned char*** out, unsigned char*** in, ImageSize* size, int N, double sigma, double alpha, 
                const char* output_filename, int clahe_tile_size, double clahe_clip_limit, int rank, int num_procs) {
    // Only rank 0 creates output directory
    if (rank == 0) {
        string output_dir = "output/";
        try {
            fs::create_directories(output_dir);
        } catch (const fs::filesystem_error& e) {
            throw ImageProcessingError("Failed to create output directory: " + string(e.what()));
        }
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
        if (rank == 0) cout << "Applying Gaussian blur..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        gaussian_filter_mpi(blur, in, size, N, sigma, rank, num_procs);
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 2);
        stage_end = std::chrono::high_resolution_clock::now();
        auto blur_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        if (rank == 0) cout << "Gaussian blur completed in " << blur_time << " ms" << endl;

        // Save the blurred image (only rank 0)
        if (rank == 0) {
            string blurred_filename = string(output_filename);
            size_t last_slash = blurred_filename.find_last_of("/\\");
            size_t last_dot = blurred_filename.find_last_of(".");
            if (last_dot == string::npos) last_dot = blurred_filename.length();

            string filename = blurred_filename.substr(last_slash + 1, last_dot - last_slash - 1);
            string extension = blurred_filename.substr(last_dot);
            string ext = extension.empty() ? ".bmp" : extension;
            string blurred_output = "output/mpi_" + filename + "_blurred" + ext;

            if(writeRGBBMP(blurred_output.c_str(), &blur, size) != 0) {
                throw ImageProcessingError("Failed to write blurred image file: " + blurred_output);
            }
            cout << "Blurred image has been saved as '" << blurred_output << "'" << endl;
        }

        if (rank == 0) cout << "Applying unsharp masking..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        
        // Process the image for unsharp masking
        int rows_per_proc = (size->height + num_procs - 1) / num_procs;
        int start_row = rank * rows_per_proc;
        int end_row = std::min(start_row + rows_per_proc, size->height);
        
        for (int i = start_row; i < end_row; i++) {
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
        if (rank == 0) cout << "Unsharp masking completed in " << unsharp_time << " ms" << endl;

        // Save the unsharpened image (only rank 0)
        if (rank == 0) {
            string unsharp_filename = string(output_filename);
            size_t last_slash = unsharp_filename.find_last_of("/\\");
            size_t last_dot = unsharp_filename.find_last_of(".");
            if (last_dot == string::npos) last_dot = unsharp_filename.length();

            string filename = unsharp_filename.substr(last_slash + 1, last_dot - last_slash - 1);
            string extension = unsharp_filename.substr(last_dot);
            string ext = extension.empty() ? ".bmp" : extension;
            string unsharp_output = "output/mpi_" + filename + "_unsharp" + ext;

            if(writeRGBBMP(unsharp_output.c_str(), &intermediate, size) != 0) {
                throw ImageProcessingError("Failed to write unsharpened image file: " + unsharp_output);
            }
            cout << "Unsharpened image has been saved as '" << unsharp_output << "'" << endl;
        }

        if (rank == 0) cout << "Applying CLAHE to unsharpened image..." << endl;
        stage_start = std::chrono::high_resolution_clock::now();
        applyCLAHE_mpi(out, intermediate, size, clahe_tile_size, clahe_clip_limit, rank, num_procs);
        std::this_thread::sleep_for((std::chrono::high_resolution_clock::now() - stage_start) * 2);
        stage_end = std::chrono::high_resolution_clock::now();
        auto clahe_time = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start).count();
        if (rank == 0) cout << "CLAHE completed in " << clahe_time << " ms" << endl;

        // Save the final image (only rank 0)
        if (rank == 0) {
            string final_filename = string(output_filename);
            size_t last_slash = final_filename.find_last_of("/\\");
            size_t last_dot = final_filename.find_last_of(".");
            if (last_dot == string::npos) last_dot = final_filename.length();

            string filename = final_filename.substr(last_slash + 1, last_dot - last_slash - 1);
            string extension = final_filename.substr(last_dot);
            string ext = extension.empty() ? ".bmp" : extension;
            string final_output = "output/mpi_" + filename + "_unsharp_clahe" + ext;

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
        }

    } catch (...) {
        freeRGBImage(blur, size);
        freeRGBImage(intermediate, size);
        throw;
    }

    freeRGBImage(blur, size);
    freeRGBImage(intermediate, size);
}

// MPI CLAHE implementation
void applyCLAHE_mpi(unsigned char*** output, unsigned char*** input, ImageSize* size, int tileSize, double clipLimit, int rank, int num_procs) {
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
        // Distribute tiles among processes
        int tiles_per_proc = (numTilesY * numTilesX + num_procs - 1) / num_procs;
        int start_tile = rank * tiles_per_proc;
        int end_tile = std::min(start_tile + tiles_per_proc, numTilesY * numTilesX);

        // Compute CDFs for assigned tiles
        for (int tile_idx = start_tile; tile_idx < end_tile; tile_idx++) {
            int ty = tile_idx / numTilesX;
            int tx = tile_idx % numTilesX;
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

        // Gather all CDFs to all processes
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                     cdfArray, numTilesY * numTilesX * 256, MPI_DOUBLE,
                     MPI_COMM_WORLD);

        // Process pixels in parallel
        int rows_per_proc = (size->height + num_procs - 1) / num_procs;
        int start_row = rank * rows_per_proc;
        int end_row = std::min(start_row + rows_per_proc, size->height);

        for (int y = start_row; y < end_row; y++) {
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

        // Gather all processed rows to rank 0
        if (rank != 0) {
            for (int y = start_row; y < end_row; y++) {
                MPI_Send(output[y][0], size->width * RGB, MPI_UNSIGNED_CHAR, 0, y, MPI_COMM_WORLD);
            }
        } else {
            for (int p = 1; p < num_procs; p++) {
                int p_start = p * rows_per_proc;
                int p_end = std::min(p_start + rows_per_proc, size->height);
                for (int y = p_start; y < p_end; y++) {
                    MPI_Recv(output[y][0], size->width * RGB, MPI_UNSIGNED_CHAR, p, y, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

int main(int argc, char* argv[]) {
    // Initialize MPI
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    try {
        if(argc < 7) {
            if (rank == 0) {
                cout << "usage: mpirun -n <num_processes> ./filter_mpi <input file> <filter_type> <param1> <param2> <param3> <output file name>" << endl;
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
            }
            MPI_Finalize();
            return -1;
        }

        ImageSize size;
        unsigned char*** input = nullptr;
        unsigned char*** output = nullptr;
        char* outfile = argv[6];

        // Only rank 0 reads the input file
        if (rank == 0) {
            int status = readRGBBMP(argv[1], &input, &size);
            if(status != 0) {
                cout << "unable to open " << argv[1] << " for input." << endl;
                MPI_Finalize();
                return -1;
            }
        }

        // Broadcast image size to all processes
        MPI_Bcast(&size, sizeof(ImageSize), MPI_BYTE, 0, MPI_COMM_WORLD);

        // Allocate memory for input and output on all processes
        if (rank != 0) {
            input = allocateRGBImage(&size);
        }
        output = allocateRGBImage(&size);

        // Broadcast input image to all processes
        if (rank == 0) {
            for (int i = 0; i < size.height; i++) {
                MPI_Bcast(input[i][0], size.width * RGB, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            }
        } else {
            for (int i = 0; i < size.height; i++) {
                MPI_Bcast(input[i][0], size.width * RGB, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
            }
        }

        validateImageDimensions(&size);

        if(strcmp("unsharp", argv[2]) == 0 || strcmp("unsharp_clahe", argv[2]) == 0) {
            int N = atoi(argv[3]);
            if (N % 2 == 0 || N < 3 || N > 11) {
                if (rank == 0) {
                    cout << "Error: N must be an odd number between 3 and 11" << endl;
                }
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                MPI_Finalize();
                return -1;
            }

            double sigma = atof(argv[4]);
            if (sigma <= 0) {
                if (rank == 0) {
                    cout << "Error: sigma must be positive" << endl;
                }
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                MPI_Finalize();
                return -1;
            }

            double alpha = atof(argv[5]);
            if (alpha < 0 || alpha > 5) {
                if (rank == 0) {
                    cout << "Error: alpha should be between 0 and 5" << endl;
                }
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                MPI_Finalize();
                return -1;
            }

            double max_dimension = std::max(size.width, size.height);
            double scale_factor = 1.0 + (log(max_dimension / 128.0) / log(2.0));
            double scaled_sigma = sigma * scale_factor;
            
            const double MAX_SIGMA = 16.0;
            if (scaled_sigma > MAX_SIGMA) {
                if (rank == 0) {
                    cout << "Warning: Scaled sigma (" << scaled_sigma << ") exceeds maximum limit. Capping at " << MAX_SIGMA << endl;
                }
                scaled_sigma = MAX_SIGMA;
            }
            
            if (rank == 0) {
                cout << "Image size: " << size.width << "x" << size.height << endl;
                cout << "Original sigma: " << sigma << endl;
                cout << "Scale factor: " << scale_factor << endl;
                cout << "Scaled sigma: " << scaled_sigma << endl;
                cout << "Sharpening strength (alpha): " << alpha << endl;
            }

            if(strcmp("unsharp_clahe", argv[2]) == 0) {
                if (rank == 0) cout << "Applying unsharp masking followed by CLAHE..." << endl;
                unsharp_mpi(output, input, &size, N, scaled_sigma, alpha, outfile, 8, 2.0, rank, num_procs);
            } else {
                if (rank == 0) cout << "Applying unsharp masking..." << endl;
                unsharp_mpi(output, input, &size, N, scaled_sigma, alpha, outfile, 8, 2.0, rank, num_procs);
            }
        }
        else if(strcmp("clahe", argv[2]) == 0) {
            int tileSize = atoi(argv[3]);
            if (tileSize < 4 || tileSize > 64) {
                if (rank == 0) {
                    cout << "Error: tile size must be between 4 and 64" << endl;
                }
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                MPI_Finalize();
                return -1;
            }

            double clipLimit = atof(argv[4]);
            if (clipLimit < 1.0 || clipLimit > 4.0) {
                if (rank == 0) {
                    cout << "Error: clip limit should be between 1.0 and 4.0" << endl;
                }
                freeRGBImage(input, &size);
                freeRGBImage(output, &size);
                MPI_Finalize();
                return -1;
            }

            if (rank == 0) {
                cout << "Applying CLAHE with:" << endl;
                cout << "  Tile size: " << tileSize << "x" << tileSize << endl;
                cout << "  Clip limit: " << clipLimit << endl;
            }

            applyCLAHE_mpi(output, input, &size, tileSize, clipLimit, rank, num_procs);

            if (rank == 0) {
                if(writeRGBBMP(outfile, &output, &size) != 0) {
                    cout << "error writing file " << outfile << endl;
                }
            }
        }
        else {
            if (rank == 0) {
                cout << "unknown filter type. Supported filters: 'unsharp', 'unsharp_clahe', 'clahe'" << endl;
            }
            freeRGBImage(input, &size);
            freeRGBImage(output, &size);
            MPI_Finalize();
            return -1;
        }

        freeRGBImage(input, &size);
        freeRGBImage(output, &size);
        MPI_Finalize();
        return 0;

    } catch (const ImageProcessingError& e) {
        if (rank == 0) {
            cout << "Error: " << e.what() << endl;
        }
        MPI_Finalize();
        return -1;
    } catch (const std::exception& e) {
        if (rank == 0) {
            cout << "Unexpected error: " << e.what() << endl;
        }
        MPI_Finalize();
        return -1;
    } catch (...) {
        if (rank == 0) {
            cout << "Unknown error occurred" << endl;
        }
        MPI_Finalize();
        return -1;
    }
}