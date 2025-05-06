#ifndef _BMPLIB_H_
#define _BMPLIB_H_

#ifdef RGB
#undef RGB
#endif
#define RGB 3

// Structure to hold image dimensions
typedef struct {
    int width;
    int height;
} ImageSize;

// Note: the read-write functions return 0 on success, 1 on error.

// read grayscale image from the file specified by filename, into inputImage
int readGSBMP(const char* filename, unsigned char*** inputImage, ImageSize* size);

// write grayscale image to the file specified by filename, from outputImage
int writeGSBMP(const char* filename, unsigned char*** outputImage, ImageSize* size);

// display grayscale image with eog, pause 0.2 seconds. (uses a temp file)
void showGSBMP(unsigned char*** inputImage, ImageSize* size);

// read full-color image from the file specified by filename, into inputImage
int readRGBBMP(const char* filename, unsigned char**** inputImage, ImageSize* size);

// write full-color image to the file specified by filename, from outputImage
int writeRGBBMP(const char* filename, unsigned char**** outputImage, ImageSize* size);

// display full-color image with eog, pause 0.2 seconds. (uses a temp file)
void showRGBBMP(unsigned char**** inputImage, ImageSize* size);

// Helper functions for memory management
unsigned char*** allocateRGBImage(ImageSize* size);
void freeRGBImage(unsigned char*** image, ImageSize* size);
unsigned char** allocateGSImage(ImageSize* size);
void freeGSImage(unsigned char** image, ImageSize* size);

#endif
