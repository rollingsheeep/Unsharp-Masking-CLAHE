#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <windows.h>
#include "bmplib.h"

using namespace std;
using std::cout;
using std::cin;
using std::endl;

typedef unsigned char uint8;
typedef unsigned short int uint16;
typedef unsigned int uint32;

// Maximum reasonable dimensions for a BMP file
const int MAX_REASONABLE_WIDTH = 8192;  // 8K width
const int MAX_REASONABLE_HEIGHT = 8192; // 8K height

//#define BMP_BIG_ENDIAN
#define BYTE_SWAP(num) (((num>>24)&0xff) | ((num<<8)&&0xff0000) | ((num>>8)&0xff00) | ((num<<24)&0xff000000))

typedef struct { 
   uint8    bfType1; 
   uint8    bfType2; 
   uint32   bfSize;
   uint16   bfReserved1; 
   uint16   bfReserved2; 
   uint32   bfOffBits; 
   uint32   biSize;          // size of structure, in bytes
   uint32   biWidth;         // bitmap width, in pixels
   uint32   biHeight;        // bitmap height, in pixels
   uint16   biPlanes;        // see below
   uint16   biBitCount;      // see below
   uint32   biCompression;   // see below
   uint32   biSizeImage;     // see below
   uint32   biXPelsPerMeter; // see below
   uint32   biYPelsPerMeter; // see below
   uint32   biClrUsed;       // see below
   uint32   biClrImportant;  // see below
} BMP_FILE_HEADER, *PBMP_FILE_HEADER; 

typedef struct {
   uint8 rgbBlue;
   uint8 rgbGreen;
   uint8 rgbRed;
} BMP_RGB_TRIPLE;

void write_hdr(uint8 *hdr, int *hdr_idx, uint32 data, uint32 size) {
    for(uint32 i = 0; i < size; i++) {
        hdr[(*hdr_idx)++] = (uint8)(data & 0xff);
        data >>= 8;
    }
}

// Memory management functions
unsigned char*** allocateRGBImage(ImageSize* size) {
    unsigned char*** image = new unsigned char**[size->height];
    for(int i = 0; i < size->height; i++) {
        image[i] = new unsigned char*[size->width];
        for(int j = 0; j < size->width; j++) {
            image[i][j] = new unsigned char[RGB];
        }
    }
    return image;
}

void freeRGBImage(unsigned char*** image, ImageSize* size) {
    for(int i = 0; i < size->height; i++) {
        for(int j = 0; j < size->width; j++) {
            delete[] image[i][j];
        }
        delete[] image[i];
    }
    delete[] image;
}

unsigned char** allocateGSImage(ImageSize* size) {
    unsigned char** image = new unsigned char*[size->height];
    for(int i = 0; i < size->height; i++) {
        image[i] = new unsigned char[size->width];
    }
    return image;
}

void freeGSImage(unsigned char** image, ImageSize* size) {
    for(int i = 0; i < size->height; i++) {
        delete[] image[i];
    }
    delete[] image;
}

int readRGBBMP(const char* filename, unsigned char**** inputImage, ImageSize* size)
{
   FILE *file;
   BMP_FILE_HEADER bfh;

   // Open the file
   if (!(file = fopen(filename, "rb"))) {
      cout << "Cannot open file: " << filename << endl;
      return 1;
   }

   // Read BMP signature
   uint8 type[2];
   if (fread(type, 1, 2, file) != 2 || type[0] != 'B' || type[1] != 'M') {
      cout << "Not a BMP file: " << filename << endl;
      fclose(file);
      return 1;
   }

   // Read file header fields
   if (fread(&bfh.bfSize, 4, 1, file) != 1 ||
       fread(&bfh.bfReserved1, 2, 1, file) != 1 ||
       fread(&bfh.bfReserved2, 2, 1, file) != 1 ||
       fread(&bfh.bfOffBits, 4, 1, file) != 1) {
      cout << "Error reading BMP file header from file: " << filename << endl;
      fclose(file);
      return 1;
   }

   // Read DIB header size
   uint32 headerSize;
   if (fread(&headerSize, 4, 1, file) != 1) {
      cout << "Error reading DIB header size from file: " << filename << endl;
      fclose(file);
      return 1;
   }

   // Read common fields that exist in all header formats
   if (fread(&bfh.biWidth, 4, 1, file) != 1 ||
       fread(&bfh.biHeight, 4, 1, file) != 1 ||
       fread(&bfh.biPlanes, 2, 1, file) != 1 ||
       fread(&bfh.biBitCount, 2, 1, file) != 1 ||
       fread(&bfh.biCompression, 4, 1, file) != 1 ||
       fread(&bfh.biSizeImage, 4, 1, file) != 1 ||
       fread(&bfh.biXPelsPerMeter, 4, 1, file) != 1 ||
       fread(&bfh.biYPelsPerMeter, 4, 1, file) != 1 ||
       fread(&bfh.biClrUsed, 4, 1, file) != 1 ||
       fread(&bfh.biClrImportant, 4, 1, file) != 1) {
      cout << "Error reading DIB header from file: " << filename << endl;
      fclose(file);
      return 1;
   }

   // Skip any additional fields in newer header formats
   if (headerSize > 40) {
      if (fseek(file, headerSize - 40, SEEK_CUR) != 0) {
         cout << "Error skipping additional header fields in file: " << filename << endl;
         fclose(file);
         return 1;
      }
   }

   // Validate bit depth
   if (bfh.biBitCount != 24) {
      cout << "Error: Only 24-bit BMP files are supported. This file has " << bfh.biBitCount << " bits per pixel." << endl;
      fclose(file);
      return 1;
   }

   // Set and validate image dimensions
   size->width = bfh.biWidth;
   size->height = bfh.biHeight;

   // Validate dimensions
   if (size->width <= 0 || size->height <= 0 ||
       size->width > MAX_REASONABLE_WIDTH || size->height > MAX_REASONABLE_HEIGHT) {
      cout << "Error: Invalid or excessively large image dimensions ("
           << size->width << "x" << size->height << ") in file: " << filename << endl;
      fclose(file);
      return 1;
   }

   // Allocate memory for the image
   *inputImage = allocateRGBImage(size);
   if (*inputImage == nullptr) {
      cout << "Error: Memory allocation failed for image from file: " << filename << endl;
      fclose(file);
      return 1;
   }

   // Seek to the start of pixel data
   if (fseek(file, bfh.bfOffBits, SEEK_SET) != 0) {
      cout << "Error seeking to pixel data in file: " << filename << endl;
      freeRGBImage(*inputImage, size);
      fclose(file);
      return 1;
   }

   // Calculate row padding (BMP rows are padded to 4-byte boundaries)
   int row_padded = (size->width * 3 + 3) & (~3);
   unsigned char* row_buffer = new unsigned char[row_padded];
   if (!row_buffer) {
      cout << "Error: Memory allocation failed for row buffer" << endl;
      freeRGBImage(*inputImage, size);
      fclose(file);
      return 1;
   }

   // Read the image data
   for(int i = 0; i < size->height; i++) {
      if (fread(row_buffer, 1, row_padded, file) != (size_t)row_padded) {
         cout << "Error reading pixel data row from file: " << filename << endl;
         delete[] row_buffer;
         freeRGBImage(*inputImage, size);
         fclose(file);
         return 1;
      }

      for(int j = 0; j < size->width; j++) {
         // BMP stores pixels as BGR, convert to RGB
         (*inputImage)[size->height-1-i][j][0] = row_buffer[j*3 + 2]; // Red
         (*inputImage)[size->height-1-i][j][1] = row_buffer[j*3 + 1]; // Green
         (*inputImage)[size->height-1-i][j][2] = row_buffer[j*3 + 0]; // Blue
      }
   }

   delete[] row_buffer;
   fclose(file);
   return 0;
}

int writeRGBBMP(const char* filename, unsigned char**** outputImage, ImageSize* size)
{
   uint8 hdr[54];
   int hdr_idx = 0;

   BMP_FILE_HEADER bfh;

   // file pointer
   FILE *file;
   
   bfh.bfType1 = 'B';
   bfh.bfType2 = 'M';
   bfh.bfSize = 0x36 + size->width * size->height * RGB;
   bfh.bfReserved1 = 0x0;
   bfh.bfReserved2 = 0x0;
   bfh.bfOffBits = 0x36;
  
   bfh.biSize = 0x28;
   bfh.biWidth = size->width;
   bfh.biHeight = size->height;
   bfh.biPlanes = 1;
   bfh.biBitCount = 24;
   bfh.biCompression  = 0;
   bfh.biSizeImage = sizeof(BMP_RGB_TRIPLE) * size->width * size->height;
   bfh.biXPelsPerMeter = 2400;
   bfh.biYPelsPerMeter = 2400;
   bfh.biClrUsed = 0;
   bfh.biClrImportant = 0;

   write_hdr(hdr, &hdr_idx, bfh.bfType1, sizeof(uint8));
   write_hdr(hdr, &hdr_idx, bfh.bfType2, sizeof(uint8));
   write_hdr(hdr, &hdr_idx, bfh.bfSize, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.bfReserved1, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.bfReserved2, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.bfOffBits, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biSize, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biWidth, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biHeight, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biPlanes, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.biBitCount, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.biCompression, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biSizeImage, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biXPelsPerMeter, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biYPelsPerMeter, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biClrUsed, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biClrImportant, sizeof(uint32));

   // write result bmp file
   if (!(file=fopen(filename,"wb")))
      {
         cout << "Cannot open file: " << filename << endl;
         return(1);
      }
   fwrite(&hdr, sizeof(unsigned char), 0x36, file);

   // Write the image data
   for(int i = 0; i < size->height; i++) {
      for(int j = 0; j < size->width; j++) {
         for(int k = 0; k < RGB; k++) {
            fwrite(&((*outputImage)[size->height-1-i][j][RGB-1-k]), sizeof(uint8), 1, file);
         }
      }
   }

   fclose(file);
   return 0;
}

int readGSBMP(const char* filename, unsigned char*** inputImage, ImageSize* size)
{
   uint8 type[2];
   int headersize = 0;
   BMP_FILE_HEADER bfh;

   /* file pointer */
   FILE *file;
   
   /* read input bmp into the data matrix */
   if (!(file=fopen(filename,"rb")))
      {
         cout << "Cannot open file: " << filename <<endl;
         return(1);
      }

   fread(type, sizeof(unsigned char), 0x2, file);
   if(type[0] != 'B' && type[1] != 'M'){
      cout << "Not a BMP file" << endl;
      return(1);
   }
   fseek(file, 8, SEEK_CUR);
   fread(&headersize, sizeof(uint32), 1, file);
#ifdef BMP_BIG_ENDIAN
   headersize = BYTE_SWAP(headersize); 
#endif

   fseek(file, headersize, SEEK_SET);
   fread(&bfh, sizeof(BMP_FILE_HEADER), 1, file);

   // Set image dimensions
   size->width = bfh.biWidth;
   size->height = bfh.biHeight;

   // Allocate memory for the image
   *inputImage = allocateGSImage(size);

   // Read the image data
   for(int i = 0; i < size->height; i++) {
      for(int j = 0; j < size->width; j++) {
         fread(&((*inputImage)[size->height-1-i][j]), sizeof(uint8), 1, file);
      }
   }

   fclose(file);
   return 0;
}

int writeGSBMP(const char* filename, unsigned char*** outputImage, ImageSize* size)
{
   uint8 hdr[54];
   int hdr_idx = 0;
   BMP_FILE_HEADER bfh;

   // file pointer
   FILE *file;
   
   bfh.bfType1 = 'B';
   bfh.bfType2 = 'M';
   bfh.bfSize = 0x36 + size->width * size->height;
   bfh.bfReserved1 = 0x0;
   bfh.bfReserved2 = 0x0;
   bfh.bfOffBits = 0x36;
  
   bfh.biSize = 0x28;
   bfh.biWidth = size->width;
   bfh.biHeight = size->height;
   bfh.biPlanes = 1;
   bfh.biBitCount = 8;
   bfh.biCompression  = 0;
   bfh.biSizeImage = size->width * size->height;
   bfh.biXPelsPerMeter = 2400;
   bfh.biYPelsPerMeter = 2400;
   bfh.biClrUsed = 256;
   bfh.biClrImportant = 256;

   write_hdr(hdr, &hdr_idx, bfh.bfType1, sizeof(uint8));
   write_hdr(hdr, &hdr_idx, bfh.bfType2, sizeof(uint8));
   write_hdr(hdr, &hdr_idx, bfh.bfSize, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.bfReserved1, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.bfReserved2, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.bfOffBits, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biSize, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biWidth, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biHeight, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biPlanes, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.biBitCount, sizeof(uint16));
   write_hdr(hdr, &hdr_idx, bfh.biCompression, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biSizeImage, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biXPelsPerMeter, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biYPelsPerMeter, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biClrUsed, sizeof(uint32));
   write_hdr(hdr, &hdr_idx, bfh.biClrImportant, sizeof(uint32));

   // write result bmp file
   if (!(file=fopen(filename,"wb")))
      {
         cout << "Cannot open file: " << filename << endl;
         return(1);
      }
   fwrite(&hdr, sizeof(unsigned char), 0x36, file);

   // Write color table
   for(int i = 0; i < 256; i++) {
      uint8 x = (uint8)i;
      fwrite(&x, sizeof(uint8), 1, file);
      fwrite(&x, sizeof(uint8), 1, file);
      fwrite(&x, sizeof(uint8), 1, file);
      uint8 z = 0;
      fwrite(&z, sizeof(uint8), 1, file);
   }

   // Write the image data
   for(int i = 0; i < size->height; i++) {
      for(int j = 0; j < size->width; j++) {
         fwrite(&((*outputImage)[size->height-1-i][j]), sizeof(uint8), 1, file);
      }
   }

   fclose(file);
   return 0;
}

int shows = 0;

void show() {
   system("start /tmp/bmplib.bmp");
   // wait longer on the first show, OS takes time to start
   if (shows == 0) Sleep(1000);

   // generally, wait 0.2 seconds = 200 milliseconds
   Sleep(200);
   shows++;
}

void showRGBBMP(unsigned char**** inputImage, ImageSize* size) {
   writeRGBBMP("/tmp/bmplib.bmp", inputImage, size);
   show();
}

void showGSBMP(unsigned char*** inputImage, ImageSize* size) {
   writeGSBMP("/tmp/bmplib.bmp", inputImage, size);
   show();
}
