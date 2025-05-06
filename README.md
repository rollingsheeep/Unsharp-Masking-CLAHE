# Image Filter Program

This program is an image processing tool that applies various filters to BMP images. It supports both RGB and grayscale images of any size.

## Available Filters

### 1. Unsharp Masking Filter
Enhances image details by subtracting a blurred version of the image and adding it back with a specified weight.
- Parameters:
  - N: Kernel size (odd number, e.g., 3, 5, 7)
  - sigma: Standard deviation for Gaussian blur
  - alpha: Sharpening factor (typically 0.5 to 2.0)
- Command: `filter.exe input.bmp unsharp N sigma alpha output.bmp`
- Example: `filter.exe input/sample.bmp unsharp 5 1.0 1.5 output.bmp`

### 2. Unsharp Masking + CLAHE
Applies unsharp masking followed by CLAHE for enhanced detail and local contrast.
- Parameters:
  - N: Kernel size (odd number, e.g., 3, 5, 7)
  - sigma: Standard deviation for Gaussian blur
  - alpha: Sharpening factor (typically 0.5 to 2.0)
  - (CLAHE parameters are fixed: tile_size=8, clip_limit=2.0)
- Command: `filter.exe input.bmp unsharp_clahe N sigma alpha output.bmp`
- Example: `filter.exe input/sample.bmp unsharp_clahe 5 1.0 1.5 output.bmp`

### 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Enhances local contrast in images by applying histogram equalization to small regions while limiting contrast amplification.
- Parameters:
  - tile_size: Size of tiles for local histogram (e.g., 8, 16, 32)
  - clip_limit: Contrast limit (1.0 to 4.0)
  - dummy: Ignored parameter (for compatibility)
- Command: `filter.exe input.bmp clahe tile_size clip_limit dummy output.bmp`
- Example: `filter.exe input/sample.bmp clahe 8 2.0 0 output.bmp`

## Output Files

The program creates output files in the `output` directory:
1. For unsharp masking:
   - The final sharpened image (e.g., `output.bmp`)
   - The intermediate Gaussian blurred image (e.g., `output_blurred.bmp`)
2. For unsharp masking + CLAHE:
   - The intermediate unsharpened image (e.g., `output_unsharp.bmp`)
   - The final enhanced image (e.g., `output_unsharp_clahe.bmp`)
3. For CLAHE:
   - The contrast-enhanced image (e.g., `output.bmp`)

## Notes

- Input and output files must be in BMP format
- The program supports both RGB and grayscale images
- Images can be of any size
- For unsharp masking:
  - N should be an odd number (3, 5, 7, etc.)
  - Higher sigma values in Gaussian blur create more blurring
  - Higher alpha values create stronger sharpening
- For CLAHE:
  - Tile size should be between 4 and 64
  - Higher clip limit values allow more contrast enhancement
  - Smaller tile sizes provide more local contrast enhancement
- The program will create an `output` directory if it doesn't exist

## Error Handling

The program will display error messages if:
- Input file cannot be opened
- Not enough arguments are provided
- Invalid parameters are specified
- Output file cannot be written
- Memory allocation fails
- Image dimensions are invalid or too large

cmake --build .