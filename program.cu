/*
 * COSC330 Assignment 04 - By Vladimir Ovechkin
 * This program generates a mandelbrot using CUDA.
 *
 * To compile this program, use the "make program" command from the Terminal.
 * To run this program, enter "make run".
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bmpfile.h"

#define WIDTH 1920
#define HEIGHT 1080
#define MAX_ITER 1000

#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

/*
 * Performs colour mixing on the value of x.
 * color -> The RGB colour values.
 * x-> The input value for determining the position on the gradient.
 * min -> The minimum value any colour can take during mixing.
 * max -> The maximum value any colour can take during mixing.
 */
void GroundColorMix(double* color, double x, double min, double max) 
{
    // Calculate positive and negative slopes for colour component interpolation
    double posSlope = (max - min) / 60;
    double negSlope = (min - max) / 60;

    // Determine the appropriate colour mixing on the value of x
    if(x < 60)
    {
        color[0] = max;
        color[1] = posSlope * x + min;
        color[2] = min;
        return;
    }

    else if (x < 120)
    {
        color[0] = negSlope * x + 2.0 * max + min;
        color[1] = max;
        color[2] = min;
        return;
    }

    else if (x < 180)
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope * x - 2.0 * max + min;
        return;
    }

    else if (x < 240)
    {
        color[0] = min;
        color[1] = negSlope * x + 4.0 * max + min;
        color[2] = max;
        return;
    }

    else if (x < 300)
    {
        color[0] = posSlope * x - 4.0 * max + min;
        color[1] = min;
        color[2] = max;
        return;
    }

    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope * x + 6 * max;
        return;
    }
}

/*
 * Produce a mandelbrot.
 * d_result ->  Pointer to the memory where the mandelbrot will be stored.
 * width -> Output BMP width.
 * height -> Output BMP height.
 * xCenter -> The x-coordinate for where the mandelbrot is generated.
 * yCenter -> The y-coordinate for where the mandelbrot is generated.
 * resolution -> The level of detail in the mandelbrot.
 */
__global__ void mandelbrotKernel(float *d_result, int width, int height, float xCenter, float yCenter, float resolution) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) 
    {
        // Calculate coordinates for this pixel
        float x = xCenter + (col - width / 2.0f) / resolution;
        float y = yCenter + (height / 2.0f - row) / resolution;

        // Mandelbrot calculations
        float a = 0.0f, b = 0.0f;
        int iter = 0;
        while (iter < MAX_ITER && a * a + b * b <= 4.0f) 
        {
            float aTemp = a * a - b * b + x;
            b = 2.0f * a * b + y;
            a = aTemp;
            ++iter;
        }

        // Map iter to color and store in d_result
        d_result[row * width + col] = (float)iter;
    }
}

int main() 
{
    // Allocate host memory for the result
    float *h_result = (float *)malloc(WIDTH * HEIGHT * sizeof(float));

    // Allocate device memory for the result
    float *d_result;
    cudaMalloc((void **)&d_result, WIDTH * HEIGHT * sizeof(float));

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    mandelbrotKernel<<<numBlocks, threadsPerBlock>>>(d_result, WIDTH, HEIGHT, -0.55f, 0.6f, 8700.0f);

    // Copy the result from device to host
    cudaMemcpy(h_result, d_result, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Generate and save the bitmap image
    bmpfile_t *bmp = bmp_create(WIDTH, HEIGHT, 32);
    for (int col = 0; col < WIDTH; col++) 
    {
        for (int row = 0; row < HEIGHT; row++) 
        {
            float normalizedValue = h_result[row * WIDTH + col] / MAX_ITER;
            double color[3];
            GroundColorMix(color, COLOUR_MAX - normalizedValue * GRADIENT_COLOUR_MAX, 1, COLOUR_DEPTH);
            rgb_pixel_t pixel = {(uint8_t)color[0], (uint8_t)color[1], (uint8_t)color[2], 0};
            bmp_set_pixel(bmp, col, row, pixel);
        }
    }
    bmp_save(bmp, "mandelbrot.bmp");
    bmp_destroy(bmp);

    // Free memory
    free(h_result);
    cudaFree(d_result);

    return 0;
}