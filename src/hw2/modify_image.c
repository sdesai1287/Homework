#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
    int newX = (int) roundf(x);
    int newY = (int) roundf(y);
    return get_pixel(im, newX, newY, c);
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix the return line)
    /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
    image newImage = make_image(w,h,im.c);
    float offset = 0.5;

    float newW = ((float) im.w) / w;
    float newH = ((float) im.h) / h;
    for (int y = 0; y < newImage.h; y++) {
        for (int x = 0; x < newImage.w; x++) {
            for (int c = 0; c < newImage.c; c++) {
                float newPixel = nn_interpolate(im, (((x + offset) * newW) - offset), (((y + offset) * newH) - offset), c);
                set_pixel(newImage, x, y, c, newPixel);
            }
        }
    }
    return newImage;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/
    // get vertices
    float v1 = get_pixel(im, floor(x), floor(y), c);
    float v2 = get_pixel(im, ceil(x), floor(y), c);
    float v3 = get_pixel(im, floor(x), ceil(y), c);
    float v4 = get_pixel(im, ceil(x), ceil(y), c);

    // get distances
    float d1 = x - floor(x);
    float d2 = ceil(x) - x;
    float d3 = y - floor(y);
    float d4 = ceil(y) - y;

    // get qs
    float q1 = v1 * d2 + v2 * d1;
    float q2 = v3 * d2 + v4 * d1;
    float q = q1 * d4 + q2 * d3;

    return q;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
    ************************************************************************/
    image newImage = make_image(w,h,im.c);
    float offset = 0.5;

    float newW = ((float) im.w) / w;
    float newH = ((float) im.h) / h;
    for (int y = 0; y < newImage.h; y++) {
        for (int x = 0; x < newImage.w; x++) {
            for (int c = 0; c < newImage.c; c++) {
                float newPixel = bilinear_interpolate(im, (((x + offset) * newW) - offset), (((y + offset) * newH) - offset), c);
                set_pixel(newImage, x, y, c, newPixel);
            }
        }
    }
    return newImage;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    // TODO
    /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/
    float sum = 0.0;

    // calculate sum
    for (int x = 0; x < im.w; x++) {
        for (int y = 0; y < im.h; y++) {
            for (int c = 0; c < im.c; c++) {
                sum += (get_pixel(im, x, y, c));
            }
        }
    }

    // avoid divide by zero
    if (sum == 0) {
        sum += 0.000001;
    }

    // divide
    float pixel = 0.0;
    for (int x = 0; x < im.w; x++) {
        for (int y = 0; y < im.h; y++) {
            for (int c = 0; c < im.c; c++) {
                pixel = get_pixel(im, x, y, c);
                set_pixel(im, x, y, c, pixel / sum);
            }
        }
    }
}

image make_box_filter(int w)
{
    // TODO
    /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
    image filter = make_image(w,w,1);
    for (int x = 0; x < filter.w; x++) {
        for (int y = 0; y < filter.h; y++) {
            set_pixel(filter, x, y, 0, 1);
        }
    }
    l1_normalize(filter);
    return filter;
}

// helper function to convolve, calculates filtered values
static float get_filtered_value(image img, image filter, int x, int y, int c, int cf) {
    float sum = 0;
    int filtered_xW = (filter.w-1)/2; 
    int filtered_yW = (filter.h-1)/2;
    for (int xf = 0; xf < filter.w; xf++) {
        for (int yf = 0; yf < filter.h; yf++) {
            float newX = x - filtered_xW + xf;
            float newY = y - filtered_yW + yf;
            float pixel1 = get_pixel(img, newX, newY, c);
            float pixel2 = get_pixel(filter, xf, yf, cf);
            sum += (pixel1 * pixel2);
        }
    }
    return sum;
    
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
    image convolve = copy_image(im);

    for (int x = 0; x < im.w; x++) {
        for (int y = 0; y < im.h; y++) {
            for (int c = 0; c < im.c; c++) {
                int filter_c = c;
                if(filter.c == 1) {
                    filter_c = 0;
                }
                float newValue = get_filtered_value(im, filter, x, y, c, filter_c);
                set_pixel(convolve, x, y, c, newValue);
            }
        }
    }
    
    // sum the channels
    if (im.c == 1 || preserve != 1) {
        image newImage = make_image(convolve.w, convolve.h, 1); 
        for (int x = 0; x < im.w; x++) {
            for (int y = 0; y < im.h; y++) {
                float sum = 0;
                for(int c = 0; c < im.c; c++) {
                    sum += get_pixel(convolve, x, y ,c);
                }
                set_pixel(newImage, x, y, 0, sum);
            }
        }
        return newImage;
    }
    return convolve;
}

image make_highpass_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float matrix[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};

    int length = sizeof(matrix) / sizeof(matrix[0]);
    for (int i = 0; i < length; i++) {
        filter.data[i] = matrix[i];
    }
    return filter;
}

image make_sharpen_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float matrix[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    int length = sizeof(matrix) / sizeof(matrix[0]);
    for (int i = 0; i < length; i++) {
        filter.data[i] = matrix[i];
    }

    return filter;
}

image make_emboss_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float matrix[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

    int length = sizeof(matrix) / sizeof(matrix[0]);
    for (int i = 0; i < length; i++) {
        filter.data[i] = matrix[i];
    }

    return filter;
}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: We would want to use preserve for highpass and emboss, because we need to preserve the colors that these have. 
// We do not need to preserve highpass since it does not have colors to preserve

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: We need to postprocess all filters by clamping the pixel values within (0, 255) to keep them in bounds

image make_gaussian_filter(float sigma)
{
    // TODO
    /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/
    int w = 6 * sigma;

    if (w % 2 == 0) {
        w = w + 1;
    }

    image kernel = make_image(w, w, 1);

    int offset = (int)(w / 2);
    float G = 0;

    for (int x = -offset; x < w - offset; x++) {
        for (int y = -offset; y < w - offset; y++) {
            // use the pdf from the assignment readme
            G = (1.0 / (TWOPI * sigma * sigma)) * (expf(-(x * x + y * y) / (2 * sigma * sigma)));
            set_pixel(kernel, x + offset, y + offset, 0, G);
        }
    }

    l1_normalize(kernel);

    return kernel;
}

image add_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w);
    assert(a.h == b.h);
    assert(a.c == b.c);

    image newImage = make_image(a.w, a.h, a.c);

    for (int x = 0; x < a.w; x++) {
        for (int y = 0; y < a.h; y++) {
            for (int c = 0; c < a.c; c++) {
                float newValue = get_pixel(a, x, y, c) + get_pixel(b, x, y, c);
                set_pixel(newImage, x, y, c, newValue);
            }
        }
    }

    return newImage;
}

image sub_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    assert(a.w == b.w);
    assert(a.h == b.h);
    assert(a.c == b.c);

    image newImage = make_image(a.w, a.h, a.c);

    for (int x = 0; x < a.w; x++) {
        for (int y = 0; y < a.h; y++) {
            for (int c = 0; c < a.c; c++) {
                float newValue = get_pixel(a, x, y, c) - get_pixel(b, x, y, c);
                set_pixel(newImage, x, y, c, newValue);
            }
        }
    }

    return newImage;
}

image make_gx_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float matrix[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    int length = sizeof(matrix) / sizeof(matrix[0]);
    for (int i = 0; i < length; i++) {
        filter.data[i] = matrix[i];
    }

    return filter;
}

image make_gy_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
    image filter = make_image(3, 3, 1);
    float matrix[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    int length = sizeof(matrix) / sizeof(matrix[0]);
    for (int i = 0; i < length; i++) {
        filter.data[i] = matrix[i];
    }

    return filter;
}

void feature_normalize(image im)
{
    // TODO
    /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
    float min = im.data[0];
    float max = min;

    for (int i = 0; i < im.c * im.w * im.h; i++) {
        if (min > im.data[i]) {
            min = im.data[i];
        }
        if (max < im.data[i]) {
            max = im.data[i];
        }
    }

    float range = max - min;

    if (range == 0) {
        for (int i = 0; i < im.c * im.w * im.h; i++) {
            im.data[i] = 0;
        }
    }
    else {
        for (int i = 0; i < im.c * im.w * im.h; i++)  {
            im.data[i] = (im.data[i] - min) / range;
        }
    }


}

image *sobel_image(image im)
{
    // TODO
    /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
    image *sobelimg = calloc(2, sizeof(image));

    image gradient_magnitude = make_image(im.w, im.h, 1);
    image gradient_direction = make_image(im.w, im.h, 1);
    sobelimg[0] = gradient_magnitude;
    sobelimg[1] = gradient_direction;

    image gx_filter = make_gx_filter();
    image gy_filter = make_gy_filter();
    image gx = convolve_image(im, gx_filter, 0);
    image gy = convolve_image(im, gy_filter, 0);


    for(int x = 0; x < im.w; x++) {
        for(int y = 0; y < im.h; y++) {
            float gx_pixel = get_pixel(gx, x, y, 0);
            float gy_pixel = get_pixel(gy, x, y, 0);
            float magnitude_pixel = sqrtf((gx_pixel * gx_pixel) + (gy_pixel * gy_pixel));
            float direction_pixel = atan2(gy_pixel, gx_pixel);
            set_pixel(gradient_magnitude, x, y, 0, magnitude_pixel);
            set_pixel(gradient_direction, x, y, 0, direction_pixel);
        }
    }

    return sobelimg;
}

image colorize_sobel(image im)
{
  // TODO
  /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************/
    image* sobel_result = sobel_image(im);
    image magnitude = sobel_result[0];
    image direction = sobel_result[1];

    feature_normalize(magnitude);
    feature_normalize(direction);

    image hsv = make_image(im.w, im.h, 3);

    for (int x = 0; x < im.w; x++) {
        for (int y = 0; y < im.h; y++) {
            set_pixel(hsv, x, y, 0, get_pixel(direction, x, y, 0));
            set_pixel(hsv, x, y, 1, get_pixel(magnitude, x, y, 0));
            set_pixel(hsv, x, y, 2, get_pixel(magnitude, x, y, 0));
        }
    }

    hsv_to_rgb(hsv);

    return hsv;
}

// EXTRA CREDIT: Median filter

/*
image apply_median_filter(image im, int kernel_size)
{
  return make_image(1,1,1);
}
*/

// SUPER EXTRA CREDIT: Bilateral filter

/*
image apply_bilateral_filter(image im, float sigma1, float sigma2)
{
  return make_image(1,1,1);
}
*/