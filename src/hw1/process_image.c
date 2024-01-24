#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    x = fmin(x, im.w - 1);
    x = fmax(0, x);
    y = fmin(y, im.h - 1);
    y = fmax(0, y);

    int pos = y * im.w + x + im.w * im.h * c;
    float pixel = im.data[pos];
    return pixel;
}

void set_pixel(image im, int x, int y, int c, float v)
{
    if(x < 0 || x >= im.w || y < 0 || y >= im.h || c < 0 || c >= im.c) {
        return;
    } else {
        int pos = x + y * im.w + im.w * im.h * c;
        im.data[pos] = v; 
    }
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    int numOfPixels = im.w * im.h * im.c;
    memcpy(copy.data, im.data, 4 * numOfPixels);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    for(int i = 0; i < im.w; i ++){
        for(int j = 0; j < im.h; j ++) {
            float red = get_pixel(im, i, j, 0);
            float green = get_pixel(im, i, j, 1);
            float blue = get_pixel(im, i, j, 2);
            float value = 0.299 * red + 0.587 * green + 0.114 * blue;
            set_pixel(gray, i, j, 0, value);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    for(int i = 0; i < im.w; i ++) {
        for (int j = 0; j < im.h; j ++) {
            float currentPixel = get_pixel(im, i, j, c);
            set_pixel(im, i, j, c,  currentPixel + v);
        }
    }
}

void clamp_image(image im)
{
    for(int i = 0; i < im.w; i ++) {
        for(int j = 0; j < im.h; j ++) {
            for(int k = 0; k < im.c; k ++) {
                float currentPixel = get_pixel(im, i, j, k);
                if( currentPixel > 1) {
                    set_pixel(im, i, j, k, 1);
                }
                if(currentPixel < 0) {
                    set_pixel(im, i, j, k, 0);
                }
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    for(int i = 0; i < im.w; i ++) {
        for(int j = 0; j < im.h; j ++) {
            float red = get_pixel(im, i, j, 0);
            float green = get_pixel(im, i, j, 1);
            float blue = get_pixel(im, i, j, 2);

            // calculate value
            float value = three_way_max(red, green, blue);

            // calculate saturation
            float min = three_way_min(red, green, blue);
            float range = value - min;
            
            float saturation = 0;
            if(value != 0)  {
                saturation = range / value;
            }
  
            //Calculate hue
            float H0 = 0;
            if(range != 0) {
                if(value == red) H0 = (green - blue) / range;
                if(value == green) H0 = (blue - red) / range + 2;
                if(value == blue) H0 = (red - green) / range + 4;
            }
            float hue;
            if(H0 < 0) {
                hue = H0 / 6 + 1;
            }
            else {
                hue = H0 / 6.0;
            }

            set_pixel(im, i, j, 0, hue);
            set_pixel(im, i, j, 1, saturation);
            set_pixel(im, i, j, 2, value);
        }
    }
}

void hsv_to_rgb(image im)
{
    for(int i = 0; i < im.w; i ++) {
        for(int j = 0; j < im.h; j ++) {
            float hue = 6 * get_pixel(im, i, j, 0);
            float saturation = get_pixel(im, i, j, 1);
            float value = get_pixel(im, i, j, 2);

            int Hi = floor(hue);

            float F = hue - Hi;
            float P = value * (1 - saturation);
            float Q = value * (1 - F * saturation);
            float T = value * (1 - (1-F) * saturation);


            float red = 0;
            float green = 0; 
            float blue = 0;

            if (Hi == 0){
                red = value;
                green = T;
                blue = P;
            } else if (Hi == 1) {
                red = Q;
                green = value;
                blue = P;
            } else if (Hi == 2) {
                red = P;
                green = value;
                blue = T;
            } else if (Hi == 3) {
                red = P;
                green = Q;
                blue = value;
            } else if (Hi == 4) {
                red = T;
                green = P;
                blue = value;
            } else if (Hi == 5) {
                red = value;
                green = P;
                blue = Q;
            }

            set_pixel(im, i, j, 0, red);
            set_pixel(im, i, j, 1, green );
            set_pixel(im, i, j, 2, blue);
        }
    }
}

void scale_image(image im, int c, float v) {

    if(c <0 || c >= im.c) {
        return;
    }
    for(int i = 0; i < im.w; i ++) {
        for(int j = 0; j < im.h; j ++) {
            float newPixel = v * get_pixel(im, i, j, c);
            set_pixel(im, i, j, c, newPixel);
        }
    }
}
