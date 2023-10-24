#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include <limits.h>

using namespace std;
using namespace cv;

float giveEnergyOfPixel(int x, int y, int width, int height, Mat &image){
    Vec3b pixelPreX, pixelPostX, pixelPreY, pixelPostY;
    if(x == 0){
        if(width <= 3){
            pixelPreX = image.at<Vec3b>(y, 1);
            pixelPostX = image.at<Vec3b>(y, 2);
        }
        else {
            pixelPreX = image.at<Vec3b>(y, 1);
            pixelPostX = image.at<Vec3b>(y, 3);
        }

    } 
    else if(x == width - 1) {
        if(width <= 3){
            pixelPreX = image.at<Vec3b>(y, 1);
            pixelPostX = image.at<Vec3b>(y, 0);
        }
        else {
            pixelPreX = image.at<Vec3b>(y, (width - 1) - 1);
            pixelPostX = image.at<Vec3b>(y, (width - 1) - 3);
        }
    }
    else {
        pixelPreX = image.at<Vec3b>(y, x-1);
        pixelPostX = image.at<Vec3b>(y, x+1);
    }

    if(y == 0){
        if(height <= 3){
            pixelPreY = image.at<Vec3b>(1, x);
            pixelPostY = image.at<Vec3b>(2, x);
        }
        else {
            pixelPreY = image.at<Vec3b>(1, x);
            pixelPostY = image.at<Vec3b>(3, x);
        }

    } 
    else if(y == height - 1) {
        if(height <= 3){
            pixelPreY = image.at<Vec3b>(1, x);
            pixelPostY = image.at<Vec3b>(0, x);
        }
        else {
            pixelPreY = image.at<Vec3b>((height - 1) - 1, x);
            pixelPostY = image.at<Vec3b>((height - 1) - 3, x);
        }
    }
    else {
        pixelPreY = image.at<Vec3b>(y-1, x);
        pixelPostY = image.at<Vec3b>(y+1, x);
    }


    int RX = abs(pixelPreX[2] - pixelPostX[2]);
    int GX = abs(pixelPreX[1] - pixelPostX[1]);
    int BX = abs(pixelPreX[0] - pixelPostX[0]);

    int RY = abs(pixelPreY[2] - pixelPostY[2]);
    int GY = abs(pixelPreY[1] - pixelPostY[1]);
    int BY = abs(pixelPreY[0] - pixelPostY[0]);

    int deltaX = RX * RX + GX * GX + BX * BX;
    int deltaY = RY * RY + GY * GY + BY * BY;

    float energy = sqrt(deltaX + deltaY);

    return energy;
}

void evalVerticalCostMatrix(int width, int height, float* energy_matrix[], float* cost_matrix[]){
    for(int x = 0; x < width; x++){
        cost_matrix[0][x] = energy_matrix[0][x];
    }

    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if(x == 0){
                cost_matrix[y][x] = energy_matrix[y][x] + min(cost_matrix[y-1][x], cost_matrix[y-1][x+1]);
                continue;
                
            } else if (x == width - 1) {
                cost_matrix[y][x] = energy_matrix[y][x] + min(cost_matrix[y-1][x-1], cost_matrix[y-1][x]);
                continue;
            } else {
                cost_matrix[y][x] = energy_matrix[y][x] + min(cost_matrix[y-1][x-1], min(cost_matrix[y-1][x], cost_matrix[y-1][x+1]));
            }
        }
    }
}

void evalHorizontalCostMatrix(int width, int height, float* energy_matrix[], float* cost_matrix[]){
    for(int y = 0; y < height; y++){
        cost_matrix[y][0] = energy_matrix[y][0];
    }

    for (int x = 1; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            if(y == 0){
                cost_matrix[y][x] = energy_matrix[y][x] + min(cost_matrix[y][x-1], cost_matrix[y+1][x-1]);
                continue;
                
            } else if (y == height - 1) {
                cost_matrix[y][x] = energy_matrix[y][x] + min(cost_matrix[y-1][x-1], cost_matrix[y][x-1]);
                continue;
            } else {
                cost_matrix[y][x] = energy_matrix[y][x] + min(cost_matrix[y-1][x-1], min(cost_matrix[y][x-1], cost_matrix[y+1][x-1]));
            }
        }
    }
}

void findVerticalSeam(int width, int height, float* energy_matrix[], float* cost_matrix[], int* seamIndexes){
    float min = INT_MAX;
    for(int x = 0; x < width; x++){
        if(min > cost_matrix[height - 1][x]){
            min = cost_matrix[height - 1][x];
            seamIndexes[height - 1] = x;
        }
    }

    int lastIndex = seamIndexes[height - 1];
    for(int y = height - 2; y >= 0; y--){
        if( lastIndex != 0 && cost_matrix[y][lastIndex-1] < cost_matrix[y][lastIndex]){
            seamIndexes[y] = lastIndex-1;
        } else {
            seamIndexes[y] = lastIndex;
        }

        if( lastIndex != width - 1 && cost_matrix[y][lastIndex+1] < cost_matrix[y][seamIndexes[y]]){
            seamIndexes[y] = lastIndex + 1;
        }
        lastIndex = seamIndexes[y];
    }
}

void findHorizontalSeam(int width, int height, float* energy_matrix[], float* cost_matrix[], int* seamIndexes){
    float min = INT_MAX;
    for(int y = 0; y < height; y++){
        if(min > cost_matrix[y][width - 1]){
            min = cost_matrix[y][width - 1];
            seamIndexes[width - 1] = y;
        }
    }

    int lastIndex = seamIndexes[width - 1];
    for(int x = width - 2; x >= 0; x--){
        if( lastIndex != 0 && cost_matrix[lastIndex-1][x] < cost_matrix[lastIndex][x]){
            seamIndexes[x] = lastIndex-1;
        } else {
            seamIndexes[x] = lastIndex;
        }

        if( lastIndex != height - 1 && cost_matrix[lastIndex+1][x] < cost_matrix[seamIndexes[x]][x]){
            seamIndexes[x] = lastIndex + 1;
        }
        lastIndex = seamIndexes[x];
    }


}

void removeVerticalSeam(int &width, int height, float* energy_matrix[], float* cost_matrix[], int* seamIndexes, Mat &image){
    for(int y = 0; y < height; y++){
        for(int x = seamIndexes[y]; x < width - 1; x++){
            swap(image.at<Vec3b>(y, x), image.at<Vec3b>(y, x+1));
            // swap(energy_matrix[y][x], energy_matrix[y][x+1]);
        }
    }
    width--;
    image.cols--;
}

void removeHorizontalSeam(int width, int &height, float* energy_matrix[], float* cost_matrix[], int* seamIndexes, Mat &image){
    for(int x = 0; x < width; x++){
        for(int y = seamIndexes[x]; y < height - 1; y++){
            swap(image.at<Vec3b>(y, x), image.at<Vec3b>(y+1, x));
            // swap(energy_matrix[y][x], energy_matrix[y+1][x]);
        }
    }
    height--;
    image.rows--;
}

void markVerticalSeamRed(int width, int &height, float* energy_matrix[], float* cost_matrix[], int* seamIndexes, Mat &image){
    for(int y = 0; y < height; y++){
        int x = seamIndexes[y]; 
        image.at<Vec3b>(y, x)[0] = 0;
        image.at<Vec3b>(y, x)[1] = 0;
        image.at<Vec3b>(y, x)[2] = 255;
    }
    imshow("Hello", image);
    waitKey(1);
}


void markHorizontalSeamRed(int width, int &height, float* energy_matrix[], float* cost_matrix[], int* seamIndexes, Mat &image){
    for(int x = 0; x < width; x++){
        int y = seamIndexes[x];
        image.at<Vec3b>(y, x)[0] = 0;
        image.at<Vec3b>(y, x)[1] = 0;
        image.at<Vec3b>(y, x)[2] = 255;
    }
    imshow("Hello", image);
    waitKey(1);
}



int main(int argc, char* argv[]) {
    if(argc < 4){
        cerr << "Give a path to image & desired output image width & height in command line argument" << endl;
        return 0;
    }

    if(argc > 4){
        cerr << "There should be only four command line arguments" << endl;
        return 0;   
    }

    //* Code starts from here
    //* Definitions
    int desiredHeight = stoi(argv[2]);
    int desiredWidth = stoi(argv[3]);
    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }
    int width = image.cols;
    int height = image.rows;
    if(desiredWidth >= width){
        cerr << "Desired width is greater or equal to current width of image" << endl;
        return 0;
    }
    if(desiredHeight >= height){
        cerr << "Desired height is greater or equal to current height of image" << endl;
        return 0;
    }
    float** energy_matrix = new float*[height];
    for (int i = 0; i < height; ++i) {
        energy_matrix[i] = new float[width];
    }
    float** vertical_cost_matrix = new float*[height];
    for (int i = 0; i < height; ++i) {
        vertical_cost_matrix[i] = new float[width];
    }
    float** horizontal_cost_matrix = new float*[height];
    for (int i = 0; i < height; ++i) {
        horizontal_cost_matrix[i] = new float[width];
    }
    int *verticalSeamIndexes = new int[height];
    int *horizontalSeamIndexes = new int[width];

    //* Main Execution Code


    while(width != desiredWidth){
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vec3b pixel = image.at<Vec3b>(y, x);
                float E = giveEnergyOfPixel(x, y, width, height, image);
                energy_matrix[y][x] = E;
            }
        }
        evalVerticalCostMatrix(width, height, energy_matrix, vertical_cost_matrix);
        findVerticalSeam(width, height, energy_matrix, vertical_cost_matrix, verticalSeamIndexes);
        markVerticalSeamRed(width, height, energy_matrix, vertical_cost_matrix, verticalSeamIndexes, image);
        removeVerticalSeam(width, height, energy_matrix, vertical_cost_matrix, verticalSeamIndexes, image);
        // cout << height << " " << width << endl;
    }


    while(height != desiredHeight){
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vec3b pixel = image.at<Vec3b>(y, x);
                float E = giveEnergyOfPixel(x, y, width, height, image);
                energy_matrix[y][x] = E;
            }
        }
        evalHorizontalCostMatrix(width, height, energy_matrix, horizontal_cost_matrix);
        findHorizontalSeam(width, height, energy_matrix, horizontal_cost_matrix, horizontalSeamIndexes);
        markHorizontalSeamRed(width, height, energy_matrix, horizontal_cost_matrix, horizontalSeamIndexes, image);
        removeHorizontalSeam(width, height, energy_matrix, horizontal_cost_matrix, horizontalSeamIndexes, image);
        // cout << height << " " << width << endl;
    }

    imwrite("output.jpeg", image);
    imshow("Hello", image);
    waitKey(0);
    return 0;
}
