#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include "omp.h"

using namespace std;
using namespace cv;

int main() {
	Mat src = imread("test.jpg");
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	Mat dst;
	int dst_gx, dst_gy;
	dst = src_gray.clone();

	int gx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	int gy[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };


	int m, n;
	m = src_gray.rows;
	n = src_gray.cols;
	printf("m, n: %d, %d\n", m, n);

	// Test basic calculation
	clock_t start1 = clock();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i > 1 && i < m - 1 && j > 1 && j < n - 1) {
				dst_gx = src_gray.at<uchar>(i - 1, j - 1)* gx[0] + src_gray.at<uchar>(i - 1, j) * gx[1] + src_gray.at<uchar>(i - 1, j + 1) * gx[2] + \
					src_gray.at<uchar>(i, j - 1) * gx[3] + src_gray.at<uchar>(i, j) * gx[4] + src_gray.at<uchar>(i, j + 1) * gx[5] + \
					src_gray.at<uchar>(i + 1, j - 1) * gx[6] + src_gray.at<uchar>(i + 1, j) * gx[7] + src_gray.at<uchar>(i + 1, j + 1) * gx[8];
				dst_gy = src_gray.at<uchar>(i - 1, j - 1) * gy[0] + src_gray.at<uchar>(i - 1, j) * gy[1] + src_gray.at<uchar>(i - 1, j + 1) * gy[2] + \
					src_gray.at<uchar>(i, j - 1) * gy[3] + src_gray.at<uchar>(i, j) * gy[4] + src_gray.at<uchar>(i, j + 1) * gy[5] + \
					src_gray.at<uchar>(i + 1, j - 1) * gy[6] + src_gray.at<uchar>(i + 1, j) * gy[7] + src_gray.at<uchar>(i + 1, j + 1) * gy[8];
				dst.at<uchar>(i, j) = round(sqrt(dst_gx * dst_gx + dst_gy * dst_gy));
				if (dst.at<uchar>(i, j) > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				if (dst.at<uchar>(i, j) < 0) {
					dst.at<uchar>(i, j) = 0;
				}
			}
			else {
				dst.at<uchar>(i, j) = src_gray.at<uchar>(i, j);
			}
		}
	}
	clock_t stop1 = clock();


	//Using OpenMP
	clock_t start2 = clock();
	#pragma omp parallel for private(dst_gx, dst_gy)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i > 1 && i < m - 1 && j > 1 && j < n - 1) {
				dst_gx = src_gray.at<uchar>(i - 1, j - 1) * gx[0] + src_gray.at<uchar>(i - 1, j) * gx[1] + src_gray.at<uchar>(i - 1, j + 1) * gx[2] + \
					src_gray.at<uchar>(i, j - 1) * gx[3] + src_gray.at<uchar>(i, j) * gx[4] + src_gray.at<uchar>(i, j + 1) * gx[5] + \
					src_gray.at<uchar>(i + 1, j - 1) * gx[6] + src_gray.at<uchar>(i + 1, j) * gx[7] + src_gray.at<uchar>(i + 1, j + 1) * gx[8];
				dst_gy = src_gray.at<uchar>(i - 1, j - 1) * gy[0] + src_gray.at<uchar>(i - 1, j) * gy[1] + src_gray.at<uchar>(i - 1, j + 1) * gy[2] + \
					src_gray.at<uchar>(i, j - 1) * gy[3] + src_gray.at<uchar>(i, j) * gy[4] + src_gray.at<uchar>(i, j + 1) * gy[5] + \
					src_gray.at<uchar>(i + 1, j - 1) * gy[6] + src_gray.at<uchar>(i + 1, j) * gy[7] + src_gray.at<uchar>(i + 1, j + 1) * gy[8];
				dst.at<uchar>(i, j) = round(sqrt(dst_gx * dst_gx + dst_gy * dst_gy));
				if (dst.at<uchar>(i, j) > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				if (dst.at<uchar>(i, j) < 0) {
					dst.at<uchar>(i, j) = 0;
				}
			}
			else {
				dst.at<uchar>(i, j) = src_gray.at<uchar>(i, j);
			}
		}
	}
	clock_t stop2 = clock();

	double div = double(stop1 - start1) / double(stop2 - start2);

	printf("Basic Calc: %d\n", stop1 - start1);
	printf("Using OpenMP: %d\n", stop2 - start2);
	printf("Div: %f\n", div);


	imshow("dst", dst);

	//imwrite("./test_dst.jpg", dst);
	waitKey(0);

	return 0;
	



}