#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <opencv.hpp>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#define max2(a,b) a>b?a:b
#define min2(a,b) a<b?a:b
#define max3(a,b,c) (a>b?a:b)>c?(a>b?a:b):c
#define min4(a,b,c,d) (a<b?a:b)<(c<d?c:d)?(a<b?a:b):(c<d?c:d)

__device__ void compare(float *point, bool &result) {
	//多传一个数占最后一位
	float max = max3(point[0], point[1], point[2]);
	float min = min4(point[3], point[4], point[5], point[6]);
	result = (max < min);
}
__device__ void condition(float* img_deriv, int choice, int index, bool &result, int *point,int* h_gpu,int* w_gpu) {
	int idx = index%(*w_gpu);
	int idy = index / (*w_gpu);
	int xl = max2(-idx, -1);
	int xr = min2((*w_gpu) - 1 - idx, 1);
	int yu = max2(-idy, -1);
	int yd = min2((*h_gpu) - 1 - idy, 1);
	float surround[3][3] = { { img_deriv[idx + xl + (idy + yu)*(*w_gpu)], img_deriv[idx + (idy + yu)*(*w_gpu)], img_deriv[idx + xr + (idy + yu)*(*w_gpu)] },
						{ img_deriv[idx + xl + idy*(*w_gpu)], img_deriv[idx + idy*(*w_gpu)], img_deriv[idx + xr + idy*(*w_gpu)] } ,
						{ img_deriv[idx + xl + (idy + yd)*(*w_gpu)], img_deriv[idx + (idy + yd)*(*w_gpu)], img_deriv[idx + xr + (idy + yd)*(*w_gpu)] } };
	if (choice == 0) {
		float input[] = { surround[0][0], surround[0][1], surround[0][2], surround[1][1], surround[2][0], surround[2][1], surround[2][2] };
		point[0] = idx + xl;
		point[1] = idy + yu;
		point[2] = idx;
		point[3] = idy + yu;
		point[4] = idx + xr;
		point[5] = idy + yu;
		compare(input, result);
	}
	else if (choice == 1) {
		float input[] = { surround[0][1], surround[0][2], surround[1][2], surround[1][0], surround[1][1], surround[2][1],255 };
		point[0] = idx;
		point[1] = idy + yu;
		point[2] = idx + xr;
		point[3] = idy + yu;
		point[4] = idx + xr;
		point[5] = idy;
		compare(input, result);
	}
	else if (choice == 2) {
		float input[] = { surround[0][2], surround[1][2], surround[2][2], surround[0][0], surround[1][0], surround[1][1], surround[2][0] };
		point[0] = idx + xr;
		point[1] = idy + yu;
		point[2] = idx + xr;
		point[3] = idy;
		point[4] = idx + xr;
		point[5] = idy + yd;
		compare(input, result);
	}
	else if (choice == 3) {
		float input[] = { surround[1][2], surround[2][2], surround[2][1], surround[0][1], surround[1][0], surround[1][1],255 };
		point[0] = idx + xr;
		point[1] = idy;
		point[2] = idx + xr;
		point[3] = idy + yd;
		point[4] = idx;
		point[5] = idy + yd;
		compare(input, result);
	}
	else if (choice == 4) {
		float input[] = { surround[2][0], surround[2][1], surround[2][2], surround[0][0], surround[0][1], surround[0][2], surround[1][1] };
		point[0] = idx + xr;
		point[1] = idy + yd;
		point[2] = idx;
		point[3] = idy + yd;
		point[4] = idx + xl;
		point[5] = idy + yd;
		compare(input, result);
	}
	else if (choice == 5) {
		float input[] = { surround[1][0], surround[2][0], surround[2][1], surround[0][1], surround[1][1], surround[1][2],255 };
		point[0] = idx + xl;
		point[1] = idy + yu;
		point[2] = idx;
		point[3] = idy + yu;
		point[4] = idx + xr;
		point[5] = idy + yu;
		compare(input, result);
	}
	else if (choice == 6) {
		float input[] = { surround[0][0], surround[1][0], surround[2][0], surround[0][2], surround[1][1], surround[1][2], surround[2][2] };
		point[0] = idx + xl;
		point[1] = idy + yd;
		point[2] = idx + xl;
		point[3] = idy;
		point[4] = idx + xl;
		point[5] = idy + yu;
		compare(input, result);
	}
	else if (choice == 7) {
		float input[] = { surround[0][0], surround[0][1], surround[1][0], surround[1][1], surround[1][2], surround[2][1] ,255 };
		point[0] = idx + xl;
		point[1] = idy;
		point[2] = idx + xl;
		point[3] = idy + yu;
		point[4] = idx;
		point[5] = idy + yu;
		compare(input, result);
	}
	else {
		result = FALSE;
	}
}
__global__ void deblur(float *src, float *src_deriv, float *dst, float *dst_deriv,int *nbytes_gpu,int* h_gpu,int* w_gpu) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < (*nbytes_gpu/sizeof(float))) {
		float alpha = 175. / 255;
		bool result = false;
		int point[] = { 0,0,0,0,0,0 };
		for (int i = 0; i < 8; ++i) {
			condition(src_deriv, i, index, result, point,h_gpu,w_gpu);
			if (result == false) {
				float new_color_b = 0;
				float new_color_g = 0;
				float new_color_r = 0;
				float new_luminance = 0;
				float mean = 0;
				mean = (src[3 * (point[0] + point[1] * (*w_gpu))] + src[3 * (point[0] + point[1] * (*w_gpu)) + 1] + src[3 * (point[0] + point[1] * (*w_gpu)) + 2] + \
					src[3 * (point[2] + point[3] * (*w_gpu))] + src[3 * (point[2] + point[3] * (*w_gpu)) + 1] + src[3 * (point[2] + point[3] * (*w_gpu)) + 2] + \
					src[3 * (point[4] + point[5] * (*w_gpu))] + src[3 * (point[4] + point[5] * (*w_gpu)) + 1] + src[3 * (point[4] + point[5] * (*w_gpu)) + 2] \
					) / 9;
				new_color_b = alpha*src[3 * index] + (1 - alpha)*mean;
				new_color_g = alpha*src[3 * index + 1] + (1 - alpha)*mean;
				new_color_r = alpha*src[3 * index + 2] + (1 - alpha)*mean;
				new_luminance = 1. / 6 * new_color_b + 1. / 2 * new_color_g + 1. / 3 * new_color_r;
				if (dst_deriv[index] < new_luminance) {
					dst_deriv[index] = new_luminance;
					dst[3 * index] = new_color_b;
					dst[3 * index + 1] = new_color_g;
					dst[3 * index + 2] = new_color_r;
				}
			}
		}
	}
}
__global__ void calculate_grad(float *src_deriv, float *dst_deriv, int *nbytes_gpu,int* h_gpu,int* w_gpu) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < (*nbytes_gpu / sizeof(float))) {
		int idx = index%(*w_gpu);
		int idy = index / (*w_gpu);
		int xl = max2(-idx, -1);
		int xr = min2((*w_gpu) - 1 - idx, 1);
		int yu = max2(-idy, -1);
		int yd = min2((*h_gpu) - 1 - idy, 1);
		dst_deriv[index] = -src_deriv[idx + xl + (idy + yu)*(*w_gpu)] - src_deriv[idx + (idy + yu)*(*w_gpu)] - src_deriv[idx + xr + (idy + yu)*(*w_gpu)] \
			- src_deriv[idx + xl + idy*(*w_gpu)] + 8 * src_deriv[index] - src_deriv[idx + xr + idy*(*w_gpu)] \
			- src_deriv[idx + xl + (idy + yd)*(*w_gpu)] - src_deriv[idx + (idy + yd)*(*w_gpu)] - src_deriv[idx + xr + (idy + yd)*(*w_gpu)];
	}
}
__global__ void grad_refine(float *src, float *src_deriv, float *dst, float *dst_deriv,int *nbytes_gpu,int* h_gpu,int* w_gpu) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < (*nbytes_gpu / sizeof(float))) {
		float alpha = 230./255;
		bool result = false;
		int point[] = { 0,0,0,0,0,0 };
		dst_deriv[index] = src_deriv[index];
		dst[3 * index] = src[3 * index];
		dst[3 * index + 1] = src[3 * index + 1];
		dst[3 * index + 2] = src[3 * index + 2];
		bool execute = false;
		for (int i = 0; i < 8; ++i) {
			condition(src_deriv, i, index, result, point,h_gpu,w_gpu);
			if (result == true) {
				float new_color_b = 0;
				float new_color_g = 0;
				float new_color_r = 0;
				float deriv = 0;
				float mean = 0;
				mean = (src[3 * (point[0] + point[1] * (*w_gpu))] + src[3 * (point[0] + point[1] * (*w_gpu)) + 1] + src[3 * (point[0] + point[1] * (*w_gpu)) + 2] + \
					src[3 * (point[2] + point[3] * (*w_gpu))] + src[3 * (point[2] + point[3] * (*w_gpu)) + 1] + src[3 * (point[2] + point[3] * (*w_gpu)) + 2] + \
					src[3 * (point[4] + point[5] * (*w_gpu))] + src[3 * (point[4] + point[5] * (*w_gpu)) + 1] + src[3 * (point[4] + point[5] * (*w_gpu)) + 2] \
					) / 9;
				new_color_b = alpha*src[3 * index] + (1 - alpha)*mean;
				new_color_g = alpha*src[3 * index + 1] + (1 - alpha)*mean;
				new_color_r = alpha*src[3 * index + 2] + (1 - alpha)*mean;
				deriv = alpha*src_deriv[index] + (1 - alpha)*(src_deriv[point[0] + point[1] * (*w_gpu)] + src_deriv[point[2] + point[3] * (*w_gpu)] + src_deriv[point[4] + point[5] * (*w_gpu)]) / 3;
				dst_deriv[index] = deriv;
				dst[3 * index] = new_color_b;
				dst[3 * index + 1] = new_color_g;
				dst[3 * index + 2] = new_color_r;
				execute = true;
				break;
			}
		}
		if (!execute) {
			dst_deriv[index] = src_deriv[index];
			dst[3 * index] = src[3 * index];
			dst[3 * index+1] = src[3 * index+1];
			dst[3 * index+2] = src[3 * index+2];
		}
	}
}
/*
__global__ void anime4k(uchar *src, uchar* deriv, uchar *dst, int nbytes) {
	int idx = 0;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < nbytes) {

	}
}
*/
__global__ void img_mean(float *src, float *dst,int *nbytes_gpu) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < *nbytes_gpu/sizeof(float)) {
		dst[index] = (src[3 * index] + src[3 * index + 1] + src[3 * index + 2]) / 3;
	}
}
int main(int argc, char **argv)
{
	std::string img_path = "C:/Users/75909/Desktop/waifu2.jpg";
	std::string save_path = "";
	cv::Mat img = cv::imread(img_path);
	//cv::Mat img;
	assert(img.channels() == 3);
	cv::resize(img, img, cv::Size((int)img.cols * 2, (int)img.rows * 2), 0, 0, cv::INTER_CUBIC);
	//while(1){
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		
		int w = img.cols;
		int h = img.rows;
		int nbytes = w*h * sizeof(float);
		dim3 block = 512;
		dim3 grid = (w*h + block.x - 1) / block.x;
		//uchar* img_data = new uchar[3 * nbytes / sizeof(float)];
		float *img_vector = new float[3 * nbytes / sizeof(float)];
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				cv::Vec3b* in_row = img.ptr<cv::Vec3b>(i);
				img_vector[3 * (i*w + j)] = in_row[j][0];
				img_vector[3 * (i*w + j) + 1] = in_row[j][1];
				img_vector[3 * (i*w + j) + 2] = in_row[j][2];
			}
		}

		/*
		for (int i = 0; i < 3 * nbytes / sizeof(float); ++i) {
			img_vector[i] = img_data[i];
		}
		*/
		//首先打印的是channel*col
		//然后按row打印
		//转的一维数组 就是直接按这个顺序衔接起来
		float *input, *output, *img_deriv, *output_deriv;
		int *w_gpu, *h_gpu, *nbytes_gpu;
		cudaMalloc((float**)&input, 3 * nbytes);
		cudaMalloc((float**)&output, 3 * nbytes);
		cudaMalloc((float**)&img_deriv, nbytes);
		cudaMalloc((float**)&output_deriv, nbytes);
		cudaMalloc((int**)&w_gpu, sizeof(int));
		cudaMalloc((int**)&h_gpu, sizeof(int));
		cudaMalloc((int**)&nbytes_gpu, sizeof(int));
		float *resul = new float[3 * nbytes / sizeof(float)];
		float *result_deriv = new float[nbytes / sizeof(float)];

		cudaMemcpy(input, img_vector, 3 * nbytes, cudaMemcpyHostToDevice);

		cudaMemcpy(w_gpu, &w, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(h_gpu, &h, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(nbytes_gpu, &nbytes, sizeof(int), cudaMemcpyHostToDevice);

		img_mean << <grid, block >> > (input, img_deriv, nbytes_gpu);

		deblur << <grid, block >> > (input, img_deriv, output, output_deriv, nbytes_gpu, h_gpu, w_gpu);

		calculate_grad << <grid, block >> > (output_deriv, img_deriv, nbytes_gpu, h_gpu, w_gpu);
		/*
		cudaMemcpy(result_deriv, img_deriv, nbytes, cudaMemcpyDeviceToHost);
		cv::Mat grad_pic = cv::Mat::zeros(img.size(), CV_8UC1);
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				uchar* out_row = grad_pic.ptr<uchar>(i);
				out_row[j] = (int)result_deriv[(i*w + j)];
			}
		}
		cv::imwrite("grad.bmp", grad_pic);
		*/
		grad_refine << <grid, block >> > (output, img_deriv, input, output_deriv, nbytes_gpu, h_gpu, w_gpu);

		cudaMemcpy(resul, input, 3*nbytes, cudaMemcpyDeviceToHost);
		//cudaMemcpy(result_deriv, output_deriv, nbytes, cudaMemcpyDeviceToHost);
		/*
		float *test = new float[3 * nbytes / sizeof(float)];
		cudaMemcpy(test, output, 3 * nbytes, cudaMemcpyDeviceToHost);
		cv::Mat test_pic = cv::Mat::zeros(img.size(), CV_8UC3);
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				cv::Vec3b* out_row = test_pic.ptr<cv::Vec3b>(i);
				out_row[j][0] = (int)test[3 * (i*w + j)];
				out_row[j][1] = (int)test[3 * (i*w + j) + 1];
				out_row[j][2] = (int)test[3 * (i*w + j) + 2];
			}
		}
		cv::imwrite("test.bmp", test_pic);
		*/
		cv::Mat out_pic = cv::Mat::zeros(img.size(), img.type());
		for (int i = 0; i < h; ++i) {
			for (int j = 0; j < w; ++j) {
				cv::Vec3b* out_row = out_pic.ptr<cv::Vec3b>(i);
				out_row[j][0] = (int)resul[3 * (i*w + j)];
				out_row[j][1] = (int)resul[3 * (i*w + j) + 1];
				out_row[j][2] = (int)resul[3 * (i*w + j) + 2];
			}
		}
		
		cudaFree(input);
		cudaFree(output);
		cudaFree(img_deriv);
		cudaFree(output_deriv);
		cudaFree(w_gpu);
		cudaFree(h_gpu);
		cudaFree(nbytes_gpu);
		
		cv::imwrite("result.bmp", out_pic);
		std::chrono::duration<double, std::milli> time_span = std::chrono::high_resolution_clock::now() - start;
		std::cout << "time:" << time_span.count() << std::endl;
		//system("pause");
	//}
	return 0;
}
