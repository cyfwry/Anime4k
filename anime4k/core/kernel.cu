#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <chrono>
#define max2(a,b) a>b?a:b
#define min2(a,b) a<b?a:b
#define max3(a,b,c) (a>b?a:b)>c?(a>b?a:b):c
#define min4(a,b,c,d) (a<b?a:b)<(c<d?c:d)?(a<b?a:b):(c<d?c:d)

texture<float, 2> tex_input_b;
texture<float, 2> tex_input_g;
texture<float, 2> tex_input_r;
texture<float, 2> tex_output_b;
texture<float, 2> tex_output_g;
texture<float, 2> tex_output_r;
texture<float, 2> tex_img_deriv;
texture<float, 2> tex_output_deriv;

__device__ void compare(float *point, bool &result) {
	//多传一个数占最后一位
	float max = max3(point[0], point[1], point[2]);
	float min = min4(point[3], point[4], point[5], point[6]);
	result = (max < min);
}
__device__ void condition(bool reverse, int choice, int idx, int idy, bool &result, int *point, int* h_gpu, int* w_gpu) {
	int xl = max2(-idx, -1);
	int xr = min2((*w_gpu) - 1 - idx, 1);
	int yu = max2(-idy, -1);
	int yd = min2((*h_gpu) - 1 - idy, 1);
	float surround[3][3];
	if (reverse) {
		surround[0][0] = tex2D(tex_output_deriv, idx + xl, idy + yu);
		surround[0][1] = tex2D(tex_output_deriv, idx, idy + yu);
		surround[0][2] = tex2D(tex_output_deriv, idx + xr, idy + yu);
		surround[1][0] = tex2D(tex_output_deriv, idx + xl, idy);
		surround[1][1] = tex2D(tex_output_deriv, idx, idy);
		surround[1][2] = tex2D(tex_output_deriv, idx + xr, idy);
		surround[2][0] = tex2D(tex_output_deriv, idx + xl, idy + yd);
		surround[2][1] = tex2D(tex_output_deriv, idx, idy + yd);
		surround[2][2] = tex2D(tex_output_deriv, idx + xr, idy + yd);
	}
	else {
		surround[0][0] = tex2D(tex_img_deriv, idx + xl, idy + yu);
		surround[0][1] = tex2D(tex_img_deriv, idx, idy + yu);
		surround[0][2] = tex2D(tex_img_deriv, idx + xr, idy + yu);
		surround[1][0] = tex2D(tex_img_deriv, idx + xl, idy);
		surround[1][1] = tex2D(tex_img_deriv, idx, idy);
		surround[1][2] = tex2D(tex_img_deriv, idx + xr, idy);
		surround[2][0] = tex2D(tex_img_deriv, idx + xl, idy + yd);
		surround[2][1] = tex2D(tex_img_deriv, idx, idy + yd);
		surround[2][2] = tex2D(tex_img_deriv, idx + xr, idy + yd);
	}
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
		point[1] = idy;
		point[2] = idx + xl;
		point[3] = idy + yd;
		point[4] = idx;
		point[5] = idy + yd;
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
		result = false;
	}
}
__device__ void deblur(float *dst_b, float *dst_g, float *dst_r, float *dst_deriv, int *nbytes_gpu, int* h_gpu, int* w_gpu, bool reverse) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_init = idx;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	while (idx<(*w_gpu) && idy<(*h_gpu)) {
		int index = idy*(*w_gpu) + idx;
		float alpha = 175.0f / 255;
		bool result = false;
		int point[] = { 0,0,0,0,0,0 };
		for (int i = 0; i < 8; ++i) {
			condition(reverse, i, idx, idy, result, point, h_gpu, w_gpu);
			if (result == false) {
				float new_color_b = 0;
				float new_color_g = 0;
				float new_color_r = 0;
				float new_luminance = 0;
				float mean_b = 0;
				float mean_g = 0;
				float mean_r = 0;
				if (reverse) {
					mean_b = (tex2D(tex_output_b, point[0], point[1]) + tex2D(tex_output_b, point[2], point[3]) + tex2D(tex_output_b, point[4], point[5])) / 3;
					mean_g = (tex2D(tex_output_g, point[0], point[1]) + tex2D(tex_output_g, point[2], point[3]) + tex2D(tex_output_g, point[4], point[5])) / 3;
					mean_r = (tex2D(tex_output_r, point[0], point[1]) + tex2D(tex_output_r, point[2], point[3]) + tex2D(tex_output_r, point[4], point[5])) / 3;
					new_color_b = alpha*tex2D(tex_output_b, idx, idy) + (1 - alpha)*mean_b;
					new_color_g = alpha*tex2D(tex_output_g, idx, idy) + (1 - alpha)*mean_g;
					new_color_r = alpha*tex2D(tex_output_r, idx, idy) + (1 - alpha)*mean_r;
				}
				else {
					mean_b = (tex2D(tex_input_b, point[0], point[1]) + tex2D(tex_input_b, point[2], point[3]) + tex2D(tex_input_b, point[4], point[5])) / 3;
					mean_g = (tex2D(tex_input_g, point[0], point[1]) + tex2D(tex_input_g, point[2], point[3]) + tex2D(tex_input_g, point[4], point[5])) / 3;
					mean_r = (tex2D(tex_input_r, point[0], point[1]) + tex2D(tex_input_r, point[2], point[3]) + tex2D(tex_input_r, point[4], point[5])) / 3;
					new_color_b = alpha*tex2D(tex_input_b, idx, idy) + (1 - alpha)*mean_b;
					new_color_g = alpha*tex2D(tex_input_g, idx, idy) + (1 - alpha)*mean_g;
					new_color_r = alpha*tex2D(tex_input_r, idx, idy) + (1 - alpha)*mean_r;
				}

				new_luminance = 1. / 6 * new_color_b + 1. / 2 * new_color_g + 1. / 3 * new_color_r;
				if (dst_deriv[index] < new_luminance) {
					dst_deriv[index] = new_luminance;
					dst_b[index] = new_color_b;
					dst_g[index] = new_color_g;
					dst_r[index] = new_color_r;
				}
			}
		}
		idx += blockDim.x*gridDim.x;
		if (idx >= (*w_gpu)) {
			idx = idx_init;
			idy += blockDim.y*gridDim.y;
		}
	}
}
__device__ void calculate_grad(bool reverse, float *dst_deriv, int *nbytes_gpu, int* h_gpu, int* w_gpu) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_init = idx;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	while (idx<(*w_gpu) && idy<(*h_gpu)) {
		int index = idy*(*w_gpu) + idx;
		int xl = max2(-idx, -1);
		int xr = min2((*w_gpu) - 1 - idx, 1);
		int yu = max2(-idy, -1);
		int yd = min2((*h_gpu) - 1 - idy, 1);
		if (reverse) {
			dst_deriv[index] = abs(-tex2D(tex_output_deriv, idx + xl, idy + yu) - 2 * tex2D(tex_output_deriv, idx, idy + yu) - tex2D(tex_output_deriv, idx + xr, idy + yu) + tex2D(tex_output_deriv, idx + xl, idy + yd) + 2 * tex2D(tex_output_deriv, idx, idy + yd) + tex2D(tex_output_deriv, idx + xr, idy + yd))
				+ abs(-tex2D(tex_output_deriv, idx + xl, idy + yu) - 2 * tex2D(tex_output_deriv, idx + xl, idy) - tex2D(tex_output_deriv, idx + xl, idy + yd) + tex2D(tex_output_deriv, idx + xr, idy + yd) + 2 * tex2D(tex_output_deriv, idx + xr, idy) + tex2D(tex_output_deriv, idx + xr, idy + yu));

		}
		else {
			dst_deriv[index] = abs(-tex2D(tex_img_deriv, idx + xl, idy + yu) - 2 * tex2D(tex_img_deriv, idx, idy + yu) - tex2D(tex_img_deriv, idx + xr, idy + yu) + tex2D(tex_img_deriv, idx + xl, idy + yd) + 2 * tex2D(tex_img_deriv, idx, idy + yd) + tex2D(tex_img_deriv, idx + xr, idy + yd))
				+ abs(-tex2D(tex_img_deriv, idx + xl, idy + yu) - 2 * tex2D(tex_img_deriv, idx + xl, idy) - tex2D(tex_img_deriv, idx + xl, idy + yd) + tex2D(tex_img_deriv, idx + xr, idy + yd) + 2 * tex2D(tex_img_deriv, idx + xr, idy) + tex2D(tex_img_deriv, idx + xr, idy + yu));

		}
		dst_deriv[index] = min2(max2(dst_deriv[index] / 2, 0), 255);
		idx += blockDim.x*gridDim.x;
		if (idx >= (*w_gpu)) {
			idx = idx_init;
			idy += blockDim.y*gridDim.y;
		}
	}
}
__device__ void grad_refine(float *dst_b, float *dst_g, float *dst_r, float *dst_deriv, int *nbytes_gpu, int* h_gpu, int* w_gpu, bool reverse, bool reverse_deriv) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_init = idx;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	while (idx<(*w_gpu) && idy<(*h_gpu)) {
		int index = idy*(*w_gpu) + idx;
		float alpha = 0.0f / 255;
		bool result = false;
		int point[] = { 0,0,0,0,0,0 };
		if (reverse_deriv) {
			dst_deriv[index] = tex2D(tex_output_deriv, idx, idy);
		}
		else {
			dst_deriv[index] = tex2D(tex_img_deriv, idx, idy);
		}
		if (reverse) {
			dst_b[index] = tex2D(tex_output_b, idx, idy);
			dst_g[index] = tex2D(tex_output_g, idx, idy);
			dst_r[index] = tex2D(tex_output_r, idx, idy);
		}
		else {
			dst_b[index] = tex2D(tex_input_b, idx, idy);
			dst_g[index] = tex2D(tex_input_g, idx, idy);
			dst_r[index] = tex2D(tex_input_r, idx, idy);
		}
		for (int i = 0; i < 8; ++i) {
			condition(reverse_deriv, i, idx, idy, result, point, h_gpu, w_gpu);
			if (result == true) {
				float new_color_b = 0;
				float new_color_g = 0;
				float new_color_r = 0;
				float deriv = 0;
				float mean_b = 0;
				float mean_g = 0;
				float mean_r = 0;
				if (reverse) {
					mean_b = (tex2D(tex_output_b, point[0], point[1]) + tex2D(tex_output_b, point[2], point[3]) + tex2D(tex_output_b, point[4], point[5])) / 3;
					mean_g = (tex2D(tex_output_g, point[0], point[1]) + tex2D(tex_output_g, point[2], point[3]) + tex2D(tex_output_g, point[4], point[5])) / 3;
					mean_r = (tex2D(tex_output_r, point[0], point[1]) + tex2D(tex_output_r, point[2], point[3]) + tex2D(tex_output_r, point[4], point[5])) / 3;
					new_color_b = alpha*tex2D(tex_output_b, idx, idy) + (1 - alpha)*mean_b;
					new_color_g = alpha*tex2D(tex_output_g, idx, idy) + (1 - alpha)*mean_g;
					new_color_r = alpha*tex2D(tex_output_r, idx, idy) + (1 - alpha)*mean_r;
				}
				else {
					mean_b = (tex2D(tex_input_b, point[0], point[1]) + tex2D(tex_input_b, point[2], point[3]) + tex2D(tex_input_b, point[4], point[5])) / 3;
					mean_g = (tex2D(tex_input_g, point[0], point[1]) + tex2D(tex_input_g, point[2], point[3]) + tex2D(tex_input_g, point[4], point[5])) / 3;
					mean_r = (tex2D(tex_input_r, point[0], point[1]) + tex2D(tex_input_r, point[2], point[3]) + tex2D(tex_input_r, point[4], point[5])) / 3;
					new_color_b = alpha*tex2D(tex_input_b, idx, idy) + (1 - alpha)*mean_b;
					new_color_g = alpha*tex2D(tex_input_g, idx, idy) + (1 - alpha)*mean_g;
					new_color_r = alpha*tex2D(tex_input_r, idx, idy) + (1 - alpha)*mean_r;
				}
				if (reverse_deriv) {
					deriv = alpha*tex2D(tex_output_deriv, idx, idy) + (1 - alpha)*(tex2D(tex_output_deriv, point[0], point[1]) + tex2D(tex_output_deriv, point[2], point[3]) + tex2D(tex_output_deriv, point[4], point[5])) / 3;
				}
				else {
					deriv = alpha*tex2D(tex_img_deriv, idx, idy) + (1 - alpha)*(tex2D(tex_img_deriv, point[0], point[1]) + tex2D(tex_img_deriv, point[2], point[3]) + tex2D(tex_img_deriv, point[4], point[5])) / 3;
				}
				dst_deriv[index] = deriv;
				dst_b[index] = new_color_b;
				dst_g[index] = new_color_g;
				dst_r[index] = new_color_r;
				break;
			}
		}
		idx += blockDim.x*gridDim.x;
		if (idx >= (*w_gpu)) {
			idx = idx_init;
			idy += blockDim.y*gridDim.y;
		}
	}
}
__device__ void img_luminance(bool reverse, float *dst, int* h_gpu, int* w_gpu) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_init = idx;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	while (idx<(*w_gpu) && idy<(*h_gpu)) {
		int index = idy*(*w_gpu) + idx;
		if (reverse) {
			dst[index] = (tex2D(tex_output_b, idx, idy) + 3 * tex2D(tex_output_g, idx, idy) + 2 * tex2D(tex_output_r, idx, idy)) / 6;
		}
		else {
			dst[index] = (tex2D(tex_input_b, idx, idy) + 3 * tex2D(tex_input_g, idx, idy) + 2 * tex2D(tex_input_r, idx, idy)) / 6;
		}
		idx += blockDim.x*gridDim.x;
		if (idx >= (*w_gpu)) {
			idx = idx_init;
			idy += blockDim.y*gridDim.y;
		}
	}
}

__global__ void kernel(float *dev_input_b, float *dev_input_g, float *dev_input_r, float *dev_output_b, float *dev_output_g, float *dev_output_r, float *img_deriv, float *output_deriv, int* h_gpu, int *w_gpu, int *nbytes_gpu) {
	img_luminance(false, img_deriv, h_gpu, w_gpu);
	deblur(dev_output_b, dev_output_g, dev_output_r, output_deriv, nbytes_gpu, h_gpu, w_gpu, false);
	//calculate_grad(true, img_deriv, nbytes_gpu, h_gpu, w_gpu);
	//grad_refine(dev_input_b, dev_input_g, dev_input_r, output_deriv, nbytes_gpu, h_gpu, w_gpu, true, false);
}

int main(int argc, char **argv)
{
	std::string img_path = "C:/Users/75909/Desktop/1080p.jpeg";
	std::string save_path = "C:/Users/75909/Desktop/result.bmp";
	std::cout << img_path << std::endl;
	std::cout << save_path << std::endl;
	cv::Mat img = cv::imread(img_path);
	assert(img.channels() == 3);
	cv::resize(img, img, cv::Size((int)img.cols * 2, (int)img.rows * 2), 0, 0, cv::INTER_CUBIC);
	int w = img.cols;
	int h = img.rows;
	int nbytes = w*h * sizeof(float);
	dim3 thread_num(8, 8);
	dim3 block_num(64, 64);
	//dim3 grid = (w*h + block.x - 1) / block.x;
	//float *img_vector = new float[3 * nbytes / sizeof(float)];
	float *img_vector_b;
	float *img_vector_g;
	float *img_vector_r;
	cudaMallocHost((void**)&img_vector_b, nbytes, cudaHostAllocDefault);
	cudaMallocHost((void**)&img_vector_g, nbytes, cudaHostAllocDefault);
	cudaMallocHost((void**)&img_vector_r, nbytes, cudaHostAllocDefault);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			cv::Vec3b* in_row = img.ptr<cv::Vec3b>(i);
			img_vector_b[i*w + j] = in_row[j][0];
			img_vector_g[i*w + j] = in_row[j][1];
			img_vector_r[i*w + j] = in_row[j][2];
		}
	}

	//首先打印的是channel*col
	//然后按row打印
	//转的一维数组 就是直接按这个顺序衔接起来
	float *dev_input_b, *dev_input_g, *dev_input_r, *dev_output_b, *dev_output_g, *dev_output_r, *img_deriv, *output_deriv;
	int *w_gpu, *h_gpu, *nbytes_gpu;
	cudaMalloc((float**)&dev_input_b, nbytes);
	cudaMalloc((float**)&dev_input_g, nbytes);
	cudaMalloc((float**)&dev_input_r, nbytes);
	cudaMalloc((float**)&dev_output_b, nbytes);
	cudaMalloc((float**)&dev_output_g, nbytes);
	cudaMalloc((float**)&dev_output_r, nbytes);
	cudaMalloc((float**)&img_deriv, nbytes);
	cudaMalloc((float**)&output_deriv, nbytes);
	cudaMalloc((int**)&w_gpu, sizeof(int));
	cudaMalloc((int**)&h_gpu, sizeof(int));
	cudaMalloc((int**)&nbytes_gpu, sizeof(int));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, tex_input_b, dev_input_b, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_input_g, dev_input_g, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_input_r, dev_input_r, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_output_b, dev_output_b, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_output_g, dev_output_g, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_output_r, dev_output_r, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_img_deriv, img_deriv, desc, w, h, w * sizeof(float));
	cudaBindTexture2D(NULL, tex_output_deriv, output_deriv, desc, w, h, w * sizeof(float));
	float *resul;
	cudaMallocHost((void**)&resul, 3 * nbytes, cudaHostAllocDefault);
	//float *result_deriv = new float[nbytes / sizeof(float)];
	cudaEvent_t g_start, g_end;
	cudaEventCreate(&g_start, 0);
	cudaEventCreate(&g_end, 0);
	float time;
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();


	cudaMemcpy(dev_input_b, img_vector_b, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input_g, img_vector_g, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input_r, img_vector_r, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu, &w, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(h_gpu, &h, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nbytes_gpu, &nbytes, sizeof(int), cudaMemcpyHostToDevice);
	//while (1) {
	cudaEventRecord(g_start);
	kernel << <block_num, thread_num >> > (dev_input_b, dev_input_g, dev_input_r, dev_output_b, dev_output_g, dev_output_r, img_deriv, output_deriv, h_gpu, w_gpu, nbytes_gpu);
	//img_luminance << <block_num, thread_num >> > (false, img_deriv, h_gpu, w_gpu);

	//deblur << <block_num, thread_num >> > (dev_output_b, dev_output_g, dev_output_r, output_deriv, nbytes_gpu, h_gpu, w_gpu, false);

	//calculate_grad << <block_num, thread_num >> > (true, img_deriv, nbytes_gpu, h_gpu, w_gpu);
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
	//grad_refine << <block_num, thread_num >> > (dev_input_b, dev_input_g, dev_input_r, output_deriv, nbytes_gpu, h_gpu, w_gpu, true, false);
	cudaDeviceSynchronize();
	cudaEventRecord(g_end);
	cudaEventSynchronize(g_end);

	cudaEventElapsedTime(&time, g_start, g_end);
	std::cout << "GPU time: " << time << std::endl;
	//}
	float *resul_b, *resul_g, *resul_r;
	cudaMallocHost((void**)&resul_b, nbytes, cudaHostAllocDefault);
	cudaMallocHost((void**)&resul_g, nbytes, cudaHostAllocDefault);
	cudaMallocHost((void**)&resul_r, nbytes, cudaHostAllocDefault);
	cudaMemcpy(resul_b, dev_input_b, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(resul_g, dev_input_g, nbytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(resul_r, dev_input_r, nbytes, cudaMemcpyDeviceToHost);


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
	//for (int i = 0; i < h; ++i) {
	//	for (int j = 0; j < w; ++j) {
	//		cv::Vec3b* out_row = out_pic.ptr<cv::Vec3b>(i);
	//		out_row[j][0] = (int)resul[3 * (i*w + j)];
	//		out_row[j][1] = (int)resul[3 * (i*w + j) + 1];
	//		out_row[j][2] = (int)resul[3 * (i*w + j) + 2];
	//	}
	//}
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			cv::Vec3b* out_row = out_pic.ptr<cv::Vec3b>(i);
			out_row[j][0] = (int)resul_b[i*w + j];
			out_row[j][1] = (int)resul_g[i*w + j];
			out_row[j][2] = (int)resul_r[i*w + j];
		}
	}

	cv::imwrite(save_path, out_pic);
	std::chrono::duration<double, std::milli> time_span = std::chrono::high_resolution_clock::now() - start;
	std::cout << "time:" << time_span.count() << std::endl;

	cudaUnbindTexture(tex_img_deriv);
	cudaUnbindTexture(tex_output_deriv);
	cudaUnbindTexture(tex_input_b);
	cudaUnbindTexture(tex_input_g);
	cudaUnbindTexture(tex_input_r);
	cudaUnbindTexture(tex_output_b);
	cudaUnbindTexture(tex_output_g);
	cudaUnbindTexture(tex_output_r);
	cudaEventDestroy(g_start);
	cudaEventDestroy(g_end);
	cudaFreeHost(img_vector_b);
	cudaFreeHost(img_vector_g);
	cudaFreeHost(img_vector_r);
	cudaFreeHost(resul);
	cudaFree(dev_input_b);
	cudaFree(dev_input_g);
	cudaFree(dev_input_r);
	cudaFree(dev_output_b);
	cudaFree(dev_output_g);
	cudaFree(dev_output_r);
	cudaFree(img_deriv);
	cudaFree(output_deriv);
	cudaFree(w_gpu);
	cudaFree(h_gpu);
	cudaFree(nbytes_gpu);

	return 0;
}