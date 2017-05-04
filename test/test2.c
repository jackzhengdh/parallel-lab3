#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define tpb 1024 // number of threads per block set to 1024 based on 

__global__ void getmaxcu(unsigned int *nums, unsigned int *max, int *mutex, int N) {

	__shared__ int sdata[tpb];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	unsigned int temp = -1;

	while (i + offset < N) {
		if (temp < nums[i + offset])
			temp = nums[i + offset];
		offset += stride;
	}
	sdata[tid] = temp;


	// if (i < N) {
	// 	if (temp < nums[i])
	// 		temp = nums[i];
	// }
	// sdata[tid] = temp;
	__syncthreads();

	for(int s=tpb/2 ; s >= 1 ; s=s/2) {
		if(tid < s) {
			if(sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid == 0)
		while (atomicCAS(mutex, 0, 1) != 0);
		if (*max < sdata[0])
			*max = sdata[0];
		atomicExch(mutex, 0);
}

int main(int argc, char *argv[]) {
	int i;
	int N;
	int nblocks;
	unsigned int *nums;
	unsigned int *output;
	int *dev_mutex;
	

	if (argc != 2) {
		printf("Error: input 1 number as size of array");
		exit(1);
	}

	N = atol(argv[1]);

	// printf("Starting malloc\n");
	nums = (unsigned int *)malloc(N * sizeof(unsigned int));
	if (!nums) {
		printf("Unable to allocate mem for nums of size %u\n", N);
		exit(1);
	}
	

	nblocks = N / tpb + 1;
	output = (unsigned int *)malloc(sizeof(unsigned int));
	if (!output) {
		printf("Unable to allocate mem for output of size %u\n", nblocks);
		exit(1);
	}

	unsigned int *dev_num, *dev_out;
	cudaMalloc((void **) &dev_num, N * sizeof(unsigned int));
	cudaMalloc((void **) &dev_out, sizeof(unsigned int));
	cudaMalloc((void **) &dev_mutex, sizeof(int));
	cudaMemset(dev_out, 0, sizeof(unsigned int));
	cudaMemset(dev_mutex, 0, sizeof(unsigned int));

	srand(time(NULL));
	for (i = 0; i < N; i++)
		nums[i] = rand() % N;

	cudaMemcpy(dev_num, nums, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
	getmaxcu<<<nblocks, tpb>>>(dev_num, dev_out, dev_mutex, N);
	cudaMemcpy(output, dev_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	printf("The maximum number in the array is: %u\n", output[0]);
		

}

