#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define tpb 1024 // number of threads per block set to 1024 based on 

__global__ void getmaxcu(unsigned int *nums, unsigned int *output, int N) {

	__shared__ int sdata[tpb];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int temp = 0;

	if (i < N) {
		if (temp < nums[i])
			temp = nums[i];
	}
	sdata[tid] = temp;
	__syncthreads();

	for(unsigned int s=tpb/2 ; s >= 1 ; s=s/2) {
		if(tid < s) {
			if(sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid == 0)
		output[blockIdx.x] = sdata[0];
}

int main(int argc, char *argv[]) {
	int i;
	int N;
	int nblocks;
	unsigned int *nums;
	unsigned int *output;
	

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
	output = (unsigned int *)malloc(nblocks * sizeof(unsigned int));
	if (!output) {
		printf("Unable to allocate mem for output of size %u\n", nblocks);
		exit(1);
	}

	unsigned int *dev_num, *dev_out;
	cudaMalloc((void **) &dev_num, N * sizeof(unsigned int));
	cudaMalloc((void **) &dev_out, nblocks * sizeof(unsigned int));

	srand(time(NULL));
	for (i = 0; i < N; i++)
		nums[i] = rand() % N;

	while (1) {
		cudaMemcpy(dev_num, nums, N*sizeof(unsigned int), cudaMemcpyHostToDevice);
		getmaxcu<<<nblocks, tpb>>>(dev_num, dev_out, N);
		cudaMemcpy(output, dev_out, nblocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
				
		if (nblocks == 1) {
			printf("The maximum number in the array is: %u\n", output[0]);
			break;
		}

		N = nblocks;
		nblocks = N / tpb + 1;
		cudaFree(dev_out);
		cudaFree(dev_num);
		free(nums);
		nums = (unsigned int *)malloc(N * sizeof(unsigned int));
		memcpy(nums, output, N*sizeof(unsigned int));
		free(output);
		output = (unsigned int *)malloc(nblocks * sizeof(unsigned int));
		cudaMalloc((void **) &dev_num, N * sizeof(unsigned int));
		cudaMalloc((void **) &dev_out, nblocks * sizeof(unsigned int));
	}
}

