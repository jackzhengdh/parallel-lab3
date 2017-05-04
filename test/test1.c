#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define tpb 1024

__global__ void getmaxcu(int *nums, int *output, int N) {

	__shared__ int sdata[tpb];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
		sdata[tid] = a[i];

	__syncthreads();

	for(unsigned int s=tpb/2 ; s >= 1 ; s=s/2) {
		if(tid < s) {
			if(sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}

	if(tid == 0)
		d[blockIdx.x] = sdata[0];
}

int main(int argc, char *argv[]) {
	unsigned int N;
	unsigned int i;
	unsigned int *nums;
	unsigned int *output;
	unsigned int nblocks;

	if (argc != 2) {
		printf("Error: input 1 number as size of array");
		exit(1);
	}

	N = atol(argv[1]);
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

	unsigned int max = 0;

	srand(time(NULL));
	for (i = 0; i < N; i++) {
		nums[i] = rand() % N;
		if (max < nums[i])
			max = nums[i];
		// printf("%d\n", nums[i]);
	}

	cudaMemcpy(dev_num, nums, N*sizeof(unsigned int), cudaMemcpyHostToDevice);

	getmaxcu<<<nblocks, tpb>>>(dev_num, dev_out, N);

	cudaMemcpy(output, dev_out, nblocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	printf("gpu max = %u\n", output[0]);

	cudaFree(dev_out);
	cudaFree(dev_num);
	free(nums);
	free(output);
}

