
// test sample code

#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define tpb 1024

__global__ void kernel_min(int *a, int *d, int N) {

	__shared__ int sdata[tpb]; //"static" shared memory

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

int main(int argc, char* argv[]) {

	if (argc != 2)
		exit(1);

	const int N = atol(argv[1]);

	int i;
	
	int nblocks;


	nblocks = N / tpb + 1;
	// const int N=tpb*nblocks;
	srand(time(NULL));

	int *a;
	a = (int*)malloc(N * sizeof(int));
	int *d;
	d = (int*)malloc(nblocks * sizeof(int));

	int *dev_a, *dev_d;

	cudaMalloc((void **) &dev_a, N*sizeof(int));
	cudaMalloc((void **) &dev_d, nblocks*sizeof(int));
	int mmm=0;
	for( i = 0 ; i < N ; i++) {
		a[i] = rand()% N;
		// printf("%d ",a[i]);
		if(mmm<a[i]) 
			mmm=a[i];
	}
	printf("");

	cudaMemcpy(dev_a , a, N*sizeof(int),cudaMemcpyHostToDevice);

	kernel_min<<<nblocks, tpb>>>(dev_a, dev_d, N);

	cudaMemcpy(d, dev_d, nblocks*sizeof(int),cudaMemcpyDeviceToHost);

	printf("cpu max %d, gpu_max = %d\n\n",mmm,d[0]);

	// for (i = 0; i < nblocks; i++)
	// 	printf("%d ", d[i]);
	// printf("\n");

	cudaFree(dev_a);
	cudaFree(dev_d);

	printf("\n");

	return 0;
}