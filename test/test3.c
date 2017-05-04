#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define N 20
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

int main(void) {
	int *a, *b, *c;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;		// device copies of a, b, c
	int size = N * sizeof(int);

	printf("Strating cudaMalloc\n");
	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	printf("Completed cudaMalloc\n");

	printf("Starting malloc\n");
	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	printf("Completed malloc\n");

	printf("Starting assigning rands\n");
	int i;
	for (i = 0; i < size; i++) {
		a[i] = rand() % size;
		b[i] = rand() % size;
		// printf("%d\n", a[i]);
	}
	printf("Completed assigning rands\n");

	printf("Starting copying data from host to device\n");
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	printf("Completed copying data from host to device\n");

	printf("Executing kernal function add\n");
	// Launch add() kernel on GPU
	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
	printf("Completed executing kernal function add\n");

	printf("Copying data from device to host\n");
	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	printf("Completed copying data from device to host\n");
	
	printf("%5s %5s %5s\n", "a[]", "b[]", "c[]");
	for (i = 0; i < size; i++)
		printf("%5d %5d %5d\n", a[i], b[i], c[i]);
	printf("\n");

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}



