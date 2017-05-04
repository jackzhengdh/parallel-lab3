#define N 20
#define THREADS_PER_BLOCK 512

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

int main(void) {
	int *a, *b, *c;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;		// device copies of a, b, c
	int size = N * sizeof(int);

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	int i;
	for (i = 0; i < size; i++) {
		a[i] = rand() % size;
		b[i] = rand() % size;
		// printf("%d\n", a[i]);
	}
	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	
	printf("%5s %5s %5s\n", "a[]", "b[]", "c[]");
	for (i = 0; i < size; i++)
		printf("%5d %5d %5d\n", a[i], b[i], c[i]);
	printf("\n");

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}



