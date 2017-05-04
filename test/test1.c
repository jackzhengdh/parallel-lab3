#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int getmaxcu(unsigned int *numbers, unsigned int size);

int main(int argc, char *argv[]) {
	unsigned int size = 0;
	unsigned int i;
	unsigned int *numbers;

	if (argc != 2) {
		printf("Error: input 1 number as size of array");
		exit(1);
	}

	size = atol(argv[1]);
	numbers = (unsigned int *)malloc(size * sizeof(unsigned int));

	if (!numbers) {
		printf("Unable to allocate memory for array of size %u\n", size);
		exit(1);
	}

	srand(time(NULL));
	for (i = 0; i < size; i++) {
		numbers[i] = rand() % size;
		printf("%d\n", numbers[i]);
	}




}

unsigned int getmaxcu()