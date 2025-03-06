#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 5

// Function to allocate an array of integers
int *allocate_array(int size) {
    int *arr = (int *)malloc(size * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return arr;
}

// Function to initialize the array with values
void initialize_array(int *arr, int size) {
    if (!arr) return; // Avoid segmentation fault
    for (int i = 0; i < size; i++) {
        arr[i] = i * 10;
    }
}

// Function to print the array
void print_array(int *arr, int size) {
    if (!arr) return; // Avoid segmentation fault
    printf("Array elements: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Function to create a duplicate of the array
int *duplicate_array(int *arr, int size) {
    if (!arr) return NULL;
    int *copy = (int *)malloc(size * sizeof(int));
    if (!copy) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    memcpy(copy, arr, size * sizeof(int));
    return copy;
}

// Function to free the allocated memory
void free_memory(int *arr) {
    if (arr) { // Ensure the pointer is not NULL
        free(arr);
    }
}

// Main function
int main() {
    int *array = allocate_array(SIZE);
    initialize_array(array, SIZE);
    print_array(array, SIZE);

    // Creating a duplicate array
    int *array_copy = duplicate_array(array, SIZE);
    print_array(array_copy, SIZE);

    // Free memory
    free_memory(array);       // Free the original array
    free_memory(array_copy);  // Free the duplicate array

    return 0; // No memory leak
}