#include <stdio.h>

// This tells C the function is defined in our .S file
extern void vector_clamp(int *a, int *c, int n);

int main() {
    int n = 8;
    int input[] = {10, 300, 50, 400, 255, 256, 0, 1000};
    int output[8];

    vector_clamp(input, output, n);

    printf("Clamped Output:\n");
    for (int i = 0; i < n; i++) {
        printf("%d -> %d\n", input[i], output[i]);
    }

    return 0;
}


