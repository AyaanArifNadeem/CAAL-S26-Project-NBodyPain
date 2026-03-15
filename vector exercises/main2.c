#include <stdio.h>

// The assembly function now RETURNS an int, so we change 'void' to 'int'
// It only needs the array address (a0) and the size (a1)
extern int nega_count(int *a, int n);

int main() {
    int n = 8;
    int input[] = {-10, -300,-50, -400, -255, 256, 0, -1000};

    // We capture the return value from a0 into 'result'
    int result = nega_count(input, n);

    printf("--- Negative Numbers Exercise Results ---\n");
    printf("Array: {");
    for (int i = 0; i < n; i++) {
        printf("%d%s", input[i], (i == n - 1) ? "" : ", ");
    }
    printf("}\n");

    // Print the final count
    printf("Total Negative Numbers found: %d\n", result);

    return 0;
}