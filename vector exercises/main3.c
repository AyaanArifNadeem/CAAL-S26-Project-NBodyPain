#include <stdio.h>

// The assembly function returns an int
// a0 = address of prices, a1 = address of quantities, a2 = n
extern int dot_product(int *prices, int *quantities, int n);

int main() {
    int n = 5;
    int prices[] = {10, 20, 30, 40, 50};     // Cost per item
    int quantities[] = {1, 2, 0, 4, 1};     // Number of items bought

    // Expected: (10*1) + (20*2) + (30*0) + (40*4) + (50*1)
    // 10 + 40 + 0 + 160 + 50 = 260

    int total = dot_product(prices, quantities, n);

    printf("--- Shopping Receipt (Dot Product) ---\n");
    for (int i = 0; i < n; i++) {
        printf("Item %d: %d @ $%d\n", i + 1, quantities[i], prices[i]);
    }
    printf("--------------------------------------\n");
    printf("Total Receipt Amount: $%d\n", total);

    return 0;
}   