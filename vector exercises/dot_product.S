.section .text
.globl dot_product

dot_product:
    li t5, 0                            #Total Sum
    vsetvli t0, x0, e32, m1, ta, ma     #We use a dummy vsetvli to tell the CPU we are using e32, m1
    vmv.v.i v0, 0                       #Initialize v0 to 0

loop:
                                      #a2 is the total number of elements
    vsetvli t1, a2, e32, m1, ta, ma   #Set the VL, we are configuring the vector type such that elements = 32 bit,
                                      #registers used individually and old data in extra parts of the vector arent >
    vle32.v v1, (a0)                  # Load chunk of Prices from a0
    vle32.v v2, (a1)                  # Load chunk of Quantities from a1

    vmul.vv v3, v1, v2                # Vector-Vector Multiplication: v3[i] = v1[i] * v2[i]

    vredsum.vs v8, v3, v0             # Reduction: Sum all elements in v3, add to v0[0], store in v4[0]

    vmv.x.s t4, v8                    # Move the result from the vector register v4[0] to scalar register t4

    add t5, t5, t4                    # Add the chunk's sum to our global total

    sub a2, a2, t1                    # Decrement remaining element count
    slli t1, t1, 2                    # Convert elements processed to byte offset (*4)
    add a0, a0, t1                    # Advance Price pointer
    add a1, a1, t1                    # Advance Quantity pointer

    bnez a2, loop                     # If elements remain, repeat

    mv a0, t5                         # Final Step: Move the answer to a0 so C can read it
    ret                               # all elements have been processed, end the function


