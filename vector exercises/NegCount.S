.section .text
.globl nega_count

nega_count:
li t5,0
                #a0 is the input array address
                #a1 is the total number of elements

loop:
    vsetvli t1, a1, e32, m1, ta, ma     #Set the VL, we are configuring the vector type such that elements = 32 bit,
                                        #registers used individually and old data in extra parts of the vector arent preserved(agnostic)
    vle32.v v1, (a0)                    #we load data from the input address to v1

    vmslt.vx v0, v1, x0                 #we set the mask in v0[i] to 1 if v1[i] < 0

    vcpop.m t4, v0                      #we count the number of 1s in the mask i.e. the count of negative numbers in this chunk

    add t5, t5, t4                      #we add the current val of t4(neg numbers in this chunk) to the global count t5(total neg numbers)

    sub a1, a1, t1                      #we subtract VL from the total number of elements to decrement our counter

    slli t1, t1, 2                      #byte offset(t1*2^2)

    add a0, a0, t1                      #increment pointer for input array

    bnez a1, loop                       #if (a1 != 0) i.e. all the elements have not been processed, go back to top of loop

    mv a0, t5                           #move the value of t5 to a0 since that is the location of the output for our c wrapper

    ret                                 # all elements have been processed, end the function