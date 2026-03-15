.section .text
.globl vector_clamp

vector_clamp:
    li t2, 255  #set clamp to 255

                #a0 is the input array address
                #a1 is the output array address
                #a2 is the total number of elements

loop:
    vsetvli t1, a2, e32, m1, ta, ma     #Set the VL, we are configuring the vector type such that elements = 32 bit,
                                        #registers used individually and old data in extra parts of the vector arent preserved(agnostic)
    vle32.v v1, (a0)                    #we load data from the input address to v1

    vmsgt.vx v0, v1, t2                 #we set the mask in v0[i] to 1 if v1[i] >255

    vmerge.vxm v2, v1, t2, v0           #we store output in v2, if v0[i] == 1 ? v2[i] = 255 : v2[i] = v1[i]

    vse32.v v2, (a1)                    #we store the new data in v2 to the output a1

    sub a2, a2, t1                      #we subtract VL from the total number of elements to decrement our counter

    slli t1, t1, 2                      #byte offset(t1*2^2)
    
    add a0, a0, t1                      #increment pointer for input array
    add a1, a1, t1                      #increment pointer for output array

    bnez a2, loop                       #if (a2 != 0) i.e. all the elements have not been processed, go back to top of >

    ret                                 # all elements have been processed, end the function