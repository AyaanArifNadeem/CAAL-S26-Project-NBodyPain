.equ N, 10
.equ NUM_STEPS, 10

# btw, .equ does not work with types like floats. unlike li, there's no fli instruction, so floats gotta be kept in memory. 

# data section, arrays and constants are here 
# ----------------
    .section .data
    .align 2            # aligning data to a 4-byte boundary for floats. 

dt: 
    .float 3600.0       # 1 hour in seconds
G:
    .float 6.6743e-11   # just the G constant
epsilon_sq:
    .float 1.0e16       # softening^2, cause computing epsilon^2 every time would add an extra fmul.s to every iteration of the inner loop 
half_dt:
    .float 0.5          # used to compute half_dt = dt * 0.5, cause kick_half_step needs it for division
zero_f:
    .float 0.0          # reset accelerations to 0


# 
# arrays for all the bodies - N elements each, 4 bytes per element
# essentially doing the 'mass = [0.0] * N'
# *****REPLACE the .float 0.0 lines with output from generate.py*****
# 

# arrays (mass_grid, p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z) are all
# declared in mydata.S by generate.py — do NOT redeclare them here or
# calculate_forces will load these zero-filled local copies instead of the
# real data, and every fdiv will produce NaN


    .section .text
    .globl main         # .text for defining where all our code is, and .globl for defining the main function

# main - effectively run_simulation() from the python code
# 1. computes the initial accelerations
# 2. runs the leapfrog loop NUM_STEP times.
#
# stack frame layout: (32 bytes cause ra + s0 + s1 = 24, but sp must be a multiple of 16 so 32)
# ra = sp + 24 (for calling subfunctions)
# s0 = sp + 16 (step counter)
# s1 = sp + 8 (NUM_STEPS constant across loops)
# sp + 0 is just padding to satisfy the alignment requirement for call functions

main:
    addi sp, sp, -32        # allocate 32 bytes
    sd ra, 24(sp)           # save return address (call clobbers ra)
    sd s0, 16(sp)           # save s0 (use it as step counter)
    sd s1, 8(sp)            # save s1 (store NUM_STEPS) 

    li s0, 0                # s0 = step counter
    li s1, NUM_STEPS        # s1 = total number of steps (loop limit)

    # we need to compute the initial accelerations, since the leapfrog integration requires a[t = 0] before the first half_kick
    call calculate_forces

    # --- DEBUG CHECKPOINT 1: print accs after initial force calc ---
    ## if all 0.0 then calculate_forces is broken. remove once verified.
    la a0, a_x
    la a1, a_y
    la a2, a_z
    li a3, 10 # number of bodies to show in debug
    call print_acc
    # --- END DEBUG ---

main_loop:
    bge s0, s1, main_loop_end   # if step >= NUM_STEPS then exit loop
    # save the loop state to the stack since function calls can clobber the s-registers
    # s1 and s0 are the loop variables so we must protect them here before the calls
    addi sp, sp, -16        # allocate 16 bytes
    sd s0, 8(sp)            # save curr step counter
    sd s1, 0(sp)            # save NUM_STEPS

    # Leapfrog Step 1: Kick (half timestep)
    # v(t + dt/2) = v(t) + a(t) * dt/2
    call kick_half_step

    # Leapfrog Step 2: Drift (full timestep)
    # x(t + dt) = x(t) + v(t + dt/2) * dt
    call drift 

    # Leapfrog Step 3: recalculate forces at new positions
    # a(t + dt) from the updated step
    call calculate_forces

    # leapfrog step 4: Kick (half timestep again)
    call kick_half_step

    # restore the loop state after all functions
    ld s1, 0(sp)            # retrieve NUM_STEPS
    ld s0, 8(sp)            # retrieve step counter
    addi sp, sp, 16         # deallocate

    addi s0, s0, 1          # step_counter++
    j main_loop             # start a new iteration

# this is where the code goes when the loop ends
main_loop_end:
    # - debug start
    addi sp, sp, -16
    sd s0, 8(sp)
    sd s1, 0(sp)

    la a0, p_x
    la a1, p_y
    la a2, p_z
    la a3, v_x
    la a4, v_y
    la a5, v_z
    li a6, 3
    call print_state
    # -de bug end, the print statements

    ld s1, 0(sp)
    ld s0, 8(sp)
    addi sp, sp, 16

    ld ra, 24(sp)   # restore ra
    ld s0, 16(sp)   # restore s0
    ld s1, 8(sp)    # restore s1
    addi sp, sp, 32 # deallocate
    
    li a0, 0        # return 0 for the C runtime
    ret 

# this is a bit of a long one, since calculate_forces() is the biggest function in the code. 
#
# PYTHON EQUIVALENT:
#   for i in range(n):                     
#       ax[i] = ay[i] = az[i] = 0
# 
#   for i in range(n):                     
#       for j in range(i+1, n):
#           dx = x[j]-x[i]; dy = ...; dz = ...
#           r_soft = sqrt(dx^2+dy^2+dz^2 + eps^2)
#           f = G*mi*mj / r_soft^3
#           ax[i] += f*dx/mi;  ax[j] -= f*dx/mj  
#           ay/az etc....
#
# 4 phases:
# 1) save everything - takes up a lot of space because of how many values need to
#    be live at the same time inside the inner loop
# 2) reset acc_x,y,z to 0 - same as the first for-loop in python
# 3) actual pairwise calculation - outer loop over i, inner loop over j
# 4) restore everything and return
#
# Integer register allocation:
#   s0 = base address of pos_x
#   s1 = ''   ''      '' pos_y
#   s2 = ''   ''      '' pos_z
#   s3 = ''   ''      '' mass
#   s4 = ''   ''      '' acc_x
#   s5 = ''   ''      '' acc_y
#   s6 = ''   ''      '' acc_z
#   s7 = outer loop counter i
#   s8 = inner loop counter j
#   s9 = N
#   s10 = byte offset for body i (i * 4)
#   s11 = byte offset for body j (j * 4)
#   t0-t3 = scratch for address arithmetic
#
# Float register allocation:
#   fs0 = G (gravitational constant)
#   fs1 = epsilon^2 (softening squared)
#   fs2 = pos_x[i] (xi, cached for inner loop)
#   fs3 = pos_y[i] (yi)
#   fs4 = pos_z[i] (zi)
#   fs5 = mass[i] (mi, cached for inner loop)
#   fs6 = dx
#   fs7 = dy
#   fs8 = dz
#   fs9 = r_soft^3 (denominator of force formula)
#   fs10 = force_magnitude
#   fs11 = scratch for load-modify-store of acc arrays
#   ft0-ft10 = temporaries (caller-saved, clobbered freely)
#
# Stack frame: 208 bytes (13 int regs x 8 + 12 float regs x 8 = 200, padded to 208)
#   sp+0: fs11      sp+8 : fs10
#   sp+16: fs9      sp+24: fs8
#   sp+32: fs7      sp+40: fs6
#   sp+48: fs5      sp+56: fs4
#   sp+64: fs3      sp+72: fs2
#   sp+80: fs1      sp+88: fs0
#   sp+96: s11      sp+104: s10
#   sp+112: s9      sp+120: s8
#   sp+128: s7      sp+136: s6
#   sp+144: s5      sp+152: s4
#   sp+160: s3      sp+168: s2
#   sp+176: s1      sp+184: s0
#   sp+192: ra      sp+200: 8-byte alignment padding

calculate_forces:
    addi sp, sp, -208   # 208-byte stack frame

    # save integer callee-saved registers
    sd ra, 192(sp)      # save return address
    sd s0, 184(sp)      # save s0
    sd s1, 176(sp)      # same for all of these
    sd s2, 168(sp)            
    sd s3, 160(sp)            
    sd s4, 152(sp)            
    sd s5, 144(sp)            
    sd s6, 136(sp)            
    sd s7, 128(sp)            
    sd s8, 120(sp)            
    sd s9, 112(sp)            
    sd s10, 104(sp)            
    sd s11, 96(sp)
    # save float callee-saved regs
    fsd fs0, 88(sp)     # save fs0
    fsd fs1, 80(sp)     # same for the rest
    fsd fs2, 72(sp)
    fsd fs3, 64(sp)
    fsd fs4, 56(sp)
    fsd fs5, 48(sp)
    fsd fs6, 40(sp)
    fsd fs7, 32(sp)
    fsd fs8, 24(sp)
    fsd fs9, 16(sp)
    fsd fs10, 8(sp)
    fsd fs11, 0(sp)

    # load base addresses of all arrays into saved regs
    la s0, p_x    # s0 = &pos_x[0]
    la s1, p_y    # same for y
    la s2, p_z    # same for z
    la s3, mass_grid  # s3 = &mass_grid[0]  (from mydata.S, not a local array)
    la s4, a_x    # s4 = &acc_x[0]
    la s5, a_y    # same for y
    la s6, a_z    # same for z

    # load fp constants 
    la t0, G  # load address of G
    flw fs0, 0(t0)  # fs0 = G

    la t0, epsilon_sq
    flw fs1, 0(t0)  # fs1 = epsilon^2

    la t0, zero_f
    flw ft11, 0(t0) # ft11 = 0.0 (used in reset loop)

    li s9, N        # s9 = N
    li s7, 0        # s7 = i = 0

reset_loop:
    bge s7, s9, reset_loop_end  # if i >= N we're done resetting
    slli s10, s7, 2     # s10 = i * 4 (floats are 4 bytes)

    add t0, s4, s10     # t0 = &acc_x[i]
    fsw ft11, 0(t0)     # acc_x[i] = 0

    add t0, s5, s10     # t0 = &acc_y[i]
    fsw ft11, 0(t0)     # acc_y[i] = 0

    add t0, s6, s10     # t0 = &acc_z[i]
    fsw ft11, 0(t0)     # acc_z[i] = 0

    addi s7, s7, 1      # i++
    j reset_loop

reset_loop_end:
    li s7, 0            # reset i = 0 for the outer loop

# outer loop: i from 0 to N-2
# inner loop: j from i+1 to N-1

outer_loop:
    addi t0, s9, -1              # t0 = N - 1 (outer loop only goes to N-2)
    bge s7, t0, outer_loop_end  # if i >= N-1 then all pairs done

    slli s10, s7, 2     # s10 = i * 4

    # cache position and mass of body i for the entire inner loop
    # avoids reloading from RAM on every j iteration
    add t0, s0, s10     # t0 = &pos_x[i]
    flw fs2, 0(t0)      # fs2 = xi

    add t0, s1, s10     # t0 = &pos_y[i]
    flw fs3, 0(t0)      # fs3 = yi

    add t0, s2, s10     # t0 = &pos_z[i]
    flw fs4, 0(t0)      # fs4 = zi

    add t0, s3, s10     # t0 = &mass[i]
    flw fs5, 0(t0)      # fs5 = mi

    addi s8, s7, 1      # s8 = j = i + 1

inner_loop:
    bge s8, s9, inner_loop_end  # if j >= N move to next i
    slli s11, s8, 2     # s11 = j * 4

    # displacement vector from body i to body j
    add t0, s0, s11     # t0 = &pos_x[j]
    flw ft0, 0(t0)      # ft0 = xj
    fsub.s fs6, ft0, fs2# fs6 = dx = xj - xi

    add t0, s1, s11     # &pos_y[j]
    flw ft0, 0(t0)      # ft0 = yj
    fsub.s fs7, ft0, fs3# fs7 = dy = yj - yi

    add t0, s2, s11     # &pos_z[j]
    flw ft0, 0(t0)      # ft0 = zj
    fsub.s fs8, ft0, fs4# fs8 = dz = zj - zi

    # softened distance: r_soft = sqrt(dx^2 + dy^2 + dz^2 + eps^2)
    # r_soft^3 = r_soft^2 * r_soft, used as denominator
    fmul.s ft0, fs6, fs6    # ft0 = dx^2
    fmul.s ft1, fs7, fs7    # ft1 = dy^2
    fmul.s ft2, fs8, fs8    # ft2 = dz^2

    fadd.s ft3, ft0, ft1    # ft3 = dx^2 + dy^2
    fadd.s ft3, ft3, ft2    # ft3 = r^2

    fadd.s ft4, ft3, fs1    # ft4 = r_soft^2 = r^2 + eps^2

    fsqrt.s ft5, ft4        # ft5 = r_soft (risc-v has a native sqrt instruction)

    fmul.s fs9, ft4, ft5    # fs9 = r_soft^3

    # force magnitude = G * mi * mj / r_soft^3
    # NOTE: divide by r_soft^3 FIRST to avoid overflow.
    # doing G*mi*mj first hits ~6e46 which blows past the single precision max of 3.4e38
    add t0, s3, s11         # t0 = &mass[j]
    flw ft6, 0(t0)          # ft6 = mj

    fdiv.s ft7, fs0, fs9    # ft7 = G / r_soft^3
    fmul.s ft7, ft7, fs5    # ft7 = (G / r_soft^3) * mi
    fmul.s fs10, ft7, ft6   # fs10 = force_magnitude = (G / r_soft^3) * mi * mj

    # force components along each axis
    fmul.s ft8, fs10, fs6   # ft8 = fx
    fmul.s ft9, fs10, fs7   # ft9 = fy
    fmul.s ft10, fs10, fs8  # ft10 = fz

    # update acc of body i: acc[i] += F / mi
    add t0, s4, s10         # t0 = &acc_x[i]
    flw fs11, 0(t0)         # fs11 = acc_x[i]
    fdiv.s ft0, ft8, fs5    # ft0 = fx / mi
    fadd.s fs11, fs11, ft0  # acc_x[i] + fx/mi
    fsw fs11, 0(t0)         # store back

    add t0, s5, s10         # t0 = &acc_y[i]
    flw fs11, 0(t0)         # fs11 = acc_y[i]
    fdiv.s ft0, ft9, fs5    # ft0 = fy / mi
    fadd.s fs11, fs11, ft0  # acc_y[i] + fy/mi
    fsw fs11, 0(t0)         # store back

    add t0, s6, s10         # t0 = &acc_z[i]
    flw fs11, 0(t0)         # fs11 = acc_z[i]
    fdiv.s ft0, ft10, fs5   # ft0 = fz / mi
    fadd.s fs11, fs11, ft0  # acc_z[i] + fz/mi
    fsw fs11, 0(t0)         # store back

    # update acc of body j: acc[j] -= F / mj  (newton's 3rd law, opposite direction)
    add t0, s4, s11         # t0 = &acc_x[j]
    flw fs11, 0(t0)         # fs11 = acc_x[j]
    fdiv.s ft0, ft8, ft6    # ft0 = fx / mj
    fsub.s fs11, fs11, ft0  # acc_x[j] - fx/mj
    fsw fs11, 0(t0)         # store back

    add t0, s5, s11         # t0 = &acc_y[j]
    flw fs11, 0(t0)         # fs11 = acc_y[j]
    fdiv.s ft0, ft9, ft6    # ft0 = fy / mj
    fsub.s fs11, fs11, ft0  # acc_y[j] - fy/mj
    fsw fs11, 0(t0)         # store back

    add t0, s6, s11         # t0 = &acc_z[j]
    flw fs11, 0(t0)         # fs11 = acc_z[j]
    fdiv.s ft0, ft10, ft6   # ft0 = fz / mj
    fsub.s fs11, fs11, ft0  # acc_z[j] - fz/mj
    fsw fs11, 0(t0)         # store back

    addi s8, s8, 1          # j++
    j inner_loop

inner_loop_end:
    addi s7, s7, 1          # i++
    j outer_loop

outer_loop_end:
    # epilogue: restore everything in reverse order
    fld fs11, 0(sp)
    fld fs10, 8(sp)
    fld fs9, 16(sp)
    fld fs8, 24(sp)
    fld fs7, 32(sp)
    fld fs6, 40(sp)
    fld fs5, 48(sp)
    fld fs4, 56(sp)
    fld fs3, 64(sp)
    fld fs2, 72(sp)
    fld fs1, 80(sp)
    fld fs0, 88(sp)

    ld s11, 96(sp)
    ld s10, 104(sp)
    ld s9, 112(sp)
    ld s8, 120(sp)
    ld s7, 128(sp)
    ld s6, 136(sp)
    ld s5, 144(sp)
    ld s4, 152(sp)
    ld s3, 160(sp)
    ld s2, 168(sp)
    ld s1, 176(sp)
    ld s0, 184(sp)
    ld ra, 192(sp)

    addi sp, sp, 208    # close the frame
    ret

# kick_half_step - update velocities by half a timestep
#
# PYTHON EQUIVALENT:
#   half_dt = dt / 2.0
#   for i in range(n):
#       vel_x[i] += acc_x[i] * half_dt
#       vel_y[i] += acc_y[i] * half_dt
#       vel_z[i] += acc_z[i] * half_dt
#
# register allocation:
#   s0 = &vel_x    s1 = &vel_y    s2 = &vel_z
#   s3 = &acc_x    s4 = &acc_y    s5 = &acc_z
#   s6 = i         s7 = N         s8 = byte offset i*4
#   fs0 = half_dt
#
# stack frame: 96 bytes
# (ra + s0-s8 = 9 regs = 72 bytes, + fs0 = 8 bytes = 80, pad to 96)
#   sp+0: fs0
#   sp+8: s8      sp+16: s7
#   sp+24: s6     sp+32: s5
#   sp+40: s4     sp+48: s3
#   sp+56: s2     sp+64: s1
#   sp+72: s0     sp+80: ra

kick_half_step:
    addi sp, sp, -96    # allocate 96 bytes

    sd ra, 80(sp)       # save return address
    sd s0, 72(sp)       # save s0
    sd s1, 64(sp)       # same for the rest
    sd s2, 56(sp)
    sd s3, 48(sp)
    sd s4, 40(sp)
    sd s5, 32(sp)
    sd s6, 24(sp)
    sd s7, 16(sp)
    sd s8,  8(sp)
    fsd fs0, 0(sp)      # save fs0

    la s0, v_x        # s0 = &vel_x[0]
    la s1, v_y        # s1 = &vel_y[0]
    la s2, v_z        # s2 = &vel_z[0]
    la s3, a_x        # s3 = &acc_x[0]
    la s4, a_y        # s4 = &acc_y[0]
    la s5, a_z        # s5 = &acc_z[0]

    # half_dt = dt * 0.5
    la t0, dt
    flw ft0, 0(t0)      # ft0 = dt
    la t0, half_dt
    flw ft1, 0(t0)      # ft1 = 0.5
    fmul.s fs0, ft0, ft1# fs0 = half_dt

    li s7, N            # s7 = N
    li s6, 0            # s6 = i = 0

kick_loop:
    bge s6, s7, kick_loop_end   # if i >= N then done
    slli s8, s6, 2      # s8 = i * 4

    # vel_x[i] += acc_x[i] * half_dt
    add t0, s3, s8      # t0 = &acc_x[i]
    flw ft2, 0(t0)      # ft2 = acc_x[i]
    fmul.s ft3, ft2, fs0# ft3 = acc_x[i] * half_dt
    add t0, s0, s8      # t0 = &vel_x[i]
    flw ft4, 0(t0)      # ft4 = vel_x[i]
    fadd.s ft4, ft4, ft3# vel_x[i] + acc_x[i]*half_dt
    fsw ft4, 0(t0)      # store back

    # vel_y[i] += acc_y[i] * half_dt
    add t0, s4, s8      # t0 = &acc_y[i]
    flw ft2, 0(t0)      # ft2 = acc_y[i]
    fmul.s ft3, ft2, fs0# ft3 = acc_y[i] * half_dt
    add t0, s1, s8      # t0 = &vel_y[i]
    flw ft4, 0(t0)      # ft4 = vel_y[i]
    fadd.s ft4, ft4, ft3# vel_y[i] + acc_y[i]*half_dt
    fsw ft4, 0(t0)      # store back

    # vel_z[i] += acc_z[i] * half_dt
    add t0, s5, s8      # t0 = &acc_z[i]
    flw ft2, 0(t0)      # ft2 = acc_z[i]
    fmul.s ft3, ft2, fs0# ft3 = acc_z[i] * half_dt
    add t0, s2, s8      # t0 = &vel_z[i]
    flw ft4, 0(t0)      # ft4 = vel_z[i]
    fadd.s ft4, ft4, ft3# vel_z[i] + acc_z[i]*half_dt
    fsw ft4, 0(t0)      # store back

    addi s6, s6, 1      # i++
    j kick_loop

kick_loop_end:
    # epilogue
    fld fs0,  0(sp)
    ld s8,   8(sp)
    ld s7,  16(sp)
    ld s6,  24(sp)
    ld s5,  32(sp)
    ld s4,  40(sp)
    ld s3,  48(sp)
    ld s2,  56(sp)
    ld s1,  64(sp)
    ld s0,  72(sp)
    ld ra,  80(sp)
    addi sp, sp, 96     # close the frame
    ret


# drift - update positions by a full timestep
#
# PYTHON EQUIVALENT:
#   for i in range(n):
#       pos_x[i] += vel_x[i] * dt
#       pos_y[i] += vel_y[i] * dt
#       pos_z[i] += vel_z[i] * dt
#
# same layout as kick_half_step, just using dt (full) instead of half_dt
#   s0 = &pos_x    s1 = &pos_y    s2 = &pos_z
#   s3 = &vel_x    s4 = &vel_y    s5 = &vel_z
#   s6 = i         s7 = N         s8 = byte offset
#   fs0 = dt
#
# stack frame: 96 bytes (same as kick_half_step)

drift:
    addi sp, sp, -96    # allocate 96 bytes

    sd ra, 80(sp)
    sd s0, 72(sp)
    sd s1, 64(sp)
    sd s2, 56(sp)
    sd s3, 48(sp)
    sd s4, 40(sp)
    sd s5, 32(sp)
    sd s6, 24(sp)
    sd s7, 16(sp)
    sd s8,  8(sp)
    fsd fs0, 0(sp)

    la s0, p_x        # s0 = &pos_x[0]
    la s1, p_y        # s1 = &pos_y[0]
    la s2, p_z        # s2 = &pos_z[0]
    la s3, v_x        # s3 = &vel_x[0]
    la s4, v_y        # s4 = &vel_y[0]
    la s5, v_z        # s5 = &vel_z[0]

    la t0, dt           # full dt this time (not halved, unlike kick_half_step)
    flw fs0, 0(t0)      # fs0 = dt

    li s7, N            # s7 = N
    li s6, 0            # s6 = i = 0

drift_loop:
    bge s6, s7, drift_loop_end  # if i >= N then done
    slli s8, s6, 2      # s8 = i * 4

    # pos_x[i] += vel_x[i] * dt
    add t0, s3, s8      # t0 = &vel_x[i]
    flw ft0, 0(t0)      # ft0 = vel_x[i]
    fmul.s ft1, ft0, fs0# ft1 = vel_x[i] * dt
    add t0, s0, s8      # t0 = &pos_x[i]
    flw ft2, 0(t0)      # ft2 = pos_x[i]
    fadd.s ft2, ft2, ft1# pos_x[i] + vel_x[i]*dt
    fsw ft2, 0(t0)      # store back

    # pos_y[i] += vel_y[i] * dt
    add t0, s4, s8      # t0 = &vel_y[i]
    flw ft0, 0(t0)      # ft0 = vel_y[i]
    fmul.s ft1, ft0, fs0# ft1 = vel_y[i] * dt
    add t0, s1, s8      # t0 = &pos_y[i]
    flw ft2, 0(t0)      # ft2 = pos_y[i]
    fadd.s ft2, ft2, ft1# pos_y[i] + vel_y[i]*dt
    fsw ft2, 0(t0)      # store back

    # pos_z[i] += vel_z[i] * dt
    add t0, s5, s8      # t0 = &vel_z[i]
    flw ft0, 0(t0)      # ft0 = vel_z[i]
    fmul.s ft1, ft0, fs0# ft1 = vel_z[i] * dt
    add t0, s2, s8      # t0 = &pos_z[i]
    flw ft2, 0(t0)      # ft2 = pos_z[i]
    fadd.s ft2, ft2, ft1# pos_z[i] + vel_z[i]*dt
    fsw ft2, 0(t0)      # store back

    addi s6, s6, 1      # i++
    j drift_loop

drift_loop_end:
    # epilogue
    fld fs0,  0(sp)
    ld s8,   8(sp)
    ld s7,  16(sp)
    ld s6,  24(sp)
    ld s5,  32(sp)
    ld s4,  40(sp)
    ld s3,  48(sp)
    ld s2,  56(sp)
    ld s1,  64(sp)
    ld s0,  72(sp)
    ld ra,  80(sp)
    addi sp, sp, 96     # close the frame
    ret