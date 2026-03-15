import pygame
import mmap
import struct
from nbody_visualizer import draw_gui

N = 100                          # must match .equ N in  assembly
FLOAT_SIZE = 4                  # single precision float = 4 bytes
FLAG_SIZE = 4                   # 1 integer flag at the start
# layout: [flag(4 bytes)] [p_x(N*4 bytes)] [p_y(N*4 bytes)] [p_z(N*4 bytes)]
FILE_SIZE = FLAG_SIZE + 3 * N * FLOAT_SIZE

# create the shared file if it doesnt exist, sized correctly
with open("nbody_shared.bin", "wb") as f:
    f.write(b'\x00' * FILE_SIZE)

# open it for memory mapping
f = open("nbody_shared.bin", "r+b")
mm = mmap.mmap(f.fileno(), FILE_SIZE)

pygame.init()   # must init before the loop calls pygame.event.get()

print("waiting for assembly to write...")

while True:
    # always pump pygame events first so the window never freezes
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            mm.close()
            f.close()
            pygame.quit()
            exit()

    # check the flag
    mm.seek(0)
    flag = struct.unpack("i", mm.read(4))[0]

    if flag == 1:
        # assembly finished writing a frame, read the three position arrays
        x = list(struct.unpack(f"{N}f", mm.read(N * FLOAT_SIZE)))
        y = list(struct.unpack(f"{N}f", mm.read(N * FLOAT_SIZE)))
        z = list(struct.unpack(f"{N}f", mm.read(N * FLOAT_SIZE)))

        # set flag back to 0 so assembly knows it can write the next frame
        mm.seek(0)
        mm.write(struct.pack("i", 0))
        mm.flush()

        # draw the frame
        if not draw_gui(x, y, z):
            break
    else:
        # no new frame yet, sleep 1ms so we dont burn cpu and starve qemu
        pygame.time.wait(1)

mm.close()
f.close()