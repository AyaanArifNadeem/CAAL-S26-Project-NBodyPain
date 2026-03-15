/*
 * debug.c — print helpers called from nbody.S
 *
 * Compile together with the assembly:
 *   riscv64-linux-gnu-gcc -static -o nbody nbody.S mydata.S debug.c
 */

#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>


/*
 * print_state — called from assembly via the a0-a6 argument registers
 *
 * Assembly calling convention (put these in a0-a6 before calling):
 *   a0 = float *px      base address of p_x array
 *   a1 = float *py      base address of p_y array
 *   a2 = float *pz      base address of p_z array
 *   a3 = float *vx      base address of v_x array
 *   a4 = float *vy      base address of v_y array
 *   a5 = float *vz      base address of v_z array
 *   a6 = int    n       number of bodies to print
 */
void print_state(float *px, float *py, float *pz,
                 float *vx, float *vy, float *vz,
                 int n)
{
    printf("---\n");
    for (int i = 0; i < n; i++) {
        printf("body %3d | pos=(%+.4e, %+.4e, %+.4e) | vel=(%+.4e, %+.4e, %+.4e)\n",
               i,
               px[i], py[i], pz[i],
               vx[i], vy[i], vz[i]);
    }
    printf("---\n");
}

/*
 * print_acc — same idea but for the acceleration arrays
 *
 * Assembly calling convention:
 *   a0 = float *ax
 *   a1 = float *ay
 *   a2 = float *az
 *   a3 = int    n
 */
void print_acc(float *ax, float *ay, float *az, int n)
{
    printf("--- accelerations ---\n");
    for (int i = 0; i < n; i++) {
        printf("body %3d | acc=(%+.4e, %+.4e, %+.4e)\n",
               i, ax[i], ay[i], az[i]);
    }
    printf("---\n");
}

void print_mass(float *m, int n)
{
    printf("--- masses ---\n");
    for (int i = 0; i < n; i++)
        printf("body %d | mass = %+.4e\n", i, m[i]);
    printf("---\n");
}

void* open_mmap(const char* path, int size) {
    int fd = open(path, O_RDWR);
    if (fd < 0) return NULL;
    void* addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    return addr;
}

