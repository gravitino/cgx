#include "cgx/cgx.cuh"
#include <cooperative_groups.h>

using namespace cooperative_groups;
using namespace cgx;

__global__ void kernel () {

    // get the current grid
    const auto group = this_grid();

    // alternatively you can use the current block
    // or any coalesced group within a block
    // more general: any object that exposed the member functions
    // thread_rank() and size()
    // const auto group = this_thread_block();
    // const auto group = tiled_partition<2>(this_thread_block());

    // the whole group generates scalar (1D) indices
    // between 0 and 7 (exlusive) with unit step size
    // 0, 1, 2, 3, 4, 5, 6, 7
    for (auto dim : range(group, 7))
        printf("%ld %ld %ld\n", dim.x, dim.y, dim.z);


    // the whole group generates sclar (1D) indices
    // from -3 to 4 with step size 2 (think about Python's range)
    // -3, -2, -1, 0, 1, 2, 3
    //for (auto dim : range(group, -3, 4, 2))
    //    printf("%ld %ld %ld\n", dim.x, dim.y, dim.z);

    // the whole group generates all 3D indices from the
    // Cartesian product [0,1) x [0, 2) x [0, 3) with unit step size
    //for (auto dim : range(group, dim3_t(1, 2, 3)))
    //    printf("%ld %ld %ld\n", dim.x, dim.y, dim.z);

    // the whole group generates all 3D indices from the
    // Cartesian product [-1,3) x [-2, 2) x [-3, 2) with
    // custom step sizes 1, 2, 3 for each dimension
    //for (auto dim : range(group, dim3_t(-1,-2,-3),
    //                             dim3_t( 3, 2, 1),
    //                             dim3_t( 1, 2, 3)))
    //    printf("%ld %ld %ld\n", dim.x, dim.y, dim.z);
}

int main (int argc, char * argv[]) {

    void * args[0];
    dim3 blocks (2, 1, 1);
    dim3 threads(2, 2, 1);

    // launch the kernel with any suitable configuration (1D, 2D, 3D)
    // whatever fits your needs best. index generation is accomlished
    // with the range iterator
    cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, args, 0);
    cudaDeviceSynchronize();
}
