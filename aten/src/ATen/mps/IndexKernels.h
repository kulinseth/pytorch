#pragma once

namespace at {
namespace mps {

static const char * indexing_metal_shaders = R"INDEX_METAL(
#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

constant uint32_t num_indices            [[function_constant(0)]];

struct IndexAB {
    // Allow up to 16 indices
    metal::array<constant const void *, 16>  indexArray [[ id(0) ]];
};

template<typename T>
kernel void index_select(
    constant const IndexAB  & indexAB           [[buffer(0)]],
    constant const void     * indexSizes        [[buffer(1)]],
    constant const void     * indexStrides      [[buffer(2)]],
    constant const uint3    * offsets           [[buffer(3)]],
    constant const void     * inputData         [[buffer(4)]],
    device   void           * outputData        [[buffer(5)]],
    uint thread_index [[thread_position_in_grid]]) {

    constant const int64_t * index_sizes   = (constant const int64_t *)indexSizes;
    constant const int64_t * index_strides = (constant const int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant const int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x);
    constant const T * in  = (constant const T*)((constant const char*)inputData  + offsets[thread_index].y + offset);
    *out = *in;
}

template<typename T>
kernel void index_put(
    constant const IndexAB  & indexAB           [[buffer(0)]],
    constant const void     * indexSizes        [[buffer(1)]],
    constant const void     * indexStrides      [[buffer(2)]],
    constant const uint3    * offsets           [[buffer(3)]],
    constant const void     * inputData         [[buffer(4)]],
    device   void           * outputData        [[buffer(5)]],
    uint thread_index [[thread_position_in_grid]]) {

    constant const int64_t * index_sizes   = (constant const int64_t *)indexSizes;
    constant const int64_t * index_strides = (constant const int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant const int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    constant const T * in  = (constant const T*)((constant const char*)inputData  + offsets[thread_index].y);
    *out = *in;
}

#define REGISTER_INDEX_OP(DTYPE, INDEX_OP_TYPE)                \
template                                                       \
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE)]]              \
kernel void index_ ## INDEX_OP_TYPE<DTYPE>(                    \
    constant IndexAB       & indexAB           [[buffer(0)]],  \
    constant const void    * indexSizes        [[buffer(1)]],  \
    constant const void    * indexStrides      [[buffer(2)]],  \
    constant const uint3   * offsets           [[buffer(3)]],  \
    constant const void    * inputData         [[buffer(4)]],  \
    device   void          * outputData        [[buffer(5)]],  \
    uint thread_index [[thread_position_in_grid]]);

#define REGISTER_INDEX_OP_ALL_DTYPES(INDEX_OP_TYPE) \
    REGISTER_INDEX_OP(float, INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(half,  INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(long,  INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(int,   INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(short, INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(char,  INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(uchar, INDEX_OP_TYPE);        \
    REGISTER_INDEX_OP(bool,  INDEX_OP_TYPE);

REGISTER_INDEX_OP_ALL_DTYPES(select);
REGISTER_INDEX_OP_ALL_DTYPES(put);

kernel void kernel_index_offsets(constant const packed_uint3 * strides         [[buffer(0)]],
                                 device uint3                * data_offsets    [[buffer(1)]],
                                 constant const uint         * iter_shape      [[buffer(2)]],
                                 constant const uint         & num_dimensions  [[buffer(3)]],
                                 constant const uint         & num_offsets     [[buffer(4)]],
                                 uint thread_index [[thread_position_in_grid]]) {
    device uint3 & localDataOffsets = data_offsets[thread_index];
    uint32_t idx = thread_index;
    for (uint32_t dim = 0; dim < num_dimensions; dim++) {
        uint32_t remainder = idx % iter_shape[dim];
        idx /= iter_shape[dim];

        for (uint32_t offset = 0; offset < num_offsets; offset++)
            data_offsets[thread_index][offset] += remainder * strides[dim][offset];
    }
}

template<typename T, typename E>
kernel void index_put_accumulate_native_dtypes(constant const IndexAB & indexAB      [[buffer(0)]],
                                               constant const void    * indexSizes   [[buffer(1)]],
                                               constant const void    * indexStrides [[buffer(2)]],
                                               constant const uint3   * offsets      [[buffer(3)]],
                                               constant const void    * inputData    [[buffer(4)]],
                                               device       void    * outputData   [[buffer(5)]],
                                               uint thread_index [[thread_position_in_grid]]) {
    constant const int64_t * index_sizes   = (constant const int64_t *)indexSizes;
    constant const int64_t * index_strides = (constant const int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant const int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
    }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    constant const E * in  = (constant const E*)((constant const char*)inputData  + offsets[thread_index].y);
    atomic_fetch_add_explicit(out, *in, memory_order_relaxed);
}

template<typename T>
__attribute__((__always_inline__)) void atomic_fetch_add_relaxed(device void * addr, T value) {
    device atomic_uint* uintAddr = (device atomic_uint*)addr;
    uint expected = atomic_load_explicit(uintAddr, memory_order_relaxed);
    T updated = as_type<T>(expected) + value;
    while (!atomic_compare_exchange_weak_explicit(uintAddr, &expected, as_type<uint>(updated), memory_order_relaxed, memory_order_relaxed)) {
        updated = as_type<T>(expected) + value;
    }
}

template<typename T>
kernel void atomic_index_put_accumulate(constant const IndexAB & indexAB           [[buffer(0)]],
                                        constant const void    * indexSizes        [[buffer(1)]],
                                        constant const void    * indexStrides      [[buffer(2)]],
                                        constant const uint3   * offsets           [[buffer(3)]],
                                        constant const void    * inputData         [[buffer(4)]],
                                        device         void    * outputData        [[buffer(5)]],
                                        uint thread_index [[thread_position_in_grid]]) {
    constant const const int64_t * index_sizes   = (constant const int64_t *)indexSizes;
    constant const const int64_t * index_strides = (constant const int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant const int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
    }
    device void * out = (device void*)((device char*)outputData + offsets[thread_index].x + offset);
    constant const T * in  = (constant const T*)((constant const char*)inputData  + offsets[thread_index].y);
    atomic_fetch_add_relaxed<T>(out, *in);
}

template
[[host_name("index_put_accumulate_float")]]
kernel void atomic_index_put_accumulate<float>(constant const IndexAB & indexAB      [[buffer(0)]],
                                               constant const void    * indexSizes   [[buffer(1)]],
                                               constant const void    * indexStrides [[buffer(2)]],
                                               constant const uint3   * offsets      [[buffer(3)]],
                                               constant const void    * inputData    [[buffer(4)]],
                                               device   void          * outputData   [[buffer(5)]],
                                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_accumulate_int")]]
kernel void index_put_accumulate_native_dtypes<atomic_int, int>(constant const IndexAB & indexAB      [[buffer(0)]],
                                                                constant const void    * indexSizes   [[buffer(1)]],
                                                                constant const void    * indexStrides [[buffer(2)]],
                                                                constant const uint3   * offsets      [[buffer(3)]],
                                                                constant const void    * inputData    [[buffer(4)]],
                                                                device   void          * outputData   [[buffer(5)]],
                                                                uint thread_index [[thread_position_in_grid]]);


)INDEX_METAL";

static const char * gather_scatter_metal_shaders = R"VIEW_OPS(
#include <metal_stdlib>

using namespace metal;

struct int5{
    int x;
    int y;
    int z;
    int w;
    int u;
};

__attribute__((always_inline))
uint getLinearIndex(const uint lane, const uint3 tgid, const uint3 tpg, const uint3 tgpg) {
    const uint flattened_threadgroup_in_grid = tgid.z * (tgpg.x * tgpg.y) + tgid.y * tgpg.x + tgid.x;
    return lane + (tpg.x * tpg.y * tpg.z * flattened_threadgroup_in_grid);
}

#define REGISTER_GATHER_OR_SCATTER_KERNEL(DTYPE, RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE)                       \
template                                                                                                           \
[[host_name(#GATHER_OR_SCATTER "_kernel_" #DTYPE #RANK)]]                                                          \
kernel void GATHER_OR_SCATTER ## _kernel_ ## RANK ## D <DTYPE, DTYPE_SIZE_STRIDE>(                                 \
                                 uint lane                                  [[thread_index_in_threadgroup]],       \
                                 uint3 tgid                                 [[threadgroup_position_in_grid]],      \
                                 uint3 tpg                                  [[threads_per_threadgroup]],           \
                                 uint3 tgpg                                 [[threadgroups_per_grid]],             \
                                 constant const DTYPE * src                 [[buffer(0)]],                         \
                                 device DTYPE * dst                         [[buffer(1)]],                         \
                                 constant const DTYPE_SIZE_STRIDE & size    [[buffer(2)]],                         \
                                 constant const DTYPE_SIZE_STRIDE & stride  [[buffer(3)]],                         \
                                 constant const int & numel                [[buffer(4)]])

#define REGISTER_GATHER_SCATTER_KERNEL(DTYPE, RANK, GATHER_SCATTER, DTYPE_SIZE_STRIDE)                                \
template                                                                                                              \
[[host_name("gather_scatter_kernel_" #DTYPE #RANK)]]                                                                  \
kernel void gather_scatter_kernel_ ## RANK ## D <DTYPE, DTYPE_SIZE_STRIDE>(                                           \
                                    uint lane                                       [[thread_index_in_threadgroup]],  \
                                     uint3 tgid                                     [[threadgroup_position_in_grid]], \
                                     uint3 tpg                                      [[threads_per_threadgroup]],      \
                                     uint3 tgpg                                     [[threadgroups_per_grid]],        \
                                     constant const DTYPE * src                     [[buffer(0)]],                    \
                                     device DTYPE * dst                             [[buffer(1)]],                    \
                                     constant const DTYPE_SIZE_STRIDE & dst_size    [[buffer(2)]],                    \
                                     constant const DTYPE_SIZE_STRIDE & dst_stride  [[buffer(3)]],                    \
                                     constant const DTYPE_SIZE_STRIDE & src_size    [[buffer(4)]],                    \
                                     constant const DTYPE_SIZE_STRIDE & src_stride  [[buffer(5)]],                    \
                                     constant const int & numel                     [[buffer(6)]])

template<typename T, typename U>
kernel void gather_scatter_kernel_5D(uint lane                      [[thread_index_in_threadgroup]],
                                     uint3 tgid                     [[threadgroup_position_in_grid]],
                                     uint3 tpg                      [[threads_per_threadgroup]],
                                     uint3 tgpg                     [[threadgroups_per_grid]],
                                     constant const T * src         [[buffer(0)]],
                                     device T * dst                 [[buffer(1)]],
                                     constant const U & dst_size    [[buffer(2)]],
                                     constant const U & dst_stride  [[buffer(3)]],
                                     constant const U & src_size    [[buffer(4)]],
                                     constant const U & src_stride  [[buffer(5)]],
                                     constant const int & numel    [[buffer(6)]]) {

}

template<typename T, typename U>
kernel void gather_scatter_kernel_4D(uint lane                      [[thread_index_in_threadgroup]],
                                     uint3 tgid                     [[threadgroup_position_in_grid]],
                                     uint3 tpg                      [[threads_per_threadgroup]],
                                     uint3 tgpg                     [[threadgroups_per_grid]],
                                     constant const T * src         [[buffer(0)]],
                                     device T * dst                 [[buffer(1)]],
                                     constant const U & dst_size    [[buffer(2)]],
                                     constant const U & dst_stride  [[buffer(3)]],
                                     constant const U & src_size    [[buffer(4)]],
                                     constant const U & src_stride  [[buffer(5)]],
                                     constant const int & numel    [[buffer(6)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;
    const int dst_sz = dst_size[3] * dst_size[2];
    const int src_sz = dst_size[3] * dst_size[2];

    U dst_local_index;
    dst_local_index.x = linear_index / (dst_sz * dst_size[1]) % dst_size[0];
    dst_local_index.y = linear_index / dst_sz % dst_size[1];
    dst_local_index.z = linear_index / dst_size[3] % dst_size[2];
    dst_local_index.w = linear_index % dst_size[3];

    U src_local_index;
    src_local_index.x = linear_index / (src_sz * src_size[1]) % src_size[0];
    src_local_index.y = linear_index / src_sz % src_size[1];
    src_local_index.z = linear_index / src_size[3] % src_size[2];
    src_local_index.w = linear_index % src_size[3];

    const U dst_strided_index = dst_local_index * dst_stride;
    const U src_strided_index = src_local_index * src_stride;
    dst[dst_strided_index.x + dst_strided_index.y + dst_strided_index.z + dst_strided_index.w] =
        src[src_strided_index.x + src_strided_index.y + src_strided_index.z + src_strided_index.w];
}

template<typename T, typename U>
kernel void gather_scatter_kernel_3D(uint lane                      [[thread_index_in_threadgroup]],
                                     uint3 tgid                     [[threadgroup_position_in_grid]],
                                     uint3 tpg                      [[threads_per_threadgroup]],
                                     uint3 tgpg                     [[threadgroups_per_grid]],
                                     constant const T * src         [[buffer(0)]],
                                     device T * dst                 [[buffer(1)]],
                                     constant const U & dst_size    [[buffer(2)]],
                                     constant const U & dst_stride  [[buffer(3)]],
                                     constant const U & src_size    [[buffer(4)]],
                                     constant const U & src_stride  [[buffer(5)]],
                                     constant const int & numel    [[buffer(6)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U dst_local_index;
    dst_local_index.x = linear_index / (dst_size[2] * dst_size[1]) % dst_size[0];
    dst_local_index.y = linear_index / dst_size[2] % dst_size[1];
    dst_local_index.z = linear_index % dst_size[2];

    U src_local_index;
    src_local_index.x = linear_index / (src_size[2] * src_size[1]) % src_size[0];
    src_local_index.y = linear_index / src_size[2] % src_size[1];
    src_local_index.z = linear_index % src_size[2];

    const U dst_strided_index = dst_local_index * dst_stride;
    const U src_strided_index = src_local_index * src_stride;
    dst[dst_strided_index.x + dst_strided_index.y + dst_strided_index.z] =
        src[src_strided_index.x + src_strided_index.y + src_strided_index.z];
}

template<typename T, typename U>
kernel void gather_scatter_kernel_2D(uint lane                      [[thread_index_in_threadgroup]],
                                     uint3 tgid                     [[threadgroup_position_in_grid]],
                                     uint3 tpg                      [[threads_per_threadgroup]],
                                     uint3 tgpg                     [[threadgroups_per_grid]],
                                     constant const T * src         [[buffer(0)]],
                                     device T * dst                 [[buffer(1)]],
                                     constant const U & dst_size    [[buffer(2)]],
                                     constant const U & dst_stride  [[buffer(3)]],
                                     constant const U & src_size    [[buffer(4)]],
                                     constant const U & src_stride  [[buffer(5)]],
                                     constant const int & numel    [[buffer(6)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U dst_local_index;
    dst_local_index.x = linear_index / dst_size[1] % dst_size[0];
    dst_local_index.y = linear_index % dst_size[1];

    U src_local_index;
    src_local_index.x = linear_index / src_size[1] % src_size[0];
    src_local_index.y = linear_index % src_size[1];

    const U dst_strided_index = dst_local_index * dst_stride;
    const U src_strided_index = src_local_index * src_stride;
    dst[dst_strided_index.x + dst_strided_index.y] = src[src_strided_index.x + src_strided_index.y];
}

template<typename T, typename U>
kernel void gather_scatter_kernel_1D(uint lane                      [[thread_index_in_threadgroup]],
                                     uint3 tgid                     [[threadgroup_position_in_grid]],
                                     uint3 tpg                      [[threads_per_threadgroup]],
                                     uint3 tgpg                     [[threadgroups_per_grid]],
                                     constant const T * src         [[buffer(0)]],
                                     device T * dst                 [[buffer(1)]],
                                     constant const U & dst_size    [[buffer(2)]],
                                     constant const U & dst_stride  [[buffer(3)]],
                                     constant const U & src_size    [[buffer(4)]],
                                     constant const U & src_stride  [[buffer(5)]],
                                     constant const int & numel    [[buffer(6)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    const U dst_local_index = linear_index % dst_size;
    const U dst_strided_index = dst_local_index * dst_stride;

    const U src_local_index = linear_index % src_size;
    const U src_strided_index = src_local_index * src_stride;

    dst[dst_strided_index] = src[src_strided_index];
}

template<typename T, typename U>
kernel void scatter_kernel_5D(uint lane                    [[thread_index_in_threadgroup]],
                              uint3 tgid                   [[threadgroup_position_in_grid]],
                              uint3 tpg                    [[threads_per_threadgroup]],
                              uint3 tgpg                   [[threadgroups_per_grid]],
                              constant const T * src       [[buffer(0)]],
                              device T * dst               [[buffer(1)]],
                              constant const U & size      [[buffer(2)]],
                              constant const U & stride    [[buffer(3)]],
                              constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U local_index;
    local_index.x = linear_index / size.u * size.w * size.z * size.y % size.x;
    local_index.y = linear_index / size.u * size.w * size.z % size.y;
    local_index.z = linear_index / size.u * size.w % size.z;
    local_index.w = linear_index / size.u % size.w;
    local_index.u = linear_index % size.u;

    U strided_index;
    strided_index.x = local_index.x * stride.x;
    strided_index.y = local_index.y * stride.y;
    strided_index.z = local_index.z * stride.z;
    strided_index.w = local_index.w * stride.w;
    strided_index.u = local_index.u * stride.u;

    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w + strided_index.u] = src[linear_index];
}

template<typename T, typename U>
kernel void scatter_kernel_4D(uint lane                    [[thread_index_in_threadgroup]],
                              uint3 tgid                   [[threadgroup_position_in_grid]],
                              uint3 tpg                    [[threads_per_threadgroup]],
                              uint3 tgpg                   [[threadgroups_per_grid]],
                              constant const T * src       [[buffer(0)]],
                              device T * dst               [[buffer(1)]],
                              constant const U & size      [[buffer(2)]],
                              constant const U & stride    [[buffer(3)]],
                              constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;
    const int sz = size[3] * size[2];

    U local_index;
    local_index.x = linear_index / (sz * size[1]) % size[0];
    local_index.y = linear_index / sz % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const U strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y + strided_index.z + strided_index.w] = src[linear_index];
}

template<typename T, typename U>
kernel void scatter_kernel_3D(uint lane                    [[thread_index_in_threadgroup]],
                              uint3 tgid                   [[threadgroup_position_in_grid]],
                              uint3 tpg                    [[threads_per_threadgroup]],
                              uint3 tgpg                   [[threadgroups_per_grid]],
                              constant const T * src       [[buffer(0)]],
                              device T * dst               [[buffer(1)]],
                              constant const U & size      [[buffer(2)]],
                              constant const U & stride    [[buffer(3)]],
                              constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const U strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y + strided_index.z] = src[linear_index];
}

template<typename T, typename U>
kernel void scatter_kernel_2D(uint lane                    [[thread_index_in_threadgroup]],
                              uint3 tgid                   [[threadgroup_position_in_grid]],
                              uint3 tpg                    [[threads_per_threadgroup]],
                              uint3 tgpg                   [[threadgroups_per_grid]],
                              constant const T * src       [[buffer(0)]],
                              device T * dst               [[buffer(1)]],
                              constant const U & size      [[buffer(2)]],
                              constant const U & stride    [[buffer(3)]],
                              constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const U strided_index = local_index * stride;
    dst[strided_index.x + strided_index.y] = src[linear_index];
}

template<typename T, typename U>
kernel void scatter_kernel_1D(uint lane                    [[thread_index_in_threadgroup]],
                              uint3 tgid                   [[threadgroup_position_in_grid]],
                              uint3 tpg                    [[threads_per_threadgroup]],
                              uint3 tgpg                   [[threadgroups_per_grid]],
                              constant const T * src       [[buffer(0)]],
                              device T * dst               [[buffer(1)]],
                              constant const U & size      [[buffer(2)]],
                              constant const U & stride    [[buffer(3)]],
                              constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    const U local_index = linear_index % size;
    const U strided_index = local_index * stride;
    dst[strided_index] = src[linear_index];
}

template<typename T, typename U>
kernel void gather_kernel_5D(uint lane                    [[thread_index_in_threadgroup]],
                              uint3 tgid                   [[threadgroup_position_in_grid]],
                              uint3 tpg                    [[threads_per_threadgroup]],
                              uint3 tgpg                   [[threadgroups_per_grid]],
                              constant const T * src       [[buffer(0)]],
                              device T * dst               [[buffer(1)]],
                              constant const U & size      [[buffer(2)]],
                              constant const U & stride    [[buffer(3)]],
                              constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U local_index;
    local_index.x = linear_index / size.u * size.w * size.z * size.y % size.x;
    local_index.y = linear_index / size.u * size.w * size.z % size.y;
    local_index.z = linear_index / size.u * size.w % size.z;
    local_index.w = linear_index / size.u % size.w;
    local_index.u = linear_index % size.u;

    U strided_index;
    strided_index.x = local_index.x * stride.x;
    strided_index.y = local_index.y * stride.y;
    strided_index.z = local_index.z * stride.z;
    strided_index.w = local_index.w * stride.w;
    strided_index.u = local_index.u * stride.u;

    dst[linear_index] = src[strided_index.x + strided_index.y + strided_index.z + strided_index.w + strided_index.u];
}

template<typename T, typename U>
kernel void gather_kernel_4D(uint lane                   [[thread_index_in_threadgroup]],
                             uint3 tgid                  [[threadgroup_position_in_grid]],
                             uint3 tpg                   [[threads_per_threadgroup]],
                             uint3 tgpg                  [[threadgroups_per_grid]],
                             constant const T * src      [[buffer(0)]],
                             device T * dst              [[buffer(1)]],
                             constant const U & size     [[buffer(2)]],
                             constant const U & stride   [[buffer(3)]],
                             constant const int & numel [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;
    const uint sz = size[3] * size[2];

    U local_index;
    local_index.x = linear_index / (sz * size[1]) % size[0];
    local_index.y = linear_index / sz % size[1];
    local_index.z = linear_index / size[3] % size[2];
    local_index.w = linear_index % size[3];

    const U strided_index = local_index * stride;
    dst[linear_index] = src[strided_index.x + strided_index.y + strided_index.z + strided_index.w];
}

template<typename T, typename U>
kernel void gather_kernel_3D(uint lane                    [[thread_index_in_threadgroup]],
                             uint3 tgid                   [[threadgroup_position_in_grid]],
                             uint3 tpg                    [[threads_per_threadgroup]],
                             uint3 tgpg                   [[threadgroups_per_grid]],
                             constant const T * src       [[buffer(0)]],
                             device T * dst               [[buffer(1)]],
                             constant const U & size      [[buffer(2)]],
                             constant const U & stride    [[buffer(3)]],
                             constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U local_index;
    local_index.x = linear_index / (size[2] * size[1]) % size[0];
    local_index.y = linear_index / size[2] % size[1];
    local_index.z = linear_index % size[2];

    const U strided_index = local_index * stride;
    dst[linear_index] = src[strided_index.x + strided_index.y + strided_index.z];
}

template<typename T, typename U>
kernel void gather_kernel_2D(uint lane                    [[thread_index_in_threadgroup]],
                             uint3 tgid                   [[threadgroup_position_in_grid]],
                             uint3 tpg                    [[threads_per_threadgroup]],
                             uint3 tgpg                   [[threadgroups_per_grid]],
                             constant const T * src       [[buffer(0)]],
                             device T * dst               [[buffer(1)]],
                             constant const U & size      [[buffer(2)]],
                             constant const U & stride    [[buffer(3)]],
                             constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    U local_index;
    local_index.x = linear_index / size[1] % size[0];
    local_index.y = linear_index % size[1];

    const U strided_index = local_index * stride;
    dst[linear_index] = src[strided_index.x + strided_index.y];
}

template<typename T, typename U>
kernel void gather_kernel_1D(uint lane                    [[thread_index_in_threadgroup]],
                             uint3 tgid                   [[threadgroup_position_in_grid]],
                             uint3 tpg                    [[threads_per_threadgroup]],
                             uint3 tgpg                   [[threadgroups_per_grid]],
                             constant const T * src       [[buffer(0)]],
                             device T * dst               [[buffer(1)]],
                             constant const U & size      [[buffer(2)]],
                             constant const U & stride    [[buffer(3)]],
                             constant const int & numel  [[buffer(4)]]) {
    const int linear_index = getLinearIndex(lane, tgid, tpg, tgpg);
    if (linear_index >= numel) return;

    const U local_index = linear_index % size;
    const U strided_index = local_index * stride;
    dst[linear_index] = src[strided_index];
}

#define REGISTER_GATHER_SCATTER_ALL_DTYPES(RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE, FUNC)        \
    FUNC(float, RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(half,  RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(int,   RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(long,  RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(short, RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(uchar, RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(char,  RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE);                                        \
    FUNC(bool,  RANK, GATHER_OR_SCATTER, DTYPE_SIZE_STRIDE)

#define REGISTER_GATHER_SCATTER_ALL_RANKS(GATHER_OR_SCATTER, FUNC)                                 \
    REGISTER_GATHER_SCATTER_ALL_DTYPES(5, GATHER_OR_SCATTER, int5, FUNC);                  \
    REGISTER_GATHER_SCATTER_ALL_DTYPES(4, GATHER_OR_SCATTER, packed_int4, FUNC);                  \
    REGISTER_GATHER_SCATTER_ALL_DTYPES(3, GATHER_OR_SCATTER, packed_int3, FUNC);                  \
    REGISTER_GATHER_SCATTER_ALL_DTYPES(2, GATHER_OR_SCATTER, packed_int2, FUNC);                  \
    REGISTER_GATHER_SCATTER_ALL_DTYPES(1, GATHER_OR_SCATTER, int, FUNC)

REGISTER_GATHER_SCATTER_ALL_RANKS(gather, REGISTER_GATHER_OR_SCATTER_KERNEL);
REGISTER_GATHER_SCATTER_ALL_RANKS(scatter, REGISTER_GATHER_OR_SCATTER_KERNEL);
REGISTER_GATHER_SCATTER_ALL_RANKS(gather_scatter, REGISTER_GATHER_SCATTER_KERNEL);

)VIEW_OPS";

}
}
