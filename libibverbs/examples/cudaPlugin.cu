#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// 宏定义
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define logd(fmt, ...) printf("[CUDA] " fmt "\n", ##__VA_ARGS__)

// 全局变量
static int g_initialized = 0;

/**
 * GPU kernel函数：触发门铃
 */
__global__ void trigger_doorbell_kernel(void *gpu_bf, void *gpu_ctrl) {
    // 触发门铃：将ctrl的值写入bf
    printf("kernel gpu_bf=%p, gpu_ctrl=%p\n", gpu_bf, gpu_ctrl);
    *((volatile uint64_t *)gpu_bf) = *(uint64_t *)gpu_ctrl;
}

/**
 * 初始化CUDA环境
 */
extern "C" int init_cuda() {
    if (g_initialized) {
        logd("CUDA already initialized");
        return 0;
    }
    
    logd("Initializing CUDA environment");
    
    // 设置CUDA设备（使用第一个可用设备）
    CUDA_CHECK(cudaSetDevice(0));
    
    // 打印GPU信息
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    logd("Found %d CUDA devices", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        logd("Device %d: %s", i, prop.name);
    }
    
    g_initialized = 1;
    logd("CUDA environment initialized successfully");
    return 0;
}

/**
 * 清理CUDA环境
 */
extern "C" int cleanup_cuda() {
    logd("Cleaning up CUDA environment");
    g_initialized = 0;
    logd("CUDA environment cleaned up successfully");
    return 0;
}

/**
 * 将主机虚拟地址转换为GPU虚拟地址
 * @param hostVA 主机虚拟地址
 * @param size 内存大小
 * @param type 内存类型：0=bf, 1=ctrl
 * @param gpuVA 输出：GPU虚拟地址
 * @return 0成功，-1失败
 */
extern "C" int ConvertHostVA2GpuVA(void *hostVA, size_t size, int type, void **gpuVA) {
    if (!g_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return -1;
    }
    
    if (!hostVA || !gpuVA) {
        fprintf(stderr, "Invalid parameters\n");
        return -1;
    }
    
    logd("Converting hostVA %p (size=%lu, type=%d) to GPU VA", hostVA, size, type);
    
    // 根据类型选择不同的标志
    auto flag = cudaHostRegisterMapped; // 默认标志
    if (type == 0) { // bf类型
        flag = cudaHostRegisterIoMemory | cudaHostRegisterMapped;
        logd("Using bf flags: cudaHostRegisterIoMemory | cudaHostRegisterMapped");
    } else if (type == 1) { // ctrl类型
        flag = cudaHostRegisterMapped;
        logd("Using ctrl flags: cudaHostRegisterMapped");
    }
    
    CUDA_CHECK(cudaHostRegister(hostVA, size, flag));
    
    // 获取GPU设备指针
    CUDA_CHECK(cudaHostGetDevicePointer(gpuVA, hostVA, 0));
    
    logd("HostVA %p -> GPU VA %p", hostVA, *gpuVA);
    return 0;
}

/**
 * 在GPU上触发门铃
 * @param gpu_bf GPU端的bf指针
 * @param gpu_ctrl GPU端的ctrl指针
 * @return 0成功，-1失败
 */
extern "C" int TriggerDoorbell(void *gpu_bf, void *gpu_ctrl) {
    if (!g_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return -1;
    }
    
    if (!gpu_bf || !gpu_ctrl) {
        fprintf(stderr, "Invalid GPU pointers\n");
        return -1;
    }
    
    logd("Triggering doorbell on GPU: bf=%p, ctrl=%p", gpu_bf, gpu_ctrl);
    
    // 启动GPU kernel
    trigger_doorbell_kernel<<<1, 1>>>(gpu_bf, gpu_ctrl);
    
    // 检查kernel执行是否成功
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    logd("GPU doorbell triggered successfully");
    return 0;
}

/**
 * 取消注册主机内存
 * @param hostVA 主机虚拟地址
 * @return 0成功，-1失败
 */
extern "C" int UnregisterHostVA(void *hostVA) {
    if (!g_initialized) {
        fprintf(stderr, "CUDA not initialized\n");
        return -1;
    }
    
    if (!hostVA) {
        fprintf(stderr, "Invalid hostVA\n");
        return -1;
    }
    
    logd("Unregistering hostVA %p", hostVA);
    CUDA_CHECK(cudaHostUnregister(hostVA));
    logd("HostVA %p unregistered successfully", hostVA);
    return 0;
}
