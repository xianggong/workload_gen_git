#ifndef __CL_UTILS_H__
#define __CL_UTILS_H__

/*!
	\file clutils.h
*/

#include <CL/cl.h>
#include "clExtensions.h"

// The cl_time type is OS specific
#ifdef _WIN32
#include <tchar.h>
#include <Windows.h>
typedef __int64 cl_time; 
#else
#include <sys/time.h>
typedef double cl_time;
#endif

cl_context cl_init_context();

void    cl_sync();
void    cl_cleanup();

cl_mem  cl_allocDevice(size_t mem_size, cl_mem_flags flags = CL_MEM_READ_WRITE);
cl_mem  cl_allocDevice_workpool(cl_context context, size_t mem_size, cl_mem_flags flags = CL_MEM_READ_WRITE);
cl_mem  cl_allocDeviceConst(size_t mem_size, void* host_ptr);
cl_mem  cl_allocDeviceConst_workpool(cl_context context, size_t mem_size, void* host_ptr);

void    cl_freeDevice(cl_mem mem);
void    cl_copyToDevice(cl_mem dst, void *src, size_t mem_size);
void    cl_copyToDevice_workpool(cl_command_queue commandQueue_workpool, cl_mem dst, void *src, size_t mem_size);
void    cl_copyToHost(void *dst, cl_mem src, size_t mem_size);
void    cl_copyToHost_workpool(cl_command_queue commandQueue_workpool, void *dst, cl_mem src, size_t mem_size);

//int     cl_errChk(const cl_int status, const char *msg, bool exitOnErr);
int     cl_errChk_sync(const cl_int status, const char *msg, bool exitOnErr);

void	cl_executeKernel(cl_kernel kernel, cl_uint work_dim, const size_t* 
			global_work_size, const size_t* local_work_size, cl_event* event);

cl_kernel* cl_precompileKernels(char* buildOptions);

pre_compiled_kernels_per_context precompile_kernels_work_pool(work_pool work_pool, char* buildOptions);

cl_program cl_compileProgram(char* kernelPath, char* compileoptions, bool verboseoptions = 0);
cl_kernel  cl_createKernel(cl_program program, const char* kernelName);

void cl_KernelTime(cl_event);
void cl_KernelTimeSync(cl_event);

// Query the current system time
void cl_getTime(cl_time* time); 

//define vector types
typedef struct{
  int x;
  int y;
} int2;

typedef struct{
  float x;
  float y;
}float2;

typedef struct{
	float x;
	float y;
	float z;
	float w;
}float4;

#define MAX_ERR_VAL 64

#define FALSE 0
#define TRUE 1

#define NUM_KERNELS 10
#define KERNEL_INIT_DET 0 
#define KERNEL_BUILD_DET 1 
#define KERNEL_SURF_DESC 2
#define KERNEL_NORM_DESC 3
#define KERNEL_NON_MAX_SUP 4
#define KERNEL_GET_ORIENT1 5
#define KERNEL_GET_ORIENT2 6
#define KERNEL_NN 7
#define KERNEL_SCAN 8
#define KERNEL_TRANSPOSE 9

#endif
