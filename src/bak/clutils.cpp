
/*********************************************************** 
*  OpenCL implementation of OpenSURF                       *
*  By Chris Gregg and Perhaad Mistry                       *
*  under the direction of Norm Rubin                       *
*  Contact: chg5w@virginia.edu, pmistry@ece.neu.edu        *
*  Advanced Micro Devices                                  *
*  August 2010                                             *
*                                                          *
*  Modified from OpenSURF code developed by C. Evans:      *
*  --- OpenSURF ---                                        *
*  This library is distributed under the GNU GPL. Please   *
*  contact chris.evans@irisys.co.uk for more information.  *
*                                                          *
*  C. Evans, Research Into Robust Visual Features,         *
*  MSc University of Bristol, 2008.                        *
*                                                          *
************************************************************/
/**
	\file clutils.cpp	
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#include "event_table.h"
#include "clutils.h"
#include "gettimeofday.h"

//! Global pointer to list of precompiled kernels
cl_kernel* kernel_list = NULL;

//! Global declaration of event_table class
event_table* surf_et = NULL;
             
//! Globally visible OpenCL contexts
cl_context context;        

//! Globally visible OpenCL cmd queue
cl_command_queue commandQueue;  

//! Globally visible OpenCL device
cl_device_id device;

//! Initialize OpenCl environment on one device
/*!
	Init function for one device. Looks for supported devices and creates a context
	\return returns a context initialized 
*/
cl_context cl_init_context() 
{
	cl_int status;   

	// Used to iterate through the platforms and devices, respectively
	cl_uint numPlatforms;
	cl_uint numDevices;

	// These will hold the platform and device we select (can potentially be 
	// multiple, but we're just doing one for now)
	// cl_platform_id platform = NULL;
	printf("OpenCL compatible devices:\n");

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_errChk(status, "Getting platform IDs", true);

	// Print some information about the available platforms
	cl_platform_id *platforms = NULL;
	cl_device_id * devices = NULL;
	if (numPlatforms > 0) 
	{
		// get all the platforms
		platforms = (cl_platform_id*)malloc(numPlatforms * 
			sizeof(cl_platform_id));
		if(platforms == NULL) {
			perror("malloc");
			exit(-1);
		}
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		cl_errChk(status, "Getting platform IDs", true);

		// Traverse the platforms array
		int device_index = 0;
		for(unsigned int i = 0; i < numPlatforms ; i++) 
		{
			char* pbuf = NULL;
			size_t pbufSize;
			status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 
				0, NULL, &pbufSize);
			cl_errChk(status, "Getting platform info", true);

			pbuf = (char*)malloc(pbufSize);
			if(pbuf == NULL) {
				perror("malloc");
				exit(-1);
			}

			status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 
				pbufSize, pbuf, NULL);
			cl_errChk(status, "Getting platform info", true);
			
			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);	
			cl_errChk(status, "Getting device IDs", true);	

			printf("Platform %d (%d devices):\n", i, numDevices);
			printf("\tVendor: %s\n", pbuf);

			if(numDevices == 0) {		
				printf("There are no devices for Platform %d\n",i);		
				exit(0);	
			}
			else
			{
				//! Allocate an array of devices of size "numDevices" 
				devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
				if(devices == NULL) {
					perror("malloc");
					exit(-1);
				}
				//! Populate Arrray with devices
				status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, 		
					devices, NULL);	
				cl_errChk(status, "Getting device IDs", true);					
			}

			// Print some information about each device
			for( unsigned int j = 0; j < numDevices; j++) 
			{
				size_t devInfoSize;
				char* devInfoStr = NULL;

				printf("\tDevice %d:\n", j);

				// Print the vendor
				status = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0,			
					NULL, &devInfoSize);		
				cl_errChk(status, "Getting vendor name", true);

				devInfoStr = (char*)malloc(devInfoSize);
				if(devInfoStr == NULL) {
					perror("malloc");
					exit(-1);
				}

				status = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, devInfoSize, 			
					devInfoStr, NULL);		
				cl_errChk(status, "Getting vendor name", true);
				printf("\t\tVendor: %s\n", devInfoStr);	

				free(devInfoStr);

				// Print the name
				status = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, 			
					NULL, &devInfoSize);	
				cl_errChk(status, "Getting device name", true);
				
				devInfoStr = (char*)malloc(devInfoSize);
				if(devInfoStr == NULL) {
					perror("malloc");
					exit(-1);
				}

				status = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, devInfoSize, 			
					devInfoStr, NULL);	
				cl_errChk(status, "Getting device name", true);
				printf("\t\tName: %s\n", devInfoStr);

				free(devInfoStr);
			}			
			free(devices);
		}		
	}
	else 
	{
		// If no platforms are available, we're sunk!
		printf("No OpenCL platforms found\n");
		exit(0);
	}
	
	// Hard-code in the platform/device to use, or uncomment 'scanf' 
	// to decide at runtime
	int platform_touse, device_touse;
	// UNCOMMENT the following two lines to manually select device each time
	//printf("Enter Platform and Device No (Seperated by Space) \n");
	//scanf("%d %d", &platform_touse, &device_touse);	
	platform_touse = 0; 
	device_touse = 0;
	printf("Using Platform %d, Device %d \n",platform_touse, device_touse);

	//! Recheck how many devices does our chosen platform have
	status = clGetDeviceIDs(platforms[platform_touse], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);	
	
	if(device_touse > (int)numDevices) 		
	{			
		printf("Invalid Device Number\n"); 
		exit(1);
	}
	else {
		//printf("No of devices is %d\n",numDevices);
	}
	
	//! Populate devices array with all the visible devices of our chosen platform
	devices = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
	if(devices == NULL) {
		perror("devices");
		exit(-1);
	}

	status = clGetDeviceIDs(platforms[platform_touse], 
					CL_DEVICE_TYPE_ALL, numDevices, 		
					devices, NULL);	
	cl_errChk(status,"Getting device IDs\n", true);	

	//!Check if Device requested is a CPU or a GPU
	cl_device_type dtype;
	device = devices[device_touse];
	status = clGetDeviceInfo(devices[device_touse], 
					CL_DEVICE_TYPE, 
					sizeof(dtype), 			
					(void *)&dtype,
					NULL);		
	cl_errChk(status,"Error in Getting Device Info\n", true);
	if(dtype == CL_DEVICE_TYPE_GPU) {
		printf("Running on a GPU\n");
	}
	else if (dtype == CL_DEVICE_TYPE_CPU) { 
		printf("Running on a CPU\n");
	}
	else {
		printf("Unknown device type\n");
		exit(0);
	}

	// Create the context
	cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, 
		(cl_context_properties)(platforms[platform_touse]), 0};
	
	cl_context_properties *cprops = cps;

	context = clCreateContextFromType(
					cprops, (cl_device_type)dtype, 		
					NULL, NULL, &status);		
	cl_errChk(status, "Creating context", true);
 
	commandQueue = clCreateCommandQueue(context, devices[device_touse], 
						CL_QUEUE_PROFILING_ENABLE, &status);
	cl_errChk(status, "creating command queue", true);

	return context;
}

/*!
	Release all resources that the user doesn't have access to.
*/
void  cl_cleanup()
{
	// XXX Currently not freeing platforms or devices

	if(commandQueue) {
		clReleaseCommandQueue(commandQueue);
	}
	if(context) {
		clReleaseContext(context);
	}
	for(int i = 0; i < NUM_KERNELS; i++) {
		clReleaseKernel(kernel_list[i]);
	}
	free(kernel_list);

	//surf_et->release();
}


/*!
	Wait till all pending commands in queue are finished
*/
void cl_sync() 
{
	clFinish(commandQueue);
}


//! Allocate memory on device
/*!
	\param mem_size Size of memory in bytes
	\param flags Optional cl_mem_flags
	\return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocDevice(size_t mem_size, cl_mem_flags flags)
{
	cl_mem mem;
	cl_int status;          

	/*!
		Logging information for keeping track of device memory
	*/
	static int allocationCount = 1;
	static size_t allocationSize = 0;

	allocationCount++;
	allocationSize += mem_size;
	
	mem = clCreateBuffer(context, flags, mem_size, NULL, &status);

	cl_errChk(status, "creating buffer", true);

	return mem;
}



//! Allocate memory on devices in work_pool
/*!
	\param contex_in_workpool A specified context in workpool
	\param mem_size Size of memory in bytes
	\param flags Optional cl_mem_flags
	\return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocDevice_workpool(cl_context context_workpool, size_t mem_size, cl_mem_flags flags)
{
	cl_mem mem;
	cl_int status;          

	/*!
		Logging information for keeping track of device memory
	*/
	static int allocationCount = 1;
	static size_t allocationSize = 0;

	allocationCount++;
	allocationSize += mem_size;
	
	mem = clCreateBuffer(context_workpool, flags, mem_size, NULL, &status);

	cl_errChk(status, "creating buffer", true);

	return mem;
}

//! Allocate constant memory on device
/*!
	\param mem_size Size of memory in bytes
	\param host_ptr Host pointer that contains the data
	\return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocDeviceConst(size_t mem_size, void* host_ptr)
{
	cl_mem mem;
	cl_int status;          

	mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						 mem_size, host_ptr, &status);
	cl_errChk(status, "Error creating const mem buffer", true);
	
	return mem;
}

//! Allocate constant memory on a device in work pool
/*!
	\param mem_size Size of memory in bytes
	\param host_ptr Host pointer that contains the data
	\return Returns a cl_mem object that points to device memory
*/
cl_mem cl_allocDeviceConst_workpool(cl_context context_workpool, size_t mem_size, void* host_ptr)
{
	cl_mem mem;
	cl_int status;          

	mem = clCreateBuffer(context_workpool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						 mem_size, host_ptr, &status);
	cl_errChk(status, "Error creating const mem buffer", true);
	
	return mem;
}

//! Copy a buffer to the device
/*!
	\param dst Valid device pointer
	\param src Host pointer that contains the data
	\param mem_size Size of data to copy
*/
void cl_copyToDevice(cl_mem dst, void* src, size_t mem_size)
{
	cl_int status;     
	status = clEnqueueWriteBuffer(commandQueue, dst, CL_TRUE, 0, 
		mem_size, src, 0, NULL, NULL); 
	cl_errChk(status, "Writing buffer", true);
}


//! Copy a buffer to a device in workpool
/*!
	\param cl_command_queue Command_queue in work pool
	\param dst Valid device pointer
	\param src Host pointer that contains the data
	\param mem_size Size of data to copy
*/
void cl_copyToDevice_workpool(cl_command_queue commandQueue_workpool, cl_mem dst, void* src, size_t mem_size)
{
	cl_int status;     
	status = clEnqueueWriteBuffer(commandQueue_workpool, dst, CL_TRUE, 0, 
		mem_size, src, 0, NULL, NULL); 
	cl_errChk(status, "Writing buffer", true);
}

//! Copy a buffer to the host
/*!
	\param dst Valid host pointer
	\param src Device pointer that contains the data
	\param mem_size Size of data to copy
*/
void cl_copyToHost(void* dst, cl_mem src, size_t mem_size) 
{
	cl_int status;          
	status = clEnqueueReadBuffer(commandQueue, src, CL_TRUE, 0, 
		mem_size, dst, 0, NULL, NULL);
	cl_errChk(status, "Reading buffer", true);
}


//! Copy a buffer to the host in workpool
/*!
	\param cl_command_queue Command_queue in work pool
	\param dst Valid host pointer
	\param src Device pointer that contains the data
	\param mem_size Size of data to copy
*/
void cl_copyToHost_workpool(cl_command_queue commandQueue_workpool, void* dst, cl_mem src, size_t mem_size) 
{
	cl_int status;          
	status = clEnqueueReadBuffer(commandQueue_workpool, src, CL_TRUE, 0, 
		mem_size, dst, 0, NULL, NULL);
	cl_errChk(status, "Reading buffer", true);
}

//! Release memory allocated on the device
/*!
	\param mem The device pointer to release
*/
void cl_freeDevice(cl_mem mem)
{  
	cl_int status;

	status = clReleaseMemObject(mem);
	cl_errChk(status, "Releasing mem object", true);
}

//! Enqueue and NDRange kernel on a device
/*!
	\param kernel The kernel to execute
	\param work_dim  The number of dimensions that define the thread structure
	\param global_work_size  Array of size 'work_dim' that defines the total threads in each dimension
	\param local_work_size  Array of size 'work_dim' that defines the size of each work group
	\param event  An event pointer that is populated if it is not NULL
*/
void cl_executeKernel(cl_kernel kernel, cl_uint work_dim, 
	const size_t* global_work_size, const size_t* local_work_size,
	cl_event* event) 
{

	cl_int status;

	status = clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL,
		global_work_size, local_work_size, 0, NULL, event);
	cl_errChk(status, "Executing kernel", true);
}

//! Convert source code file into cl_program
/*!
Compile Opencl source file into a cl_program. The cl_program will be made into a kernel in PrecompileKernels()

\param kernelPath  Filename of OpenCl code
\param compileoptions Compilation options
\param verbosebuild Switch to enable verbose Output
*/
cl_program cl_compileProgram(char* kernelPath, char* compileoptions, bool verbosebuild )
{
	cl_int status;          
	FILE *fp = NULL;
	char *source = NULL;
	long int size;

	printf("\t%s\n", kernelPath);

	// Determine the size of the source file
#ifdef _WIN32
	fopen_s(&fp, kernelPath, "rb");
#else
	fp = fopen(kernelPath, "rb");
#endif
	if(!fp) {
		printf("Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if(status != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	if(size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}
	rewind(fp);

	// Allocate enough space for the source code
	source = (char *)malloc(size + 1);
	if(source == NULL) {
		perror("malloc");
		exit(-1);
	}
	// fill with NULLs (just for fun)
	for (int i = 0; i < size+1; i++)  {
		source[i] = '\0';
	}

	// Read in the source code
	fread(source, 1, size, fp);
	source[size] = '\0';
	//printf("source:%s",source);

	// Create the program object
	cl_program clProgramReturn = clCreateProgramWithSource(context, 1, 
		(const char **)&source, NULL, &status);
	cl_errChk(status, "Creating program", true);

	free(source);
	fclose(fp);

	// Try to compile the program
	status = clBuildProgram(clProgramReturn, 0, NULL,compileoptions, NULL, NULL);
	if(cl_errChk(status, "Building program", false) || verbosebuild == 1) 
	{

		cl_build_status build_status;

		clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_STATUS, 
			sizeof(cl_build_status), &build_status, NULL);

		if(build_status == CL_SUCCESS && verbosebuild == 0) {
			return clProgramReturn;      
		}	

		//char *build_log;
		size_t ret_val_size;
		printf("Device: %p",device);
		clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG, 0, 
			NULL, &ret_val_size);

		char *build_log = NULL;
		build_log = (char *) malloc(ret_val_size+1);
		if(build_log == NULL) { 
			printf("Couldnt Allocate Build Log of Size %d \n",ret_val_size); 
			exit(-1);
		}

		clGetProgramBuildInfo(clProgramReturn, device, CL_PROGRAM_BUILD_LOG, 
			ret_val_size+1, build_log, NULL);

		// to be careful, terminate with \0
		// there's no information in the reference whether the string is 0 
		// terminated or not
		build_log[ret_val_size] = '\0';

		printf("Build log:\n %s...\n", build_log);
		if(build_status != CL_SUCCESS) {
			exit(1);
		}	
		else
			return clProgramReturn;	  
	}

	// print the ptx information
	// cl_printBinaries(clProgram);

	return clProgramReturn;
}

//! Create a kernel from compiled source
/*!
Create a kernel from compiled source

\param program  Compiled OpenCL program
\param kernel_name  Name of the kernel in the program
\return Returns a cl_kernel object for the specified kernel
*/
cl_kernel cl_createKernel(cl_program program, const char* kernel_name) {
	
	cl_kernel kernel;
	cl_int status;

	kernel = clCreateKernel(program, kernel_name, &status);
	cl_errChk(status, "Creating kernel", true);

	return kernel;
}

cl_program compile_program_work_pool(_work_pool_context context, char * program_path, char * compileoptions, bool verbosebuild)
{
	cl_int status;          
	FILE *fp;
	char *source;
	long int size;

	printf("Pre Compiling Function: Kernel file is: %s, device is: %s\n", program_path, context.device_name);

#ifdef _WIN32
    fopen_s(&fp, program_path, "rb");
#else
    fp = fopen(program_path, "rb");
#endif
	if(!fp) {
		printf("Could not open kernel file\n");
		exit(-1);
	}
	status = fseek(fp, 0, SEEK_END);
	if(status != 0) {
		printf("Error seeking to end of file\n");
		exit(-1);
	}
	size = ftell(fp);
	printf("size:**********************%d\n",size);
	if(size < 0) {
		printf("Error getting file position\n");
		exit(-1);
	}
	
	rewind(fp);

	source = (char *)malloc(size + 1);
	// fill with NULLs
	for (int i=0;i<size+1;i++) source[i]='\0';
	if(source == NULL) {
		printf("Error allocating space for the kernel source\n");
		exit(-1);
	}

	//fread(source, size, 1, fp);   // TODO add error checking here
	fread(source,1,size,fp);
	source[size] = '\0';
	//printf("source:%s",source);
	cl_program clProgramReturn = clCreateProgramWithSource(context.context, 1, 
	(const char **)&source, NULL, &status);
	if(cl_errChk(status, "creating program", true)) {
		       exit(1);
	}

	free(source);
	fclose(fp);

	status = clBuildProgram(clProgramReturn, 0, NULL,compileoptions, NULL, NULL);
	if(cl_errChk(status, "building program", true) || verbosebuild == 1) 
	{

		cl_build_status build_status;

		clGetProgramBuildInfo(clProgramReturn, context.device, CL_PROGRAM_BUILD_STATUS, 
			sizeof(cl_build_status), &build_status, NULL);

		if(build_status == CL_SUCCESS && verbosebuild == 0) {
			return clProgramReturn;      
		}	

		//char *build_log;
		size_t ret_val_size;
		printf("Device: %p",context.device);
		clGetProgramBuildInfo(clProgramReturn, context.device, CL_PROGRAM_BUILD_LOG, 0, 
			NULL, &ret_val_size);

		char *build_log = (char *) malloc(ret_val_size+1);
		if(build_log == NULL){ printf("Couldnt Allocate Build Log of Size %d \n",ret_val_size); exit(1);}

		clGetProgramBuildInfo(clProgramReturn, context.device, CL_PROGRAM_BUILD_LOG, 
			ret_val_size+1, build_log, NULL);

		printf("After build log call\n");
		// to be careful, terminate with \0
		// there's no information in the reference whether the string is 0 
		// terminated or not
		build_log[ret_val_size] = '\0';

		printf("Build log:\n %s...\n", build_log);
		getchar();
		if(build_status != CL_SUCCESS) {
			exit(1);
		}	
		else
			return clProgramReturn;	  
	}

	// print the ptx information
	//   cl_printBinaries(clProgram);
	//    printf("Done Compiling the Program\n");
	return clProgramReturn;
}


//! SURF specific kernel precompilation call
/*!
*/
cl_kernel* cl_precompileKernels(char* buildOptions)
{
	// XXX We are currently not freeing the program objects
	
	printf("Precompiling kernels...\n");
	
	struct timeval ustart, uend;   	
	double elapsed = 0.0f;  	
	gettimeofday(&ustart, NULL);

	//Allocating a Bunch of Events when the kernels are precompiled	
	surf_et = new event_table[1];
	surf_et->n=NUM_KERNELS;
	//! Initial Allocation of 100 Events in Vector
	surf_et->initial_allocate(100);

	kernel_list = NULL;
	/*
	kernel_list = (cl_kernel *)malloc(sizeof(cl_kernel)*NUM_KERNELS);
	if(kernel_list == NULL) {
		perror("malloc");
		exit(-1);
	}
	
	cl_program prog1 = cl_compileProgram("CLSource/det_kernel.cl", false);
	kernel_list[KERNEL_INIT_DET] = cl_createKernel(prog1, "init_det");
	kernel_list[KERNEL_BUILD_DET] = cl_createKernel(prog1, "g_build_det");

	cl_program prog2 = cl_compileProgram("CLSource/surf64Descriptor_kernel.cl", false);
	kernel_list[KERNEL_SURF_DESC] = cl_createKernel(prog2, "surf64Descriptor_kernel");

	cl_program prog3 = cl_compileProgram("CLSource/normSurf64_kernel.cl", false);  
	kernel_list[KERNEL_NORM_DESC] = cl_createKernel(prog3, "normSurf64Descriptor_kernel");

	cl_program prog4 = cl_compileProgram("CLSource/non_max_supression_kernel.cl", false);   
	kernel_list[KERNEL_NON_MAX_SUP] = cl_createKernel(prog4, "non_max_supression_kernel");

	cl_program prog5 = cl_compileProgram("CLSource/getOrientation_kernels.cl", false);    
	kernel_list[KERNEL_GET_ORIENT1] = cl_createKernel(prog5, "getOrientationStep1");
	kernel_list[KERNEL_GET_ORIENT2] = cl_createKernel(prog5, "getOrientationStep2");

	cl_program prog6 = cl_compileProgram("CLSource/nearestNeighbor_kernel.cl", false);
	kernel_list[KERNEL_NN] = cl_createKernel(prog6, "NearestNeighbor");

	cl_program prog7 = cl_compileProgram("CLSource/scan.cl", false);
	kernel_list[KERNEL_SCAN] = cl_createKernel(prog7, "scan");
	kernel_list[KERNEL_TRANSPOSE] = cl_createKernel(prog7, "transpose");
	*/
	gettimeofday(&uend, NULL);
	elapsed += uend.tv_sec * 1000 + double(uend.tv_usec)/1000;
	elapsed -= ustart.tv_sec * 1000 + double(ustart.tv_usec)/1000;  
	printf("\tTime for Off-Critical Path Compilation: %.3f milliseconds\n\n",elapsed);
	
	return kernel_list;
}

//! SURF specific kernel precompilation call
/*!
*/

pre_compiled_kernels_per_context precompile_kernels_work_pool(work_pool work_pool_1, char* buildOptions)
{
	// XXX We are currently not freeing the program objects
	
	printf("Precompiling kernels PER context...\n");
	
	struct timeval ustart, uend;   	
	double elapsed = 0.0f;  	
	gettimeofday(&ustart, NULL);

	//Allocating a Bunch of Events when the kernels are precompiled	
	surf_et = new event_table[1];
	surf_et->n=NUM_KERNELS;
	//! Initial Allocation of 100 Events in Vector
	surf_et->initial_allocate(100);

	pre_compiled_kernels_per_context pre_compiled_kernels = (pre_compiled_kernels_per_context)malloc(work_pool_1.total_num_devices * sizeof(_pre_compiled_kernels_per_context));

	for(unsigned int i=0;i<work_pool_1.total_num_devices;i++)
	{
		pre_compiled_kernels[i].context = work_pool_1.context[i].context;
		pre_compiled_kernels[i].pre_compiled_kernels = (cl_kernel *)malloc(sizeof(cl_kernel)*NUM_KERNELS);
		if(pre_compiled_kernels[i].pre_compiled_kernels == NULL) {
			perror("malloc");
			exit(-1);
		}

		cl_program prog1 = compile_program_work_pool(work_pool_1.context[i], "CLSource/det_kernel.cl", NULL, FALSE);
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_INIT_DET] = cl_createKernel(prog1, "init_det");
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_BUILD_DET] = cl_createKernel(prog1, "g_build_det");

		cl_program prog2 = compile_program_work_pool(work_pool_1.context[i], "CLSource/surf64Descriptor_kernel.cl", NULL, FALSE);
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_SURF_DESC] = cl_createKernel(prog2, "surf64Descriptor_kernel");

		cl_program prog3 = compile_program_work_pool(work_pool_1.context[i], "CLSource/normSurf64_kernel.cl", NULL, FALSE);
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_NORM_DESC] = cl_createKernel(prog3, "normSurf64Descriptor_kernel");

		cl_program prog4 = compile_program_work_pool(work_pool_1.context[i], "CLSource/non_max_supression_kernel.cl", NULL, FALSE);   
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_NON_MAX_SUP] = cl_createKernel(prog4, "non_max_supression_kernel");

		cl_program prog5 = compile_program_work_pool(work_pool_1.context[i], "CLSource/getOrientation_kernels.cl", NULL, FALSE);   
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_GET_ORIENT1] = cl_createKernel(prog5, "getOrientationStep1");
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_GET_ORIENT2] = cl_createKernel(prog5, "getOrientationStep2");

		cl_program prog6 = compile_program_work_pool(work_pool_1.context[i], "CLSource/nearestNeighbor_kernel.cl", NULL, FALSE);
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_NN] = cl_createKernel(prog6, "NearestNeighbor");

		cl_program prog7 = compile_program_work_pool(work_pool_1.context[i], "CLSource/scan.cl", NULL, FALSE);
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_SCAN] = cl_createKernel(prog7, "scan");
		pre_compiled_kernels[i].pre_compiled_kernels[KERNEL_TRANSPOSE] = cl_createKernel(prog7, "transpose");
	}
	
	gettimeofday(&uend, NULL);
	elapsed += uend.tv_sec * 1000 + double(uend.tv_usec)/1000;
	elapsed -= ustart.tv_sec * 1000 + double(ustart.tv_usec)/1000;  
	printf("\tTime for Off-Critical Path Compilation: %.3f milliseconds\n\n",elapsed);
	
	return pre_compiled_kernels;
}


//! OpenCl error code list
/*!
	An array of character strings used to give the error corresponding to the error code \n

	The error code is the index within this array
*/
/*
char *cl_errs[MAX_ERR_VAL] = {
	"CL_SUCCESS",                         // 0                            
	"CL_DEVICE_NOT_FOUND",                //-1                         
	"CL_DEVICE_NOT_AVAILABLE",            //-2                    
	"CL_COMPILER_NOT_AVAILABLE",          //-3                 
	"CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4            
	"CL_OUT_OF_RESOURCES",                //-5                         
	"CL_OUT_OF_HOST_MEMORY",              //-6                      
	"CL_PROFILING_INFO_NOT_AVAILABLE",    //-7            
	"CL_MEM_COPY_OVERLAP",                //-8                        
	"CL_IMAGE_FORMAT_MISMATCH",           //-9                   
	"CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
	"CL_BUILD_PROGRAM_FAILURE",           //-11           
	"CL_MAP_FAILURE",                     //-12
	"",                                   //-13
	"",                                   //-14
	"",                                   //-15
	"",                                   //-16
	"",                                   //-17
	"",                                   //-18
	"",                                   //-19
	"",                                   //-20
	"",                                   //-21
	"",                                   //-22
	"",                                   //-23
	"",                                   //-24
	"",                                   //-25
	"",                                   //-26
	"",                                   //-27
	"",                                   //-28
	"",                                   //-29
	"CL_INVALID_VALUE",                   //-30
	"CL_INVALID_DEVICE_TYPE",             //-31
	"CL_INVALID_PLATFORM",                //-32
	"CL_INVALID_DEVICE",                  //-33
	"CL_INVALID_CONTEXT",                 //-34
	"CL_INVALID_QUEUE_PROPERTIES",        //-35
	"CL_INVALID_COMMAND_QUEUE",           //-36
	"CL_INVALID_HOST_PTR",                //-37
	"CL_INVALID_MEM_OBJECT",              //-38
	"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
	"CL_INVALID_IMAGE_SIZE",              //-40
	"CL_INVALID_SAMPLER",                 //-41
	"CL_INVALID_BINARY",                  //-42
	"CL_INVALID_BUILD_OPTIONS",           //-43
	"CL_INVALID_PROGRAM",                 //-44
	"CL_INVALID_PROGRAM_EXECUTABLE",      //-45
	"CL_INVALID_KERNEL_NAME",             //-46
	"CL_INVALID_KERNEL_DEFINITION",       //-47
	"CL_INVALID_KERNEL",                  //-48
	"CL_INVALID_ARG_INDEX",               //-49
	"CL_INVALID_ARG_VALUE",               //-50
	"CL_INVALID_ARG_SIZE",                //-51
	"CL_INVALID_KERNEL_ARGS",             //-52
	"CL_INVALID_WORK_DIMENSION ",         //-53
	"CL_INVALID_WORK_GROUP_SIZE",         //-54
	"CL_INVALID_WORK_ITEM_SIZE",          //-55
	"CL_INVALID_GLOBAL_OFFSET",           //-56
	"CL_INVALID_EVENT_WAIT_LIST",         //-57
	"CL_INVALID_EVENT",                   //-58
	"CL_INVALID_OPERATION",               //-59
	"CL_INVALID_GL_OBJECT",               //-60
	"CL_INVALID_BUFFER_SIZE",             //-61
	"CL_INVALID_MIP_LEVEL",               //-62
	"CL_INVALID_GLOBAL_WORK_SIZE"};       //-63
*/

//! OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl
\param status Error value as cl_int
\param msg User provided error message 
\return True if Error Seen, False if no error
*/
/*
int cl_errChk(const cl_int status, const char * msg, bool exitOnErr)
{

	if(status != CL_SUCCESS) {
		printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);
		
		if(exitOnErr) {
			exit(-1);
		}
		
		return TRUE;
	}
	return FALSE;
}
*/
//! Synchronous OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl, Waits till all 
commands finish before checking error code
\param status Error value as cl_int
\param msg User provided error message 
\return True if Error Seen, False if no error
*/
/*
int cl_errChk_sync(const cl_int status, const char * msg, bool exitOnErr)
{
	cl_sync();
	if(status != CL_SUCCESS) {
		printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);

		if(exitOnErr) {
			exit(-1);
		}
		return TRUE;
	}
	return FALSE;
}
*/
//! Time kernel execution using cl_event
/*!
	Prints out the time taken between the start and end of an event
	\param event_time
*/
void cl_KernelTime(cl_event event_time)
{
	cl_int kerneltimer;
	cl_ulong starttime;
	cl_ulong endtime;

	kerneltimer = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_START,
										  sizeof(cl_ulong), &starttime, NULL);
	cl_errChk(kerneltimer, "Timing kernel failed", true);

	kerneltimer = clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_END,
										  sizeof(cl_ulong), &endtime, NULL);
	cl_errChk(kerneltimer, "Timing kernel failed", true);

	unsigned long elapsed =  (unsigned long)(endtime - starttime);
	printf("\tKernel Execution\t%ld ns\n",elapsed); 	
}

//! Synchronously Time kernel execution using cl_event
/*!
	Prints out the time taken between the start and end of an event.\n
	Adds synchronization in order to be sure that events have 
	occured otherwise profiling calls will fail \n
	
	Shouldnt be used on critical path due to the necessary flushing of the queue
	\param event_time
*/
void cl_KernelTimeSync(cl_event event_time)
{
	cl_int kerneltimer;
	clFlush(commandQueue);
	clFinish(commandQueue);

	cl_ulong starttime;
	cl_ulong endtime;

	kerneltimer = clGetEventProfilingInfo(event_time, 
		CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &starttime, NULL);
	cl_errChk(kerneltimer, "Timing kernel failed", true);

	kerneltimer = clGetEventProfilingInfo(event_time, 
		CL_PROFILING_COMMAND_END  ,
		sizeof(cl_ulong), &endtime, NULL);
	cl_errChk(kerneltimer, "Timing kernel failed", true);

	unsigned long elapsed =  (unsigned long)(endtime - starttime);
	printf("\tTime Elapsed in Kernel is %ld ns\n",elapsed); 	
}

//! Grab the current time using a system-specific timer
/*void cl_getTime(cl_time* time) 
{

#ifdef _WIN32
    int status = QueryPerformanceCounter((LARGE_INTEGER*)time);
    if(status == 0) {
        perror("QueryPerformanceCounter");
        exit(-1);
    }
#else
    // Use gettimeofday to get the current time
    struct timeval curTime;
    gettimeofday(&curTime, NULL);
    
    // Convert timeval into double
    *time = curTime.tv_sec * 1000 + (double)curTime.tv_usec/1000;
#endif
}*/