#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "clExtensions.h"

static clCreateSubDevicesEXT_fn pfn_clCreateSubDevicesEXT = NULL;

//#define VERBOSE

//Only one of below scheduling schemes can be enabled
//#define ROUND_ROBIN
#define STATIC_ABILITY
//#define DYNAMIC
//#define ONE_DEVICE




//! Function which sets status
/*!
Set status
\param status Error value as cl_int
\param num Status code
*/
void set_status(cl_int* status, cl_int num) {

	if(status != NULL) {
		*status = num;
	}
	
	return;
}

//! OpenCl error code list
/*!
	An array of character strings used to give the error corresponding to the error code \n
	The error code is the index within this array
*/

	/*char *cl_errs[MAX_ERR_VAL] = {*/
	const char *cl_errs[MAX_ERR_VAL] = {
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


//! OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl
\param status Error value as cl_int
\param msg User provided error message 
\return True if Error Seen, False if no error
*/

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

//! Synchronous OpenCl Error checker
/*!
Checks for error code as per cl_int returned by OpenCl, Waits till all 
commands finish before checking error code
\param status Error value as cl_int
\param msg User provided error message 
\return True if Error Seen, False if no error
*/

int cl_errChk_sync(const cl_int status, const char * msg, bool exitOnErr)
{
	//cl_sync();
	if(status != CL_SUCCESS) {
		printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);

		if(exitOnErr) {
			exit(-1);
		}
		return TRUE;
	}
	return FALSE;
}

//! Work Unit Constructor
/*!
Construct a work unit.
\param status, Error value
\param work_pool, Work Pool the work unit will belong to
\param prefer_context, The context the work unit prefers to running on (currently not support)
\param program_with_path, Kernel information (program with path)
\param kernel_name, Kernel information (kernel name)
\param dep, Dependency information
\param work_dim, Work demonsion
\param global_work_offset, Global Work Offset
\param flags, Flags indicating work unit status
\param kernel_list, Pre-compiled kernels for every possible contexts
\param kernel_no, The index of the compiled kernel in the pre-compiled kernel list
\param kernel_index, The index of the kernel in the program flow
*/
void work_unit::init(work_pool *work_pool,
	cl_context prefer_context=NULL,
	char* program_with_path=NULL,
	char* kernel_name=NULL,
	work_unit_dependency dep=NULL,
	cl_uint work_dim=0,
	const size_t* global_work_offset=NULL,
	const size_t* global_work_size=NULL,
	const size_t* local_work_size=NULL,
	cl_uint flags=0,
	pre_compiled_kernels_per_context kernel_list=NULL,
	cl_uint kernel_no=0,
	cl_uint kernel_index = 0,
	cl_int* status=NULL
	)
{

	this->program_with_path = program_with_path;
	this->kernel_name = kernel_name;
	//this->kernel_index = kernel_index;
	this->dependency = dep;
	this->work_dim = work_dim;
	this->global_work_offset = global_work_offset;
	this->prefer_context = prefer_context;

	this->global_work_size = global_work_size;
	this->local_work_size = local_work_size;

	this->context_all = (cl_context *)malloc(work_pool->total_num_devices * sizeof(cl_context));
	this->program_all = (cl_program *)malloc(work_pool->total_num_devices * sizeof(cl_program));
	this->kernel_all = (cl_kernel *)malloc(work_pool->total_num_devices * sizeof(cl_kernel));

	
	for(unsigned int i=0;i<work_pool->total_num_devices;i++)
	{
		if(kernel_list != NULL)
		{
			this->context_all[i] = kernel_list[i].context;
			this->kernel_all[i] = kernel_list[i].pre_compiled_kernels[kernel_no];
		}
		else
		{
			this->context_all[i] = work_pool->context[i].context;

			this->program_all[i] = compile_program(work_pool->context[i], this->program_with_path, NULL, FALSE);

			this->kernel_all[i] = create_kernel(this->program_all[i], this->kernel_name);
		}
	}
	
	this->work_unit_status = CL_WORKUNIT_INITIALIZED; 
	set_status(status, CL_SUCCESS);
}

char* work_unit::load_source(const char* file_name, int *file_size)
{
	FILE* input ;
	long size;
	char *content;

#ifdef _WIN32
	fopen_s(&input, file_name, "rb");
#else
	input = fopen(file_name, "rb");
#endif
	if(input == NULL) 
		return NULL;

	if(fseek(input, 0, SEEK_END) == -1) 
		return NULL;
	
	size = ftell(input);
	if(size == -1) 
		return NULL;
	if(fseek(input, 0, SEEK_SET) == -1) 
		return NULL;

	content = (char*) calloc(1, (size_t) size +1); 
	if(content == NULL) 
		return NULL;

	fread(content, 1, (size_t)size, input);
	if(ferror(input)) {
		free(content);
		return NULL;
	}

	fclose(input);
	content[size] = '\0';
	*file_size = size;
	return content;
}
//! Work unit compiling function
/*!
Compile the kernels for specific context
\param context, The context which the program will be compiled to
\param program_path, Program with path
\param compileoptions, Compile options
\param verbosebuild,  Display options in building
\return The compiled program
*/
cl_program work_unit::compile_program(_work_pool_context context, char * program_path, char * compileoptions, bool verbosebuild)
{
	cl_int status;          
	FILE *fp;
	char *source;
	int size;

	printf("Pre Compiling Function: Kernel file is: %s, device is: %s\n", program_path, context.device_name);

	source = this->load_source(program_path, &size);

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

//! Work unit kernel creation function
/*!
Create kernel with compiled program
\param program, Compiled program
\param kernel_name, The kernel name
\return Kernel
*/
cl_kernel work_unit::create_kernel(cl_program program, const char* kernel_name) {
	
	cl_kernel kernel;
	cl_int status;

	printf("Creating kernel: Kernel name is: %s\n", kernel_name);
	kernel = clCreateKernel(program, kernel_name, &status);
	cl_errChk(status, "Creating kernel", true);

	return kernel;
}

//! Work unit set arguments function
/*!
Set arguments of the work unit
\param program, Compiled program
\param kernel_name, The kernel name
\return Kernel
*/
void work_unit::set_argument(cl_int index, cl_int type, cl_int int_value, float float_value, void * data, cl_int data_size, cl_int flag, cl_int * status) 
{
	printf(">>>>>>within unit set argument\n");

	work_unit_arg arg_tmp = (work_unit_arg)malloc(sizeof(_work_unit_arg));
	arg_tmp->index = index;
	arg_tmp->value_int = int_value;
	arg_tmp->value_float = float_value;
	arg_tmp->arg_pointer = data;
	arg_tmp->size = data_size;
	arg_tmp->read_write_flag = flag;

	if(FLOAT_ARRAY_TYPE == type)
	{
		printf(">>>>>>set arg of a floating point data array\n");
		arg_tmp->type = FLOAT_ARRAY_TYPE;
		this->arguments.push_back(arg_tmp);
		set_status(status, CL_SUCCESS);
	}
	else if (INT_TYPE == type)
	{
		printf(">>>>>>set arg of a int value\n");
		arg_tmp->type = INT_TYPE;
		this->arguments.push_back(arg_tmp);
		set_status(status, CL_SUCCESS);
	}
	else
	{
		printf("coming ....\n");
	}
}

//! Grab the current time using a system-specific timer
void cl_getTime(cl_time* time) 
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
}

//! Compute the elapsed time between two timer values
double cl_computeTime(cl_time start, cl_time end) 
{
#ifdef _WIN32
	__int64 freq;
	int status;

	status = QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	if(status == 0) {
		perror("QueryPerformanceFrequency");
		exit(-1);
	}

    // Return time in ms
	return double(end-start)/(double(freq)/1000.0);
#else

	return end-start;
#endif
}

void * pthread_scheduler(void* work_pool_scheduler_arg)
{
	scheduler_thread_data *data_from_workpool;
	//data_from_workpool = (scheduler_thread_data *)malloc(sizeof(scheduler_thread_data));
	//memcpy(data_from_workpool, work_pool_scheduler_arg, sizeof(scheduler_thread_data));
	data_from_workpool = (scheduler_thread_data *)work_pool_scheduler_arg;
	printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^no of work units: %d\n\n", data_from_workpool->work_pool_in->num_work_units);

	data_from_workpool->work_pool_in->work_pool_scheduler(data_from_workpool->thread_id);

	//pthread_exit((void *)work_pool_scheduler_arg);
	data_from_workpool->work_pool_in->thread_exit[data_from_workpool->thread_id] = 1;
	//if(data_from_workpool->thread_id != data_from_workpool->work_pool_in->total_num_devices-1)
	pthread_exit((void *)pthread_scheduler);
	//ExitThread(10);
	return NULL;
}



//! Work Pool Scheduler Function
/*!
Reset the buffer used in one frame, and release the buffer
\param thread_id, The thread id which calls this function (for debugging)
*/
void work_pool::work_pool_scheduler(int device_id)
{
	cl_int status;
	cl_uint total_index;
	while(1)
	{		
		if(this->done == 1)
			break;
		else
		{
			total_index = this->query();
#ifdef ROUND_ROBIN 			
			if((total_index-1) % (this->total_num_devices) == device_id)
			{
				printf("###### [Scheduler]: I'm taking unit no.%d, and give it to device: %d\n", total_index, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
					NULL,
					NULL,
					NULL,
					NULL,
					&status);
				//if(device_id == 3)
				clFinish(this->context[device_id].command_queue);

				this->num_on_this_device[device_id]++;
			}
#elif defined(ONE_DEVICE)
			if(device_id == 0)
			{
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
					NULL,
					NULL,
					NULL,
					NULL,
					&status);
				//if(this->num_on_this_device[device_id] == total_unfinished_work_units-1)
				clFinish(this->context[device_id].command_queue);

				this->num_on_this_device[device_id]++;

				//if(this->num_on_this_device[device_id] == (total_unfinished_work_units * 10)/16)
				//	break;
			}

#elif defined(STATIC_ABILITY)
			if(device_id == 0)
			{
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
					NULL,
					NULL,
					NULL,
					NULL,
					&status);

				clFinish(this->context[device_id].command_queue);

				this->num_on_this_device[device_id]++;

				if(this->num_on_this_device[device_id] == (total_unfinished_work_units * 8)/16)
					break;
			}
			else if (device_id == 1)
			{
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
					NULL,
					NULL,
					NULL,
					NULL,
					&status);
				
				clFinish(this->context[device_id].command_queue);

				this->num_on_this_device[device_id]++;

				if(this->num_on_this_device[device_id] == (total_unfinished_work_units * 8)/16)
					break;
			}
			/*else if (device_id == 2)
			{
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
												NULL,
												NULL,
												NULL,
												NULL,
												&status);
				clFinish(this->context[device_id].command_queue);

				this->num_on_this_device[device_id]++;

				if(this->num_on_this_device[device_id] == (total_unfinished_work_units * 1)/16)
					break;
			}*/
			/*else if ((this->index_out == 21) && (device_id == 3))
			{
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
												NULL,
												NULL,
												NULL,
												NULL,
												&status);
				this->num_on_this_device[device_id]++;
			}*/
#elif defined(DYNAMIC)
				if(device_id == 0)
				{
					int init_scheduled_num = (total_unfinished_work_units * 10)/16;

					printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
					this->extract_and_distribute(this->context[device_id], 												      
						NULL,
						NULL,
						NULL,
						NULL,
						&status);

					clFinish(this->context[device_id].command_queue);

					cl_getTime(&this->unit_end_time[total_index]);
					this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]]=cl_computeTime(unit_start_time[total_index], unit_end_time[total_index]);
				//printf("!!!!!! execution time of work unit %d: %f on device %d\n", total_index, cl_computeTime(unit_start_time[total_index], unit_end_time[total_index]), device_id);

					this->num_on_this_device[device_id]++;


				//Dynamically moved work units across devices
					if(this->num_on_this_device[device_id] >= 2)
					{
						if ((this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]] > this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-1]) && (this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-1] > this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-2]))
						{
							this->busy_direction[device_id] = UP;
						}
					}

					if(this->busy_direction[device_id] == UP)
					{
						for(int i=0;i<this->total_num_devices;i++)
						{
							if(i != device_id && (this->busy_direction[i] != UP))
							{
								this->dynamic_offset[device_id] --;
								this->dynamic_offset[i] ++;
								break;
							}
						}
					}

					if(this->num_on_this_device[device_id] == init_scheduled_num + this->dynamic_offset[device_id])
						break;
				}
				else if (device_id == 1)
				{
					int init_scheduled_num = (total_unfinished_work_units * 5)/16;

					printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
					this->extract_and_distribute(this->context[device_id], 												      
						NULL,
						NULL,
						NULL,
						NULL,
						&status);

					clFinish(this->context[device_id].command_queue);

					cl_getTime(&this->unit_end_time[total_index]);
					this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]]=cl_computeTime(unit_start_time[total_index], unit_end_time[total_index]);
				//printf("!!!!!! execution time of work unit %d: %f on device %d\n", total_index, cl_computeTime(unit_start_time[total_index], unit_end_time[total_index]), device_id);

					this->num_on_this_device[device_id]++;

				//Dynamically moved work units across devices
					if(this->num_on_this_device[device_id] >= 2)
					{
						if ((this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]] > this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-1]) && (this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-1] > this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-2]))
						{
							this->busy_direction[device_id] = UP;
						}
					}

					if(this->busy_direction[device_id] == UP)
					{
						for(int i=0;i<this->total_num_devices;i++)
						{
							if(i != device_id && (this->busy_direction[i] != UP))
							{
								this->dynamic_offset[device_id] --;
								this->dynamic_offset[i] ++;
								break;
							}
						}
					}

					if(this->num_on_this_device[device_id] == init_scheduled_num + this->dynamic_offset[device_id])
						break;
				}
				else if (device_id == 2)
				{
					int init_scheduled_num = (total_unfinished_work_units * 1)/16;

					printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
					this->extract_and_distribute(this->context[device_id], 												      
						NULL,
						NULL,
						NULL,
						NULL,
						&status);
					clFinish(this->context[device_id].command_queue);

					cl_getTime(&this->unit_end_time[total_index]);
					this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]]=cl_computeTime(unit_start_time[total_index], unit_end_time[total_index]);
				//printf("!!!!!! execution time of work unit %d: %f on device %d\n", total_index, cl_computeTime(unit_start_time[total_index], unit_end_time[total_index]), device_id);

					this->num_on_this_device[device_id]++;

				//Dynamically moved work units across devices
					if(this->num_on_this_device[device_id] >= 2)
					{
						if ((this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]] > this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-1]) && (this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-1] > this->execution_time_queue_per_device[device_id][this->num_on_this_device[device_id]-2]))
						{
							this->busy_direction[device_id] = UP;
						}
					}

					if(this->busy_direction[device_id] == UP)
					{
						for(int i=0;i<this->total_num_devices;i++)
						{
							if(i != device_id && (this->busy_direction[i] != UP))
							{
								this->dynamic_offset[device_id] --;
								this->dynamic_offset[i] ++;
								break;
							}
						}
					}

					if(this->num_on_this_device[device_id] == init_scheduled_num + this->dynamic_offset[device_id])
						break;
				}
			/*else if ((this->index_out == 21) && (device_id == 3))
			{
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
												NULL,
												NULL,
												NULL,
												NULL,
												&status);
				this->num_on_this_device[device_id]++;
			}*/
#else
				printf("########### [Scheduler]: I'm taking unit %d, and give it to device: %d\n", this->index_out, device_id);
				this->extract_and_distribute(this->context[device_id], 												      
					NULL,
					NULL,
					NULL,
					NULL,
					&status);
				clFinish(this->context[device_id].command_queue);

				this->num_on_this_device[device_id]++;
#endif
			}

		}
	}

//! Work Pool Constructor
/*!
Construct a work pool.
\param status, Error value
\param max_size, The max capacity of the work pool
*/
work_pool::work_pool( )
{

}


//! Work Pool Constructor
/*!
Construct a work pool.
\param status, Error value
\param max_size, The max capacity of the work pool
*/
void work_pool::init(int max_size, unsigned int init_number_work_units, cl_int* status)
{
	//cl_int local_status;
	
	this->done = 0;
	
	this->context = work_pool_get_contexts();

	for(unsigned int i = 0; i < this->total_num_devices ; i++) 
	{
		printf("Context No.%d \n\tfor device [ %s ] \n\tfrom vendor [ %s ]\n", i, context[i].device_name, context[i].device_vendor);
	}	
	
	this->max_size = max_size;

	work_pool_start = (work_unit **)malloc(sizeof(work_unit *)*(max_size+1));
	if(work_pool_start == NULL) {
		work_pool_state = WORK_POOL_FAIL;
		set_status(status, -1);
		return;
	}

	this->index_end = 0;
	this->index_in = 0;
	this->index_out = 0;

	this->work_unit_index = 0;


	num_work_units = 0;
	work_pool_state = WORK_POOL_INIT;
	this->num_sleeping_devices = 0;

	this->num_on_this_device = (unsigned int*)malloc(sizeof(int)*total_num_devices);
	this->thread_exit = (unsigned int*)malloc(sizeof(int)*total_num_devices);

	for(int i=0;i<this->total_num_devices;i++)
	{
		this->num_on_this_device[i] = 0;
		this->thread_exit[i] = 0;
	}

	this->total_unfinished_work_units = init_number_work_units;

	this->unit_start_time = (cl_time *)malloc(sizeof(cl_time)*init_number_work_units);
	this->unit_end_time = (cl_time *)malloc(sizeof(cl_time)*init_number_work_units);

	for(int i=0;i<this->total_unfinished_work_units;i++)
	{
		this->unit_start_time[i] = 0;
		this->unit_end_time[i] = 0;
	}

	this->execution_time_queue_per_device = (double **)malloc(sizeof(double *)*this->total_num_devices);

	for(unsigned int i = 0; i < this->total_num_devices ; i++) 
	{
		this->execution_time_queue_per_device[i] = (double *)malloc(sizeof(double)*(init_number_work_units/2));
	}

	this->busy_direction = (unsigned int *)malloc(sizeof(int)*this->total_num_devices);
	this->dynamic_offset = (int *)malloc(sizeof(int)*this->total_num_devices);

	for(unsigned int i = 0; i < this->total_num_devices ; i++) 
	{
		this->busy_direction[i] = STAY;
		this->dynamic_offset[i] = 0;
	}

	init_buffer_table(this->buffer_table);

	printf("Scheduler is constantly running in background\n");
	
	pthread_mutex_init(&this->work_unit_q_mutex, NULL);
	pthread_cond_init (&this->work_unit_q_not_empty_cv, NULL);
	pthread_cond_init (&this->work_unit_q_full_cv, NULL);

	//pthread_attr_init(&this->work_pool_thread_attr);
	//pthread_attr_setdetachstate(&this->work_pool_thread_attr, PTHREAD_CREATE_JOINABLE);

	this->work_pool_scheduler_thread = (pthread_t *)malloc(sizeof(pthread_t)*this->total_num_devices);
	
	
	for(int i=0;i<this->total_num_devices;i++)
	{

		scheduler_thread_data *data_from_workpool = (scheduler_thread_data *)malloc(sizeof(scheduler_thread_data));
		data_from_workpool->work_pool_in = this;
		data_from_workpool->thread_id = i;

		//printf("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^no of work units: %d\n\n", data_from_workpool->work_pool_in->num_work_units);		

		int rc_workpool;
		//pthread win32 has problem in joinable threads
		//rc_workpool = pthread_create(&this->work_pool_scheduler_thread[i], &this->work_pool_thread_attr, pthread_scheduler, (void *)data_from_workpool);
		rc_workpool = pthread_create(&this->work_pool_scheduler_thread[i], NULL, pthread_scheduler, (void *)data_from_workpool);
		if (rc_workpool) {
			printf("ERROR; return code from creating work pool scheduler thread is %d\n", rc_workpool);
			exit(-1);
		}

		
	}
	set_status(status, CL_SUCCESS);

}

//! Get contexts information for all possible devices on the platform
/*!
Get contexts information for all possible devices on the platform
\return A pointer to the contexts created
*/
work_pool_context work_pool::work_pool_get_contexts()
{
	//printf("in work_pool_get_contexts\n");

	cl_int status; 

	total_num_devices = 0;
	cl_uint device_idx = 0;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	printf("Number of platforms detected:%d\n", numPlatforms);

	if (numPlatforms > 0) 
	{
	    //this->context = (work_pool_context)malloc(numPlatforms * sizeof(_work_pool_context));

		platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);

		for(unsigned int i = 0; i < numPlatforms ; i++) 
		{
			cl_uint numDevices;
			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
			if(cl_errChk(status, "checking for devices", true)) 					
				exit(1);										
			if(numDevices == 0) {		
				printf("There are no devices for Platform %d\n",i);		
				exit(0);
			}
			else
			{
				total_num_devices = total_num_devices + numDevices;
			}
		}

		this->context = (work_pool_context)malloc(total_num_devices * sizeof(_work_pool_context));

		for(unsigned int i = 0; i < numPlatforms ; i++) 
		{			
			cl_uint numDevices;
			status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
			if(cl_errChk(status, "checking for devices", true)) 					
				exit(1);										
			if(numDevices == 0) {		
				printf("There are no devices for Platform %d\n",i);		
				exit(0);
			}
			else
			{
				printf("\tNo of devices for Platform %d is %u\n",i, numDevices);	
				//! Allocate an array of devices of size "numDevices" 
				devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
				//! Populate Arrray with devices
				status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, 		
					devices, NULL);	
				if(cl_errChk(status, "getting device IDs", true)) {					
					exit(1);	
				}
			}

			for( unsigned int j = 0; j < numDevices; j++) 
			{
				this->context[device_idx].platform = platforms[i];
				this->context[device_idx].device = devices[j];

				printf("Context index: %d:\n", device_idx);

				printf("\tPlatform %d:\t", i);
				status = clGetPlatformInfo(context[device_idx].platform, CL_PLATFORM_VENDOR, sizeof(context[device_idx].platform_vendor), context[device_idx].platform_vendor, NULL); 
				printf("Vendor: %s\n", context[device_idx].platform_vendor);

				printf("\tDevice: %d\t", j);		
				status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_VENDOR, sizeof(context[device_idx].device_vendor), context[device_idx].device_vendor, NULL);
				cl_errChk(status, "Getting Device Vendor Info\n", true);
				printf("Vendor: %s\n", context[device_idx].device_vendor);
				status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_NAME, sizeof(context[device_idx].device_name), context[device_idx].device_name, NULL);		
				cl_errChk(status, "Getting Device Name\n", true);
				printf("\t\t\tName: %s\n", context[device_idx].device_name);	
				status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(context[device_idx].device_max_compute_units), (void *)&context[device_idx].device_max_compute_units, NULL);		
				cl_errChk(status, "Getting Device Number of Units\n", true);
				printf("\t\t\tMax Compute Units: %d\n", context[device_idx].device_max_compute_units);	
				status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(context[device_idx].device_max_frequency), (void *)&context[device_idx].device_max_frequency, NULL);		
				cl_errChk(status, "Getting Device Clock Frequency\n", true);
				printf("\t\t\tMax Clock Frequency: %d\n", context[device_idx].device_max_frequency);	


				status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_TYPE, sizeof(context[device_idx].dtype), (void *)&(context[device_idx].dtype), NULL);	
				if(cl_errChk(status,"Error in Getting Device Info\n", true)) 
					exit(1);
				/*if(context[device_idx].dtype == CL_DEVICE_TYPE_GPU) 
					printf("Creating GPU Context\n");
				else if (context[device_idx].dtype == CL_DEVICE_TYPE_CPU) 
					printf("Creating CPU Context\n");
				else 
					perror("This Context Type Not Supported\n"); 
				*/

				if(context[device_idx].dtype == CL_DEVICE_TYPE_GPU) 
				{
					printf("Creating GPU Context\n");
					cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(context[device_idx].platform), 0};
					cl_context_properties *cprops = cps;
					context[device_idx].context = clCreateContextFromType(cprops, (cl_device_type)(context[device_idx].dtype), NULL, NULL, &status);		
					if(cl_errChk(status, "creating Context", true))
						exit(1);

					context[device_idx].command_queue = clCreateCommandQueue(context[device_idx].context, context[device_idx].device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
					if(cl_errChk(status, "creating command queue", true))
						exit(1);

					context[device_idx].work_pool_context_idx = device_idx;
					device_idx++;
				}
				else if (context[device_idx].dtype == CL_DEVICE_TYPE_CPU) 
				{
					printf("Creating sub-CPU Context\n");

					char* deviceExtensions = NULL;;
					size_t extStringSize = 0;

					// Get device extensions 
					status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_EXTENSIONS, 0, deviceExtensions,  &extStringSize);	
					if(cl_errChk(status, "checking extensions", true))
						exit(1);
					
					deviceExtensions = new char[extStringSize];
					if(NULL == deviceExtensions)
					{
						printf("Failed when allocating string for checking extensions\n");
						exit(1);
					}

					status = clGetDeviceInfo(context[device_idx].device, CL_DEVICE_EXTENSIONS, extStringSize, deviceExtensions,  NULL);	
					if(cl_errChk(status, "checking extensions", true))
						exit(1);

					//printf("extension: %s\n", deviceExtensions);

					if(!strstr(deviceExtensions, "cl_ext_device_fission"))
					{
						printf("This CPU device doesn't support fission!\n");
						exit(1);
					}

					delete deviceExtensions;

					// Initialize required partition property
					cl_device_partition_property_ext partitionPrty[] =
					{ CL_DEVICE_PARTITION_BY_COUNTS_EXT,
						2, CL_PARTITION_BY_COUNTS_LIST_END_EXT,
						CL_PROPERTIES_LIST_END_EXT 
					};

					// Initialize clCreateSubDevicesEXT function pointer
					INIT_CL_EXT_FCN_PTR(this->context[device_idx].platform, clCreateSubDevicesEXT);

					cl_uint numSubDevices = 0;
					// Get number of sub-devices
					status = pfn_clCreateSubDevicesEXT(context[device_idx].device, partitionPrty, 0, NULL, &numSubDevices);

					if(cl_errChk(status, "checking number of sub devices in fission extensions", true))
						exit(1);

					printf("No of subdevice: %d\n", numSubDevices);

					cl_device_id *subDevices= (cl_device_id*)malloc(numSubDevices * sizeof(cl_device_id));
					if(NULL == subDevices)
					{
						printf("Failed to allocate memory(subDevices)\n");
						exit(1);
					}

					status = pfn_clCreateSubDevicesEXT(context[device_idx].device, partitionPrty, numSubDevices,subDevices, NULL);
					if(cl_errChk(status, "Creating sub devices using fission extensions", true))
						exit(1);

					/* Create sub device using OpenCL Fission, updated by Enqiang 07/25/2011*/
					/*
					cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(context[device_idx].platform), 0};
					cl_context_properties *cprops = cps;
					context[device_idx].context = clCreateContextFromType(cprops, (cl_device_type)(context[device_idx].dtype), NULL, NULL, &status);		
					if(cl_errChk(status, "creating Context", true))
						exit(1);

					context[device_idx].command_queue = clCreateCommandQueue(context[device_idx].context, context[device_idx].device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
					if(cl_errChk(status, "creating command queue", true))
						exit(1);
					*/

					cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(context[device_idx].platform), 0};
					cl_context_properties *cprops = cps;
					context[device_idx].context = clCreateContext(cprops, numSubDevices, subDevices, NULL, NULL, &status);	
					if(cl_errChk(status, "creating Context", true))
						exit(1);
					context[device_idx].command_queue = clCreateCommandQueue(context[device_idx].context, subDevices[0], CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
					if(cl_errChk(status, "creating command queue", true))
						exit(1);

					context[device_idx].work_pool_context_idx = device_idx;
					device_idx++;
				}
				else
					perror("This Context Type Not Supported\n");

			}						
		}
	}

	return context;
}

//! Copy work unit information
/*!
Copy work unit information
\param work_unit_from, The work unit which the information is copied FROM
\param work_unit_to, The work unit which the information is copied TO
\return
*/
void work_pool::work_units_copy(work_unit* work_unit_from, work_unit* work_unit_to)
{
	//if(work_unit_to->context_all == NULL)
		//work_unit_to->context_all = (cl_context *)malloc(sizeof(cl_context)*this->total_num_devices);
	//memcpy((void *)work_unit_to->context_all, (void *)work_unit_from->context_all, sizeof(cl_context)*this->total_num_devices);
	work_unit_to->context = work_unit_from->context;
	work_unit_to->dependency = work_unit_from->dependency;
	work_unit_to->flags = work_unit_from->flags;
	work_unit_to->global_work_offset = work_unit_from->global_work_offset;
	work_unit_to->global_work_size = work_unit_from->global_work_size;
	work_unit_to->kernel = work_unit_from->kernel;
	work_unit_to->kernel_name= work_unit_from->kernel_name;
	work_unit_to->local_work_size = work_unit_from->local_work_size;
	work_unit_to->program = work_unit_from->program;
	work_unit_to->program_with_path = work_unit_from->program_with_path;
	work_unit_to->work_dim = work_unit_from->work_dim;
	work_unit_to->work_unit_status = work_unit_from->work_unit_status;
	//work_unit_to->kernel_index = work_unit_from->kernel_index;

	return;
}

//! Enqueue work unit to work pool
/*!
Enqueue work unit to work pool
\param work_unit_in, The work unit which is enqueued
\return status, Operation status
*/
void work_pool::enqueue(work_unit* work_unit_in, cl_uint priority, cl_int* status)
{
	//printf("[Enqueue]: In the work_pool_enqueue\n");
	//cl_uint work_unit_total_index = this->query();
	//cl_getTime(&this->unit_start_time[work_unit_total_index]);
	pthread_mutex_lock (&this->work_unit_q_mutex);
#ifdef VERBOSE	
	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
	printf("@@@@@@ [Enqueue]:status at beginning of enqueue, num_work_units: %d, index_in: %d\n", this->num_work_units, this->index_in);
#endif
	
	//pre-check the priority
	if( priority > PRIORITY_LEVEL || priority < 0)
	{
#ifdef VERBOSE		
		printf("@@@@@@ [Enqueue]: Priority is out of range, set to lowest \n");
#endif
		priority = PRIORITY_LEVEL;
	}
	this->work_unit_index ++;

	if(this->num_work_units == WORKPOOL_CAP)
	{
#ifdef VERBOSE		
		printf("@@@@@@ [Enqueue]: Queue is full, wait for some space \n");
#endif
		//printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
		pthread_cond_wait(&this->work_unit_q_full_cv, &this->work_unit_q_mutex);
	}
	
	if(this->num_work_units == 0)
	{
#ifdef VERBOSE		
		//printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
		printf("@@@@@@ [Enqueue]: Queue is empty, num_work_units: %d,  index_in: %d\n", this->num_work_units, this->index_in);
#endif
		if(this->index_in != this->index_out)
		{
			printf("error happened\n");
			exit(1);
		}
		
		if(this->work_pool_start[this->index_in] == (work_unit *)0xcdcdcdcd)
		{
			this->work_pool_start[this->index_in] = (work_unit *)malloc(sizeof(work_unit));
		}
		//this->work_units_copy(work_unit_in, this->work_pool_start[this->index_in]);
		memcpy(this->work_pool_start[this->index_in], work_unit_in, sizeof(work_unit));
		this->work_pool_start[this->index_in]->priority = priority;
		this->work_pool_start[this->index_in]->unit_index = this->work_unit_index;
		

		//copy the aug vector; this can not be done by memcpy

		//You have to include the work_unit class to the work_pool class so that you can only use the pointer
		//But the work_unit will not be independent: same work unit can not be enqueued to different work pool
		//this->work_pool_start[this->index_in] = work_unit_in;

		if(this->index_in == max_size - 1)
			this->index_in = 0;
		else
			this->index_in++;

		this->num_work_units++;
		
		//printf("[Enqueue]: We have one unit in the pool\n");
		work_pool_state = WORK_POOL_NONEMPTY;

		while(this->num_sleeping_devices != 0)
		{
			pthread_cond_signal(&this->work_unit_q_not_empty_cv);
			this->num_sleeping_devices--;
			//If the number of work units is less than number of active devices, break
			//if(this->num_work_units <= this->total_num_devices - this->num_sleeping_devices)
				//break;
		}
		
	}
	
	else if (this->num_work_units < max_size)
	{
#ifdef VERBOSE		
		//printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
		printf("@@@@@@ [Enqueue]: Queue is not empty or full, num_work_units: %d,  index_in: %d\n", this->num_work_units, this->index_in);
#endif
		for(unsigned int enqueue_slot=this->num_work_units; enqueue_slot< max_size; enqueue_slot++)
		{
			if(this->work_pool_start[this->index_in] == (work_unit *)0xcdcdcdcd)
			{
				this->work_pool_start[this->index_in] = (work_unit *)malloc(sizeof(work_unit));
				break;
			}
			else if(this->work_pool_start[this->index_in]->work_unit_status == CL_WORKUNIT_WAITING)
			{
				if(this->index_in == max_size - 1)
					this->index_in = 0;
				else
					this->index_in++;
			}
			else if(this->work_pool_start[this->index_in]->work_unit_status == CL_WORKUNIT_INITIALIZED)
			{
				if(this->index_in == max_size - 1)
					this->index_in = 0;
				else
					this->index_in++;
			}
			else if(this->work_pool_start[this->index_in]->work_unit_status == CL_WORKUNIT_COMPLETE)
			{
				break;
			}
			else
			{
				printf("Invalid work unit status, error!\n");
				exit(-1);
			}	
		}
		
		memcpy(this->work_pool_start[this->index_in], work_unit_in, sizeof(work_unit));
		this->work_pool_start[this->index_in]->priority = priority;
		this->work_pool_start[this->index_in]->unit_index = this->work_unit_index;
		//std::copy(work_unit_in->arguments.begin(), work_unit_in->arguments.end(), this->work_pool_start[this->index_in]->arguments.begin()); 
		//this->work_pool_start[this->index_in]->arguments = work_unit_in->arguments;
		//this->work_pool_start[this->index_in] = work_unit_in;
#ifdef VERBOSE
		printf("@@@@@@ [Enqueue]: Finish Enqueue operation\n");
#endif

		if(this->index_in == max_size - 1)
			this->index_in = 0;
		else
			this->index_in++;
		
		this->num_work_units++;

		if(this->num_work_units == max_size)
		{
			//printf("[Enqueue]: Enqueue one unit successfully, and pool is full\n");
			work_pool_state = WORK_POOL_FULL;
		}
		else
		{
			//printf("[Enqueue]: Enqueue one unit successfully\n");
			work_pool_state = WORK_POOL_NONEMPTY;
		}
	}
	else if(this->num_work_units > max_size)
	{
#ifdef VERBOSE
		printf("@@@@@@ [Enqueue]: Enqueue failed, queue is already full. Something is wrong.\n");
#endif
		work_pool_state = WORK_POOL_FAIL;
		exit(-1);
	}
	
	//work_pool_state = WORK_POOL_NONEMPTY;
#ifdef PRINT_PROFILING	    
	printf("[In Enqueue] index_out: %d, index_in: %d\n", index_out, index_in);
	printf("[In Enqueue] work units status: \n");
	for(int i=0;i<max_size;i++)
	{
		if(this->work_pool_start[i] != (work_unit *)0xcdcdcdcd)
		{
			printf("%d ", this->work_pool_start[i]->work_unit_status);
		}
		else
		{
			break;
		}
	}
	printf(" \n\n");
#endif    

	//if(this->num_work_units == 1)
	//pthread_cond_signal(&this->work_unit_q_not_empty_cv);

#ifdef VERBOSE
	printf("@@@@@@ [Enqueue]: Done processing work unit no.%d.\n", this->work_unit_index);
	printf("@@@@@@ [Enqueue]: after enqueue, num_work_units: %d, index_in: %d\n", this->num_work_units, this->index_in);
	printf("@@@@@@ [Enqueue]: Release the lock.\n");
	printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
#endif

	//cl_uint work_unit_total_index = this->query();
	//cl_getTime(&this->unit_start_time[work_unit_total_index]);

	pthread_mutex_unlock (&this->work_unit_q_mutex);


}

//! Dequeue work unit and distribute to device
/*!
Dequeue work unit and distribute to device
\param context, The device context which the work unit is distributing to
\param pfn_init_callback, The call back funtion to initilize kernel execution
\param init_args, The arguments for the pfn_init_callback function
\param pfn_finalize_callback, The call back funtion to finalize kernel execution
\param finalize_args, The arguments for the pfn_finalize_callback function
\param status, Operation status
*/
void work_pool::extract_and_distribute(_work_pool_context context,  
	void (*pfn_init_callback)(work_pool *, _work_pool_context, work_unit *, void*),	
	void* init_args,							
	void (*pfn_finalize_callback)(work_pool *, _work_pool_context, void*),	
	void* finalize_args,
	cl_int* status)
{
	pthread_mutex_lock (&this->work_unit_q_mutex);
	//printf("[in Extract and Execute]: kernel index: %d\n", work_unit_ready->kernel_index);
#ifdef VERBOSE
	printf("#########################################################################\n");
	printf("########### [Extract]: execute work unit at index: %d num_work_units: %d\n", this->index_out,  this->num_work_units);
#endif
	if(this->num_work_units == 0)
	{
#ifdef VERBOSE		
		printf("########### [Extract]: wait for the signal, queue is empty.\n");
		printf("#########################################################################\n");
#endif
		//work_pool_state = WORK_POOL_EMPTY;
		if(this->done != 1)
		{
			//pthread_mutex_lock (&this->work_unit_q_mutex);
			this->num_sleeping_devices++;
			pthread_cond_wait(&this->work_unit_q_not_empty_cv, &this->work_unit_q_mutex);
			//pthread_mutex_unlock (&this->work_unit_q_mutex);
		}
		
	}
	
	if(this->num_work_units > 0)
	{
#ifdef VERBOSE
		printf("#########################################################################\n");
		printf("########### [Extract]: Queue is not empty\n");
#endif
		//pthread_mutex_lock (&this->work_unit_q_mutex);
		//work_unit *work_unit_ready = (work_unit *)malloc(sizeof(work_unit));
		//Operations on the queue
		

		//check dependency and find the next available one
		for(unsigned int m=0;m<this->num_work_units;m++)
		{
			//work_unit_ready = this->work_pool_start[this->index_out];
			//memcpy(work_unit_ready, this->work_pool_start[this->index_out], sizeof(work_unit));
			//work_unit_ready->priority = this->work_pool_start[this->index_out]->priority;

			if(this->work_pool_start[this->index_out]->work_unit_status == CL_WORKUNIT_INITIALIZED)
			{

				if(this->work_pool_start[this->index_out]->dependency != NULL && this->work_pool_start[this->index_out]->dependency->num_events_in_wait_list != 0)
				{
					printf("---------------------------------------------------------------------------------ever enter here?\n");
					int dep_flag = 0;
					for(unsigned int parents=0;parents<this->work_pool_start[this->index_out]->dependency->num_events_in_wait_list;parents++)
					{
						//get the event status of each one
						cl_int event_status;
						//clCreateUserEvent(context.context, status);
						clGetEventInfo(this->work_pool_start[this->index_out]->dependency->event_wait_list[parents], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &event_status, NULL);
						if(event_status != CL_QUEUED)
						{
							dep_flag = 1;
							break;
						}
					}

					if(dep_flag == 0)
					{
						this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_READY;
						break;
					}
					else
					{
						this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_WAITING;

						if(this->index_out == max_size - 1)
							this->index_out = 0;
						else
							this->index_out++;
					}
					
				}
				else
				{
					this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_READY;
					break;
				}
			}
			else if(this->work_pool_start[this->index_out]->work_unit_status == CL_WORKUNIT_WAITING)
			{
				for(unsigned int parents=0;parents<this->work_pool_start[this->index_out]->dependency->num_events_in_wait_list;parents++)
				{
					//TODO: If work unit is waiting for a dependent work unit to finish
					//get the event status of each one
					//if all are done, work_unit_status = CL_WORKUNIT_READY, break;
					//otherwise assign the work_unit_status = CL_WORKUNIT_WAITING, index_out++						
				}
			}
			else if(this->work_pool_start[this->index_out]->work_unit_status == CL_WORKUNIT_COMPLETE)
			{
				if(this->work_pool_start[this->index_out]->dependency != NULL && this->work_pool_start[this->index_out]->dependency->num_events_in_wait_list != 0)
				{
					printf("---------------------------------------------------------------------------------ever enter here?\n");
					int dep_flag = 0;
					for(unsigned int parents=0;parents<this->work_pool_start[this->index_out]->dependency->num_events_in_wait_list;parents++)
					{
						//get the event status of each one
						cl_int event_status;
						//clCreateUserEvent(context.context, status);
						clGetEventInfo(this->work_pool_start[this->index_out]->dependency->event_wait_list[parents], CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &event_status, NULL);
						if(event_status != CL_QUEUED)
						{
							dep_flag = 1;
							break;
						}
					}

					if(dep_flag == 0)
					{
						this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_READY;
						break;
					}
					else
					{
						this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_WAITING;

						if(this->index_out == max_size - 1)
							this->index_out = 0;
						else
							this->index_out++;
					}
					
				}
				else
				{
					this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_READY;
					break;
				}							
			}
			else
			{
				printf("########### Invalid work unit status, error!\n");
				exit(-1);
			}					
			
		}
		//pthread_mutex_unlock (&this->work_unit_q_mutex);

		if(this->work_pool_start[this->index_out]->work_unit_status == CL_WORKUNIT_READY)
		{
			//pthread_mutex_lock (&this->work_unit_q_mutex);
#ifdef VERBOSE
			printf("########### [Extract]: Found the ready work unit, dequeue!\n");
#endif
			work_unit *work_unit_ready = (work_unit *)malloc(sizeof(work_unit));
			memcpy(work_unit_ready, this->work_pool_start[this->index_out], sizeof(work_unit));
			work_unit_ready->priority = this->work_pool_start[this->index_out]->priority;
			this->work_pool_start[this->index_out]->work_unit_status = CL_WORKUNIT_COMPLETE;
#ifdef VERBOSE
			printf("########### [Extract]: This ready work unit no.%d priority: %d\n", work_unit_ready->unit_index, work_unit_ready->priority);
#endif

			if(this->index_out == max_size - 1)
				this->index_out = 0;
			else
				this->index_out++;			

			if(this->num_work_units == WORKPOOL_CAP)
			{
#ifdef VERBOSE
				printf("########### [Extract]: Signal the work pool is not full anymore\n");
#endif
				pthread_cond_signal(&this->work_unit_q_full_cv);
			}

			this->num_work_units--;

			if(this->num_work_units == 0)
				work_pool_state = WORK_POOL_EMPTY;		
			else
				work_pool_state = WORK_POOL_NONEMPTY;			

#ifdef VERBOSE
			printf("########### [Extract]: index operations finished\n");
			printf("########### [Extract]: after extract: index_out: %d, num_work_units: %d\n", this->index_out,  this->num_work_units);
#endif
			
			//pthread_mutex_unlock (&this->work_unit_q_mutex);

			work_unit_ready->context = context.context;
			work_unit_ready->program = work_unit_ready->program_all[context.work_pool_context_idx];
			
			//cl_kernel kernel = NULL;
			work_unit_ready->kernel = work_unit_ready->kernel_all[context.work_pool_context_idx];


			if(pfn_init_callback != NULL)
				pfn_init_callback(this, context, work_unit_ready, init_args);

			//TODO: set arguments
			cl_int set_arg_status = 0;

			for(unsigned int arg_num=0; arg_num <work_unit_ready->arguments.size(); arg_num++)
			{
				
				if(work_unit_ready->arguments.at(arg_num)->type == INT_ARRAY_TYPE || work_unit_ready->arguments.at(arg_num)->type == FLOAT_ARRAY_TYPE)
				{
					cl_mem data_tmp;
					data_tmp = this->request_buffer(context, work_unit_ready->arguments.at(arg_num)->arg_pointer, work_unit_ready->arguments.at(arg_num)->size, NULL, work_unit_ready->arguments.at(arg_num)->read_write_flag);
					/*if(work_unit_ready->arguments.at(arg_num)->read_write_flag == 0)
					{
						data_tmp = this->request_buffer(context, work_unit_ready->arguments.at(arg_num)->arg_pointer, work_unit_ready->arguments.at(arg_num)->size, NULL, READ_ONLY);
					}
					else
					{
						data_tmp = this->request_buffer(context, work_unit_ready->arguments.at(arg_num)->arg_pointer, work_unit_ready->arguments.at(arg_num)->size, NULL, READ_WRITE);
					}*/
					//set_arg_status |= clEnqueueWriteBuffer(context.command_queue, data_tmp, CL_TRUE, 0, work_unit_ready->arguments.at(arg_num).size, work_unit_ready->arguments.at(arg_num).arg_pointer, 0, NULL, NULL);
					//Added for testing by Enqiang
					/*float *data_test_before = (float *)malloc(work_unit_ready->arguments.at(arg_num).size);
					*status = clEnqueueReadBuffer(context.command_queue, data_tmp, CL_TRUE, 0, work_unit_ready->arguments.at(arg_num).size, (void *)data_test_before, 0, NULL, NULL);
					//float *data = (float *)work_unit_ready->arguments.at(arg_num).arg_pointer;
					printf("within data read back: data_test_before[1]: %f\n", data_test_before[1]);
					*/
					set_arg_status  |= clSetKernelArg(work_unit_ready->kernel, work_unit_ready->arguments.at(arg_num)->index, sizeof(cl_mem), (void *)&data_tmp); 
				}
				else if (work_unit_ready->arguments.at(arg_num)->type == INT_TYPE)
				{
					set_arg_status  |= clSetKernelArg(work_unit_ready->kernel, work_unit_ready->arguments.at(arg_num)->index, sizeof(int), (void *)&work_unit_ready->arguments.at(arg_num)->value_int); 
				}			
			}
			cl_errChk(set_arg_status, "Error setting work unit args", true);

			//cl_event event_test;

			//event_test = clCreateUserEvent(context.context, status);	

			cl_uint work_unit_total_index = this->query();
			cl_getTime(&this->unit_start_time[work_unit_total_index]);

			//printf("[Extract]: executing kernel\n");
			*status = clEnqueueNDRangeKernel(context.command_queue, 
				work_unit_ready->kernel, 
				work_unit_ready->work_dim, 
				work_unit_ready->global_work_offset, 
				work_unit_ready->global_work_size,
				work_unit_ready->local_work_size, 
				0,
				NULL,
				NULL);
			cl_errChk(*status, "Executing kernel", true);
			//clFinish(context.command_queue);
			//cl_uint work_unit_total_index = this->query();
			//cl_getTime(&this->unit_start_time[work_unit_total_index]);

			clFlush(context.command_queue);			

			//cl_int event_status;
			//clGetEventInfo(*work_unit_ready->dependency->event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &event_status, NULL);

			/*if(event_status == CL_COMPLETE)
				printf("------event_status: %d, CL_COMPLETE\n", event_status);
			else if(event_status == CL_RUNNING)
				printf("------event_status: %d, CL_RUNNING\n", event_status);
			else if(event_status == CL_SUBMITTED)
				printf("------event_status: %d, CL_SUBMITTED\n", event_status);
			else if(event_status == CL_QUEUED)
				printf("------event_status: %d, CL_QUEUED\n", event_status);*/

			//printf("[Extract]: done executing kernel\n");
				if(pfn_finalize_callback != NULL)
					pfn_finalize_callback(this, context, finalize_args);

			//pthread_mutex_lock (&this->work_unit_q_mutex);
				for(unsigned int arg_num=0; arg_num <work_unit_ready->arguments.size(); arg_num++)
				{

					if(work_unit_ready->arguments.at(arg_num)->read_write_flag == READ_WRITE)
					{
						cl_mem data_output = this->request_buffer(context, work_unit_ready->arguments.at(arg_num)->arg_pointer, work_unit_ready->arguments.at(arg_num)->size, NULL, CL_FALSE);
					//float *data_test = (float *)malloc(work_unit_ready->arguments.at(0).size);
						*status = clEnqueueReadBuffer(context.command_queue, data_output, CL_FALSE, 0, work_unit_ready->arguments.at(arg_num)->size, work_unit_ready->arguments.at(arg_num)->arg_pointer, 0, NULL, NULL);
					//float *data = (float *)work_unit_ready->arguments.at(arg_num).arg_pointer;
					//printf("within data read back: data_test[1]: %f\n", data_test[1]);
						cl_errChk(*status, "Reading output from buffer", true);
					}						
				}


			}


#ifdef VERBOSE
			printf("########### [Extract]: Finish execution of the work unit\n");
#endif
		//pthread_mutex_lock (&this->work_unit_q_mutex);

#ifdef PRINT_PROFILING	    
			printf("[in Dequeue] work units status: \n");
			for(int i=0;i<WORKPOOL_CAP;i++)
			{
				if(this->work_pool_start[i] != (work_unit *)0xcdcdcdcd)
				{
					printf("%d ", this->work_pool_start[i]->work_unit_status);
				}
				else
				{
					break;
				}
			}
			printf(" \n\n");
#endif				

		}

#ifdef VERBOSE
		printf("########### [Extract]: Exit extraction, and realease the lock\n");
		printf("#########################################################################\n");
#endif
		pthread_mutex_unlock (&this->work_unit_q_mutex);

	}

//! Init buffer table
/*!
Init buffer table
\param buffer_table, The buffer table
*/
void work_pool::init_buffer_table(_buffer_table buffer_table)
{
	//printf("initialize buffer table\n");

	this->buffer_table.num_devices = this->total_num_devices;
	this->buffer_table.num_entries = 0;
	//this->buffer_table.entry_list = NULL;

	return;
}

//! Buffer management
/*!
Buffer management across devices (platforms)
\param context_requested, The device context which the requested buffer will be on
\param data, The original host data pointer
\param size, The size of the requested buffer
\param desc, The description (currently unused)
\param init, The flag which indicates if the requested buffer has to be initialized to a certain value
*/
double total_buffer_time;
double total_transfer_time;

cl_mem work_pool::request_buffer(_work_pool_context context_requested, void *data, cl_int size, char* desc, cl_bool read_only_flag)
{
	cl_int status;
	buffer_entry entry;

	cl_time begin_time, end_time;
	cl_time begin_transfer_time, end_transfer_time;
	cl_getTime(&begin_time);    

	for(unsigned int j=0;j<buffer_table.entry_list.size();j++)
	{
		buffer_entry entry_lookup = buffer_table.entry_list.at(j);
		if(entry_lookup->data == (int)data)
		{
			//data is already in the vector
			
			if(entry_lookup->valid_idx == context_requested.work_pool_context_idx)
			{
				cl_getTime(&end_time);
				total_buffer_time = total_buffer_time + cl_computeTime(begin_time, end_time);
                //printf("Buffer management(existing) time for this frame: %f\n", cl_computeTime(begin_time, end_time));
				return entry_lookup->buffer[entry_lookup->valid_idx];
			}
			else   //Not tested yet
			{
				if(entry_lookup->coherent_flag[context_requested.work_pool_context_idx] == READ_ONLY)
				{
					entry_lookup->valid_idx = context_requested.work_pool_context_idx;
					return entry_lookup->buffer[context_requested.work_pool_context_idx];
				}

				if(entry_lookup->coherent_flag[context_requested.work_pool_context_idx] == WRITE_ONLY)
				{
					entry_lookup->valid_idx = context_requested.work_pool_context_idx;
					return entry_lookup->buffer[context_requested.work_pool_context_idx];
				}

				cl_getTime(&begin_transfer_time);
                //copy buffer through CPU data pointer
				char *data_for_copy = (char *)malloc(sizeof(char)*size);
				status = clEnqueueReadBuffer(entry_lookup->pool_context[entry_lookup->valid_idx].command_queue, entry_lookup->buffer[entry_lookup->valid_idx], CL_TRUE, 0, 
					size, data_for_copy, 0, NULL, NULL); 
				if(status != CL_SUCCESS) {
					printf("error copying data to buffer\n");
					exit(-1);
				}
				
				entry_lookup->valid_idx = context_requested.work_pool_context_idx;

				entry_lookup->pool_context[entry_lookup->valid_idx] = context_requested;
                //if(entry_lookup->buffer[entry_lookup->valid_idx] == NULL)
                //{
				entry_lookup->buffer[entry_lookup->valid_idx] = clCreateBuffer(context_requested.context, CL_MEM_READ_WRITE, size, NULL, &status);		
				if(status != CL_SUCCESS) {
					printf("error creating buffer\n");
					exit(-1);
				}
                //}

				status = clEnqueueWriteBuffer(context_requested.command_queue, entry_lookup->buffer[entry_lookup->valid_idx], CL_TRUE, 0, 
					size, data_for_copy, 0, NULL, NULL); 
				if(status != CL_SUCCESS) {
					printf("error copying data to buffer\n");
					exit(-1);
				}

				entry_lookup->coherent_flag[entry_lookup->valid_idx] = read_only_flag;

				free(data_for_copy);
				cl_getTime(&begin_transfer_time);
				cl_getTime(&end_time);
				cl_getTime(&end_transfer_time);
				total_buffer_time = total_buffer_time + cl_computeTime(begin_time, end_time);
				total_transfer_time = total_transfer_time + cl_computeTime(begin_transfer_time, end_transfer_time);
                //printf("Buffer transfer time for this frame: %f\n", cl_computeTime(begin_transfer_time, end_transfer_time));
                //printf("Buffer management(transfer) time for this frame: %f\n", cl_computeTime(begin_time, end_time));
				return entry_lookup->buffer[entry_lookup->valid_idx];					
			}				
		} //if(entry_lookup->data == (int)data)       
	}

	//if the data is new to the buffer table
	entry = (buffer_entry)malloc(sizeof(_buffer_entry));
	entry->data = (int)data;
	entry->num_devices = this->total_num_devices;
	entry->pool_context = (work_pool_context)malloc(sizeof(_work_pool_context) * this->total_num_devices);
	entry->buffer = (cl_mem *)malloc(sizeof(cl_mem) * this->total_num_devices);
	entry->coherent_flag = (int *)malloc(sizeof(int) * this->total_num_devices);

	entry->valid_idx = context_requested.work_pool_context_idx;
	entry->pool_context[entry->valid_idx] = context_requested;

	if (read_only_flag == READ_ONLY)
	{
		entry->buffer[entry->valid_idx] = clCreateBuffer(context_requested.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			size, data, &status);
		cl_errChk(status, "Error creating read_only mem buffer", true);
		entry->coherent_flag[entry->valid_idx] = READ_ONLY;
	}
	else if (read_only_flag == WRITE_ONLY)
	{
		entry->buffer[entry->valid_idx] = clCreateBuffer(context_requested.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
			size, data, &status);
		cl_errChk(status, "Error creating write_only mem buffer", true);
		entry->coherent_flag[entry->valid_idx] = WRITE_ONLY;
	}
	else if (read_only_flag == READ_WRITE)
	{
		entry->buffer[entry->valid_idx] = clCreateBuffer(context_requested.context, CL_MEM_READ_WRITE, size, NULL, &status);		
		status |= clEnqueueWriteBuffer(context_requested.command_queue, entry->buffer[entry->valid_idx], CL_TRUE, 0, size, (void *)entry->data, 0, NULL, NULL);
		if(status != CL_SUCCESS) {
			printf("error creating read_write buffer\n");
			exit(-1);

		}
		entry->coherent_flag[entry->valid_idx] = READ_WRITE;
	}
	else
	{
		printf("error type of buffer requested\n");
		exit(-1);
	}

    /*if(read_only_flag == CL_TRUE)
	{
		status = clEnqueueWriteBuffer(context_requested.command_queue, entry->buffer[entry->valid_idx], CL_TRUE, 0, 
									  size, data, 0, NULL, NULL); 
		if(status != CL_SUCCESS) {
			printf("error copying data to buffer\n");
			exit(-1);
		}
	}*/

	//cl_copyToDevice_workpool(context.command_queue, buffer_entry->buffer[i], buffer_entry->data, size);

		this->buffer_table.entry_list.push_back(entry);
		buffer_table.num_entries++;
		cl_getTime(&end_time);
		total_buffer_time = total_buffer_time + cl_computeTime(begin_time, end_time);
    //printf("Buffer management(new) time for this frame: %f\n", cl_computeTime(begin_time, end_time));
		return entry->buffer[entry->valid_idx];

}

//! Query the information of the next work unit
/*!
Query the information of the next work unit
*/
cl_uint work_pool::query()
{
	//pthread_mutex_lock (&this->work_unit_q_mutex);
	if(this->work_pool_start[this->index_out] == (work_unit *)0xcdcdcdcd)
		return 0;
	else
	{
		//pthread_mutex_unlock (&this->work_unit_q_mutex);
		return this->work_pool_start[this->index_out]->unit_index;
	}
	
	//return 0;
}

//! Reset the buffer used in one frame
/*!
Reset the buffer used in one frame, and release the buffer
\param thread_id, The thread id which calls this function (for debugging)
*/
void work_pool::reset_buffer(int thread_id)
{
	
	cl_int status;

	for(unsigned int j=0;j<this->buffer_table.entry_list.size();j++)
	{
	//printf("thread_id: %d, buffer entry no. %d\n", thread_id, j);
		buffer_entry entry = this->buffer_table.entry_list.at(j);
		for(int i=0;i<this->buffer_table.num_devices;i++)
		{
			if(entry->buffer[i] != (cl_mem)0xcdcdcdcd)
			{
				status = clReleaseMemObject(entry->buffer[i]);
				cl_errChk(status, "Releasing mem object", true);
			}
		}
	}

	this->buffer_table.num_entries  = 0;
	this->buffer_table.entry_list.clear();

	//num_work_units = 0;
	//work_pool_state = WORK_POOL_EMPTY;

}

//! work_pool_finish, join all the threads
/*!
Query the information of the next work unit
*/
void work_pool::finish()
{
	
	while(this->num_work_units != 0 && this->work_pool_state != WORK_POOL_EMPTY);
	this->done = 1;

	pthread_mutex_destroy(&this->work_unit_q_mutex);
	pthread_cond_destroy(&this->work_unit_q_full_cv);
	pthread_cond_destroy(&this->work_unit_q_not_empty_cv);
	
	while(1)
	{
		sleep(1000);
		int exit = 1;
		for(int i=0;i<this->total_num_devices;i++)
		{
			exit &= this->thread_exit[i];
		}		
		if(exit == 1)
			break;
	}
	//Sleep(100000);

	this->reset_buffer(0);
	
	//pthread_attr_destroy(&this->work_pool_thread_attr);

	/*for(int i=0;i<this->total_num_devices;i++)
	{
		pthread_join(this->work_pool_scheduler_thread[i], NULL);
	}*/



	//pthread_exit(NULL);

		for(int i=0;i<this->total_num_devices;i++)
		{
			printf("!!!!!! on %d device, %d work units were executed\n", i, num_on_this_device[i]);
		}

#ifdef VERBOSE
		for(int i=0;i<this->total_num_devices;i++)
		{
			for(int j=0;j<this->num_on_this_device[i];j++)
			{
				printf("!!!!!! on %d device, execution time of %d work units is %f\n", i, j, this->execution_time_queue_per_device[i][j]);
			}
		}
#endif	
}
