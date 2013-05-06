#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <vector>
#include "pthread.h"

#ifdef _WIN32
#include <tchar.h>
#include <Windows.h>
typedef __int64 cl_time; 
#else
#include <sys/time.h>
typedef double cl_time;
#endif

//#define PRINT_PROFILING

#define WORKPOOL_CAP 22
#define PRIORITY_LEVEL 256


// Init extension function pointers
#define INIT_CL_EXT_FCN_PTR(name) \
if(!pfn_##name) { \
	pfn_##name = (name##_fn) clGetExtensionFunctionAddress(#name); \
	if(!pfn_##name) { \
		printf("Cannot get pointer to ext. fcn.\n "); \
		exit(1); \
	} \
}

static int starvation = 0;
static int full = 0;

typedef struct {
	cl_context      context;
	cl_kernel		kernel;
	cl_uint			work_dim;
	const size_t*	global_work_offset;
	const size_t*	global_work_size;
	const size_t*	local_work_size;
	cl_int          status;
} _cl_workunit, *cl_workunit;

typedef struct {
	cl_int id;
	cl_int num_workunits;
	cl_workunit* workunit_list;
	cl_int* wait_list;
} _cl_dependency, *cl_dependency;

typedef struct {
	cl_context			context;
	cl_int              num_devices;
	cl_device_id*		device_list;
	cl_command_queue*	queue_list;
} _cl_workpool, *cl_workpool;



cl_workpool clCreateWorkPool(cl_context context,
	cl_int num_devices,
	cl_device_id* device_list,
	cl_uint flags, 
	cl_int* status);

cl_int clReleaseWorkPool(cl_workpool workpool);

cl_workunit clCreateWorkUnit(cl_context context,
	cl_kernel kernel,
	cl_uint work_dim,
	const size_t* global_work_offset,
	const size_t* global_work_size,
	const size_t* local_work_size,
	cl_uint flags,
	cl_int* status);

cl_int clReleaseWorkUnit(cl_workunit workunit);

cl_int clEnqueueWorkUnit(cl_workpool workpool,
	cl_workunit workunit,
	void (*pfn_init_callback)(void*),
	void* init_args,
	void (*pfn_finalize_callback)(void*),
	void* finalize_args,
	cl_event* event);

cl_dependency clCreateDependency(cl_int num_workunits,
	cl_workunit* workunit_list,
	cl_int* status);

cl_int clWaitForDependency(cl_dependency dep);

#define CL_WORKUNIT_INITIALIZED 0x0000
#define CL_WORKUNIT_WAITING     0x0001
#define CL_WORKUNIT_READY       0x0002
#define CL_WORKUNIT_COMPLETE    0x0003
#define CL_WORKUNIT_INVALID     0x0004



#define FALSE 0
#define TRUE 1

#define MAX_ERR_VAL 64

int cl_errChk(const cl_int status, const char *msg, bool exitOnErr);
void cl_sync();
void cl_getTime(cl_time* time); 
double cl_computeTime(cl_time start, cl_time end);

#define WORK_POOL_INIT    0x000F
#define WORK_POOL_EMPTY    0x0000
#define WORK_POOL_NONEMPTY 0x0001
#define WORK_POOL_FULL     0x0002
#define WORK_POOL_FAIL     0x00FF

#define CL_TRUE_INIT_CONST 0x03




typedef struct {
	cl_int id;
	cl_uint num_events_in_wait_list;
	const cl_event *event_wait_list;
	cl_event *event;
} _work_unit_dependency, *work_unit_dependency;

typedef struct {
	cl_uint work_pool_context_idx;
	cl_platform_id platform;
	cl_device_id device;
	char platform_vendor[100];
	char device_vendor[100];
	char device_name[100];
	cl_uint device_max_compute_units;
	cl_uint device_max_frequency;
	cl_device_type dtype;
	cl_context context;          
	cl_command_queue command_queue;  
} _work_pool_context, *work_pool_context;

typedef struct {
	cl_context context;          
	cl_kernel* pre_compiled_kernels;
} _pre_compiled_kernels_per_context, *pre_compiled_kernels_per_context;

typedef struct {
	cl_int num_devices;
	int data;
	work_pool_context pool_context;	
	cl_mem* buffer;
	cl_int valid_idx;
	int* coherent_flag; //1->read_only; 2->write_only; 3->read_write
} _buffer_entry, *buffer_entry;


typedef struct {
	cl_int              num_devices;
	//work_pool_context	pool_context;	
	cl_int              num_entries;	
	std::vector<buffer_entry> entry_list;
} _buffer_table, buffer_table;

typedef struct {
	cl_int              index;
	void*				arg_pointer;
	cl_int              size;
	cl_int              type; //0-pointer to int data array; 1- pointer to float data array; 2-integer value; 3-float value;
	cl_int              value_int;
	float               value_float;
	cl_int              read_write_flag; //0-read only; 1-write only; 2-read and write; 3-must write back;
} _work_unit_arg, *work_unit_arg;




class work_pool;


class work_unit {

public:

	  //! Constructor
	  //work_unit( );
	void init(work_pool *work_pool,
		cl_context prefer_context,
		char* program_with_path,
		char* kernel_name,
		work_unit_dependency dep,
		cl_uint work_dim,
		const size_t* global_work_offset,
		const size_t* global_work_size,
		const size_t* local_work_size,
		cl_uint flags,
		pre_compiled_kernels_per_context kernel_list,
		cl_uint kernel_no,
		cl_uint kernel_index,
		cl_int* status); 

	cl_context* context_all;
	char* program_with_path;
	cl_program* program_all;
	char* kernel_name;
	cl_kernel* kernel_all;

	cl_context context;
	cl_program program;
	cl_kernel kernel;

	cl_context prefer_context;

	cl_uint work_dim;
	work_unit_dependency dependency;
	const size_t* global_work_offset;
	const size_t* global_work_size;
	const size_t* local_work_size;

	std::vector<work_unit_arg> arguments;

	cl_uint priority;
	cl_uint unit_index;
	cl_uint flags;
	cl_uint work_unit_status;

	//cl_uint kernel_index;

	//pre_compiled_kernels_per_context pre_compiled_kernels_per_context;

	cl_program compile_program(_work_pool_context context, char * program_path, char * compileoptions, bool verbosebuild);
	cl_kernel create_kernel(cl_program program, const char* kernel_name);
	void set_argument(cl_int index, cl_int type, cl_int int_value, float float_value, void * data, cl_int data_size, cl_int flag, cl_int * status);

	  /*
	  void (*pfn_init_callback)(void* init_args);
	  void* init_args;
	  void (*pfn_finalize_callback)(void* finalize_args);
	  void* finalize_args;
	  */
	};



	class work_pool {

	public:

    //! Standard Constructor
		work_pool( );
		void init(int max_size, unsigned int init_number_work_units, cl_int* status);

		work_pool_context context;
		work_unit **work_pool_start;
		//work_unit *work_pool_end;
		//work_unit *work_pool_in;
		//work_unit *work_pool_out;
		unsigned int index_end;
		unsigned int index_in;
		unsigned int index_out;

		unsigned int work_unit_index;


		cl_uint max_size;
		cl_uint num_work_units;
		cl_uint work_pool_status;
		cl_uint total_num_devices;

		cl_uint num_sleeping_devices;

		cl_uint work_pool_state;

		_buffer_table buffer_table;

		work_unit_dependency **dep;

		pthread_t *work_pool_scheduler_thread;
		pthread_attr_t work_pool_thread_attr;
		void* work_pool_scheduler_arg;

		pthread_mutex_t work_unit_q_mutex;
		pthread_cond_t work_unit_q_not_empty_cv;
		pthread_cond_t work_unit_q_full_cv;

		unsigned int done;
		unsigned int extract_done;

		unsigned int *num_on_this_device;

		unsigned int *thread_exit;

		unsigned int total_unfinished_work_units;

	//for profiling
		cl_time *unit_start_time, *unit_end_time;

		double **execution_time_queue_per_device;
	unsigned int *busy_direction; //0-stay;1-UP, more busier;2-DOWN, less busier
	int *dynamic_offset;

	void work_pool_scheduler(int device_id);
	friend void *pthread_scheduler(void *work_pool_scheduler_arg);

	work_pool_context work_pool_get_contexts();
	void work_units_copy(work_unit* work_unit_from, work_unit* work_unit_to);
	void enqueue(work_unit* work_unit, cl_uint priority, cl_int* status);
	void extract_and_distribute(_work_pool_context context, 		              
		void (*pfn_init_callback)(work_pool *, _work_pool_context, work_unit *, void*),	
		void* init_args,							
		void (*pfn_finalize_callback)(work_pool *, _work_pool_context, void*),	
		void* finalize_args,
		cl_int* status);

	cl_mem work_pool::request_buffer(_work_pool_context context, void *data, cl_int size, char* desc = NULL, cl_bool init = CL_FALSE);

	void init_buffer_table(_buffer_table buffer_table);

	

	cl_uint query();

	void reset_buffer(int thread_id);
	void finish();

private:

	cl_uint numPlatforms;
	cl_platform_id *platforms;
	cl_device_id *devices;
	
};

typedef struct 
{
	int				thread_id;
	work_pool		*work_pool_in;
}_scheduler_thread_data, scheduler_thread_data;

#define READ_ONLY 1
#define WRITE_ONLY 2
#define READ_WRITE 3

#define INT_ARRAY_TYPE 0
#define FLOAT_ARRAY_TYPE 1
#define INT_TYPE 2

#define STAY 0 //stay
#define UP 1 //more busier
#define DOWN 2 //less busier







