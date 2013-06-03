#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <clExtensions.h>

#define VEC_NUMBER 1
#define SAME_VEC_NUMBER 64

// OpenCL kernel. Each work item takes care of one element of c

int main( int argc, char* argv[] )
{
	// Length of vectors
	unsigned int n = 1024*16; //max vector size is 1M
	//unsigned int n = 1024; //max vector size is 1M


	// Host input vectors
	float **h_a;
	float **h_b;
	// Host output vector
	float **h_c;
	// Output vector for verification
	float **h_c_v;

	// Size, in bytes, of each vector
	unsigned int bytes[VEC_NUMBER];

	unsigned int random = 4;

	//Initialize the vectors
	h_a = (float **)malloc(sizeof(float *)*VEC_NUMBER);
	h_b = (float **)malloc(sizeof(float *)*VEC_NUMBER);
	h_c = (float **)malloc(sizeof(float *)*VEC_NUMBER);
	h_c_v = (float **)malloc(sizeof(float *)*VEC_NUMBER);

	//bytes = (int *)malloc(sizeof(int)*vec_number);

	unsigned int i, j;
	for( i=0;i<VEC_NUMBER;i++)
	{
		bytes[i] = n/random;
		h_a[i] = (float *)malloc(bytes[i]*sizeof(float));
		h_b[i] = (float *)malloc(bytes[i]*sizeof(float));
		h_c[i] = (float *)malloc(bytes[i]*sizeof(float));
		h_c_v[i] = (float *)malloc(bytes[i]*sizeof(float));
		for(j=0;j<bytes[i];j++)
		{
			h_a[i][j] = (float)i + 1.0;
			h_b[i][j] = (float)j + 1.0;
			h_c[i][j] = 0;
			h_c_v[i][j] = h_a[i][j] + h_b[i][j];
		}
	}

	cl_time totalStart, totalEnd;
	printf("\n################### Work Pool execution of the Workload #####################\n\n");
	cl_int status = false;

	unsigned int init_num_work_units = VEC_NUMBER * SAME_VEC_NUMBER;

	// init work pool
	work_pool work_pool_vec;
	
	work_pool_vec.init(WORKPOOL_CAP, init_num_work_units, &status);
	if(cl_errChk(status, "Initialize work pool", true)) 					
		exit(1);	

	work_pool_vec.total_unfinished_work_units = VEC_NUMBER * SAME_VEC_NUMBER;

	work_unit work_unit_vec[VEC_NUMBER];
	

	cl_getTime(&totalStart);
	
	for(int i=0;i<VEC_NUMBER;i++)
	{		
		size_t globalSize, localSize;

		// Number of work items in each local work group
		localSize = 64;

		// Number of total work items - localSize must be devisor
		globalSize = (size_t)ceil(n/(float)localSize)*localSize;

		work_unit_vec[i].init(&work_pool_vec, 
			NULL, 
			"../src/vectoradd.cl",
			"vecAdd", 
			NULL, 
			1,
			NULL, 
			&globalSize,
			&localSize,
			0, 
			NULL, 
			0, 
			0, 
			&status);
		if(cl_errChk(status, "Initialize a work unit", true)) 					
			exit(1);
		
		
		work_unit_vec[i].set_argument(0, FLOAT_ARRAY_TYPE, 0, 0.0, h_a[i], bytes[i]*sizeof(float), READ_ONLY, &status);
		if(cl_errChk(status, "set argument", true)) 					
			exit(1);

		work_unit_vec[i].set_argument(1, FLOAT_ARRAY_TYPE, 0, 0.0, h_b[i], bytes[i]*sizeof(float), READ_ONLY, &status);
		if(cl_errChk(status, "set argument", true)) 					
			exit(1);

		work_unit_vec[i].set_argument(2, FLOAT_ARRAY_TYPE, 0, 0.0, h_c[i], bytes[i]*sizeof(float), WRITE_ONLY, &status);
		if(cl_errChk(status, "set argument", true)) 					
			exit(1);

		work_unit_vec[i].set_argument(3, INT_TYPE, bytes[i], 0.0, NULL, 0, READ_ONLY, &status);
		if(cl_errChk(status, "set argument", true)) 					
			exit(1);

		for(int j=0;j<SAME_VEC_NUMBER;j++)
		{		
			work_pool_vec.enqueue(&work_unit_vec[i], PRIORITY_LEVEL-j, &status);
		}	

	}

	work_pool_vec.finish();

	cl_getTime(&totalEnd);

	printf("execution time: %f\n", cl_computeTime(totalStart, totalEnd));


	//check the result with cpu execution

	/*int result = 0;
	for(i=0; i<VEC_NUMBER; i++)
	{
		for(j=0; j<bytes[i]; j++)
		{
			if(h_c[i][j] != h_c_v[i][j])
			{
				result = 1;
				printf("h_c[i][j]: %f, h_c_v[i][j]: %f\n", h_c[i][j], h_c_v[i][j]);
				break;
			}
		}
	}

	if( 0 == result)
	{
		printf("final result is correct\n");
	}
	else
	{
		printf("$$$$$$ incorrect !!! $$$$$\n");
	}*/

	//release host memory
		free(h_a);
		free(h_b);
		free(h_c);

	//pthread_exit(NULL);

		return 0;
	}
