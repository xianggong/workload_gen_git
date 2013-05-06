#include "event_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>
#include <CL/cl.h>

/**
	Traverse event_list vector and print out all the time-stamps
	Also print corresponding information of event_name vector 
	to get complete trace
*/
//! Print timestamp for all cl_event entries in event_list vector
void event_table::print_trace_table()
{
	//cl_sync();

	int i;
	printf("Kernel Trace\n");
	for(i=0;i<n;i++)
	{

		printf("%d\tENQ\t",i);
		cl_TimeStamp(event_list[i], CL_PROFILING_COMMAND_QUEUED  ,event_name[i]);
		printf("%d\tSUBMIT\t",i);
		cl_TimeStamp(event_list[i], CL_PROFILING_COMMAND_SUBMIT  ,event_name[i]);
		printf("%d\tSTART\t",i);
		cl_TimeStamp(event_list[i], CL_PROFILING_COMMAND_START  ,event_name[i]);
		printf("%d\tEND\t",i);
		cl_TimeStamp(event_list[i], CL_PROFILING_COMMAND_END  ,event_name[i]);
	}
}

//! Print out execution duration for all cl_event entries in event_list
/*!
print_kernel_time_table assumes a Flush and Finish have
been called before Final Data Gathering
*/
void event_table::print_kernel_time_table()
{
	int i;
	printf("Kernel Execution Time Table\n");
	for(i=0;i<n;i++)
	{		
		cout<<event_name[i]<<"\t";		
		cl_KernelTime(event_list[i]);
	}

}

//! Query the cl_event passed 
/**
	\param event_time cl_event passed (Usually an event from the cl_event vector)
	\param profile_mode Event type queried
	\param event_name String to describe event
*/
void event_table::cl_TimeStamp(
			cl_event event_time,
			cl_profiling_info profile_mode,
			string event_name)
{
	cl_int kerneltimer;
	cl_ulong startTime;

	kerneltimer = clGetEventProfilingInfo(event_time,
		profile_mode,
		sizeof(cl_ulong), &startTime, NULL);
	cl_errChk(kerneltimer, "Profiling error", true);

	cout<<event_name<<"\t";
	printf("%lu\n",startTime);
	
}

//! Time kernel execution using cl_event
/*!
	Prints out the time taken between the start and end of an event
	\param event_time Event whose duration required
*/
void event_table::cl_KernelTime(cl_event event_time)
{
	cl_int kerneltimer;
	cl_ulong starttime;
	cl_ulong endtime;

	kerneltimer = clGetEventProfilingInfo(event_time,
		CL_PROFILING_COMMAND_START,
		sizeof(cl_ulong), &starttime, NULL);
	cl_errChk(kerneltimer, "Profiling error", true);

	kerneltimer = clGetEventProfilingInfo(event_time,
		CL_PROFILING_COMMAND_END  ,
		sizeof(cl_ulong), &endtime, NULL);
	cl_errChk(kerneltimer, "Profiling error", true);

	unsigned long elapsed =  (unsigned long)(endtime - starttime);
	printf("Kernel Execution\t%ld ns\n",elapsed);
}


event_table::event_table()
{
	n=0;
}


//! Returns an Index and pushes an event onto the vector
int event_table::assign_io_event(string event_desc)
{
	if(n > MAX_ALLOWED_EVENT_COUNT)
	{
		printf("Too many events for Safe and Clean Profiling\n");
		exit(1);
	}
	else
	{
		io_n = io_n+1;
		io_event_list.resize(n);
		io_event_name.push_back("Desc");
		printf("Total IO Events Allocated %d\n",io_n);
		return io_n;
	}
}

//! Returns an Index and pushes an event onto the vector
int event_table::assign_event(string event_desc)
{

	if(n > MAX_ALLOWED_EVENT_COUNT)
	{
		printf("Too many events for Safe and Clean Profiling\n");
		exit(1);
	}
	else
	{
		n=n+1;
		//
		event_name.push_back(event_desc);
	}
	//printf("Total Events Allocated %d\n",n);
	return n-1;
}

void event_table::initial_allocate(int m)
{
	event_list.resize(m);
}


/**
	Used to assign more than one event in cases when the 
	operating scope of the event handler sees more than one 
	Enqueue-type call.\n
		
	This is useful for cases like BuildDet where we need multiple
	events per kernel per frame
*/
int event_table::assign_multiple_events(int t, string event_desc)
{
	int old_value=n;
	n = n+t;
	//event_list.resize(n);

	for(int i=0;i<t;i++)
	{
		std::stringstream out;
		out <<event_desc <<"_"<<i<<"\t";
		event_name.push_back(out.str());
		out.flush();
	}
	//printf("%d Requested \t Total Events Allocated %d\n",t,n);
	return old_value;
}

//! Needs to be assigned to a static variable
cl_event * event_table::get_event(int i)
{
	//printf("Get Event\n");
	if(n> MAX_ALLOWED_EVENT_COUNT)
	{
		printf("Invalid Event Index");
		exit(1);
	}
	else
	{
		//! Add Bound Checking
		//printf("Size is %d\n",event_list.size());
		//printf("Return Event %d\n",i);

#ifdef _DISABLE_PROFILING_		
		return NULL;
#else
		return &(event_list[i]);
#endif // _DISABLE_PROFILING_

	}
}

void event_table::clear_event_list()
{
	event_name.clear();
	event_list.clear();
}

void event_table::release()
{
	for(int i = 0; i < (int)event_list.size(); i++) {
		clReleaseEvent(event_list[i]);
	}
}
