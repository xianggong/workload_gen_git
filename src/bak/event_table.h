/**
	OpenCl Event handling framework	
	Events managed by a vector and handles passed onto OpenCl code
	Perhaad Mistry and Chris Gregg

*/
#ifndef _EVENT_TABLE_
#define _EVENT_TABLE_

#include <iostream>
#include <vector>

#include <sstream>
#include <string>
#include <CL/cl.h>
#include "clutils.h"
 
using namespace std;
//! Add usage with at for bound checking

//! Maximum Size of event_list vector 
//! Just added for simplicity sake
#define MAX_ALLOWED_EVENT_COUNT 100

//! A Better Event Handling Class
class event_table
{	
private:	
	//! A list of Events that are provided to external functions
	std::vector<cl_event> event_list;
	//! A list of events for IO Actions
	std::vector<cl_event> io_event_list;

	//! A list of strings for IO event logging
	std::vector<std::string> io_event_name;

	//! A list of strings for Kernel event logging
	std::vector<std::string> event_name;

public:
	  
	event_table();

	//! No of Events in Table
	int n;
	int io_n;

	int get_event_id();
	cl_event * return_event(int n=0);
	
	void print_trace_table();
	void clear_event_list();

	cl_event *  get_event(int);
	
	void print_kernel_time_table();
	void initial_allocate(int m);

	//! Assign a position in the event_list vector 
	/** 
		Store the return value to a static variable and 
		only call this function once
	*/
	int assign_event(string event_desc);

	int assign_io_event(string  event_desc);

	//! Assign t events in the event_list
	int assign_multiple_events(int t, string event_desc);


	//! Get Time Stamp
	void cl_TimeStamp(cl_event , cl_profiling_info, string );	
	//! Calculate Kernel Execution Time by difference between start and end times
	void cl_KernelTime(cl_event event_time);

	void release();
};

#endif // //_EVENT_TABLE_