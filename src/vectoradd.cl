__kernel void vecAdd(  __global float *a,                       
                       __global float *b,                       
                       __global float *c,                       
                       const unsigned int n)                    
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);
	int i;  
	float r=1.0; 
	

	if (id < n)
	{
		for(i=0;i<1000000;i++)                                          
			c[id] = c[id]+ a[id]*r + b[id];
	}
}                                                               
                                                                