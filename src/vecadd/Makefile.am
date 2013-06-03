bin_PROGRAMS = vecadd

vecadd_SOURCES = \
		 vectoradd.cpp \
		 vectoradd.h 

CLWORKPOOL = \
	$(top_builddir)/lib/libclworkpool.a

vecadd_LDFLAGS = $(CLWORKPOOL) -lOpenCL

AM_CPPFLAGS = @CL_WORKPOOL_INCLUDES@
