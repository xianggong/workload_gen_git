if HAVE_OPENCL

lib_LTLIBRARIES = libclworkpool.la

libclworkpool_la_LDFLAGS = -lpthread -lOpenCL

libclworkpool_la_SOURCES = \
			   clExtensions.cpp \
			   clExtensions.h \
			   \
			   gettimeofday.cpp \
			   gettimeofday.h

AM_CFLAGS = -m32

endif
