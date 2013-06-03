AC_INIT([clWorkpool], [0.1])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_MACRO_DIR([m4])
AC_PROG_CXX
AC_PROG_INSTALL
AC_LIBTOOL_DLOPEN
AC_PROG_LIBTOOL

#############
# Test OpenCL
#############

# Check if OpenCL is present
have_opencl=no
if test "x$AMDAPPSDKROOT" != "x"
then
  # Use temporary CFLAGS, LDFLAGS
  saved_cflags="$CFLAGS"
  saved_ldflags="$LDFLAGS"
  saved_libs="$LIBS"
  CFLAGS="-I$AMDAPPSDKROOT/include"
  LDFLAGS="-L$AMDAPPSDKROOT/lib/x86 -L$AMDAPPSDKROOT/lib/x86_64"
  LIBS="-lOpenCL"

  # Try program
  AC_LINK_IFELSE(
  [
    AC_LANG_PROGRAM(
      [[#include <CL/cl.h>]],
      [[
        cl_int ret;
        cl_platform_id platform;
        cl_uint num_platforms;
        ret = clGetPlatformIDs(1, &platform, &num_platforms);
              return CL_SUCCESS;
      ]]
    )
  ],
  [
    have_opencl=yes
  ])

  # Restore original CFLAGS, LDFLAGS, LIBS
  CFLAGS="$saved_cflags"
  LDFLAGS="$saved_ldflags"
  LIBS="$saved_libs"
fi

# Warning if not
if test "x$have_opencl" = "xno" -a "x$enable_opencl" = "xyes"
then
  AC_MSG_WARN([
  
  The AMD OpenCL SDK is not presently installed in your system, as
  detected by the absence of the AMDAPPSDKROOT environment variable. The
  headers and libraries provided by this package are needed to build the
  CL workpool library and the benchmarks.
  ])
fi

# Conditional for Makefile.am
AM_CONDITIONAL(HAVE_OPENCL, test x$have_opencl = xyes)


##########
# Includes
##########

CL_WORKPOOL_INCLUDES=" \
\
-I\$(top_srcdir)/lib \
\
"

AC_SUBST(CL_WORKPOOL_INCLUDES)

###########
# Makefiles
###########
      
AC_CONFIG_FILES([
                       Makefile
                       lib/Makefile
                       src/Makefile
                       src/vecadd/Makefile
                       ])
AC_OUTPUT
