
#
# 'make depend' uses makedepend to automatically generate dependencies
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'FeedforwardNT'
# 'make clean'  removes all .o and executable files
#
# $(error   VAR is $(VAR))
# $(warning VAR is $(VAR))
# $(info    VAR is $(VAR))

PLATEFORM = $(shell uname -s)

ifeq ($(PLATEFORM),Darwin)
	CC = clang
else
	CC = nvcc
endif

# Compile flags
ifeq ($(PLATEFORM),Darwin)
	CFLAGS = -Wall -Ofast
else
	CFLAGS = -O3 -Xptxas -v
endif

ifeq ($(PLATEFORM),Darwin)
	INCLUDES = -I../include -I/Applications/netcdf-c/include 
else
	INCLUDES = -I../include -I/usr/local/include 
endif

ifeq ($(PLATEFORM),Darwin)
	LFLAGS = -L/Applications/netcdf-c/lib -lnetcdf
else
	LFLAGS = -L/usr/local/lib -lnetcdf 
endif

ifeq ($(PLATEFORM),Darwin)
	LIBS = -lm
else
	LIBS = -lm
endif

ifeq ($(PLATEFORM),Darwin)
SRCS = $(shell echo *.c)
else
SRCS = $(shell echo *.cu)
endif
$(info Source files: $(SRCS))

# define the C object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
ifeq ($(PLATEFORM),Darwin)
OBJS = $(SRCS:.c=.o)	
else
OBJS = $(SRCS:.cu=.o)	
endif

ifeq ($(PLATEFORM),Linux)
.SUFFIXES:
.SUFFIXES: .cu .o
endif

# define the executable file
MAIN = wrfToNN

.PHONY: depend clean

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

ifeq ($(PLATEFORM),Darwin)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@
else
.cu.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@
endif

clean:
	$(RM) *.o *~ $(MAIN) Makefile.bak

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it

gpu.o: /usr/include/stdio.h ../include/gpu.h
interpolate.o: ../include/interpolate.h /usr/include/math.h
interpolate.o: ../include/memory.h /usr/include/stdlib.h /usr/include/stdio.h
interpolate.o: /usr/include/string.h
main.o: /usr/include/dirent.h /usr/include/features.h
main.o: /usr/include/stdc-predef.h /usr/include/unistd.h
main.o: /usr/local/include/netcdf.h /usr/include/errno.h
main.o: ../include/interpolate.h /usr/include/math.h ../include/memory.h
main.o: /usr/include/stdlib.h /usr/include/stdio.h /usr/include/string.h
memory.o: ../include/memory.h /usr/include/stdlib.h /usr/include/stdio.h
memory.o: /usr/include/string.h
