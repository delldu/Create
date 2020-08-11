#/************************************************************************************
#***
#***	Copyright 2020 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2020-08-11 19:52:12
#***
#************************************************************************************/
#

LIB_NAME := lib/libimage

# Setup environment
TORCH_INSTALL_DIR :=/usr/local/libtorch

INCS := -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++14 -I${TORCH_INSTALL_DIR}/include -Iinclude
LIBS := -Wl,--no-as-needed \
	-L${TORCH_INSTALL_DIR}/lib \
	-lc10 -lc10_cuda -ltorch -ltorch_cuda -ltorch_cpu \
 	-lopencv_core -lopencv_imgproc -lopencv_imgcodecs

# Source file
SOURCE :=  src/image.c

DEFINES :=

CXXFLAGS := -fPIC
LDFLAGS := -fPIC -O3

DEFINES := 
CFLAGS := -O2 -fPIC -Wall -Wextra
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}
OBJECTS := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: staticlib
# sharelib

sharelib: $(OBJECTS)
	$(LD) $(LDFLAGS) -shared -soname $(LIB_NAME).so -o $(LIB_NAME).so $(OBJECTS)


staticlib:$(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB_NAME).a $(OBJECTS)


#****************************************************************************
# Depend block
#****************************************************************************
depend:

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cpp
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


clean:
	rm -rf *.a *.so *.o $(OBJECTS)

