#/************************************************************************************
#***
#***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2021-01-25 15:25:52
#***
#************************************************************************************/
#
CUDA_INSTALL_DIR := /usr/local/cuda

TARGET := onnx2trt

SOURCE := src/main.cc

INCS	:= -I$(CUDA_INSTALL_DIR)/include
	#  -I$(BUILD_DIR)/include
LIBS	:= \
	-L$(CUDA_INSTALL_DIR)/lib64 -lcudart_static \
	-lnvinfer -lnvonnxparser \
	-ldl -lrt \
 	-lpthread
 	
# 	-lcublas -lcudnn \

DEFINES := -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -std=gnu++14

CFLAGS := -O2 -fPIC -Wall -Wextra
LDFLAGS := -fPIC

#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CFLAGS   := ${CFLAGS} ${DEFINES}
CXXFLAGS := ${CXXFLAGS} ${DEFINES}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: ${TARGET}

${TARGET}: ${OBJECT}
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	${STRIP} ${TARGET}
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.cc
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@


#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}
