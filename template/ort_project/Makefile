#/************************************************************************************
#***
#***	Copyright 2021 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2021-01-12 23:53:43
#***
#************************************************************************************/
#

ONNXRUNTIME_INSTALL_DIR := /opt/onnxruntime

TARGET := demo

SOURCE :=  \
	engine.c \
	main.c

INCS	:=  -I$(ONNXRUNTIME_INSTALL_DIR)/include
	#  -I$(BUILD_DIR)/include
LIBS	:= -L$(ONNXRUNTIME_INSTALL_DIR)/lib -lonnxruntime
	#-L$(BUILD_DIR)/lib -lstd... -lstdc++

DEFINES := -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++14 

CXXFLAGS := -fPIC -Wall -Wextra
LDFLAGS := -fPIC -O3


#****************************************************************************
# Makefile code common to all platforms
#****************************************************************************
CXXFLAGS := ${CXXFLAGS} ${DEFINES} ${INCS}

OBJECT := $(addsuffix .o,$(basename ${SOURCE}))

#****************************************************************************
# Compile block
#****************************************************************************
all: ${TARGET}

${TARGET}: ${OBJECT}
	${CXX} ${LDFLAGS} -o ${TARGET} ${OBJECT} ${LIBS}

install:
	install -m 755 ${TARGET} ${INSTALL_DIR}/bin

#****************************************************************************
# common rules
#****************************************************************************
%.o : %.c
	${CXX} ${CXXFLAGS} ${INCS} -c $< -o $@

#****************************************************************************
# Depend block
#****************************************************************************
depend:

clean:
	rm -f core ${OBJECT} ${TARGET}
