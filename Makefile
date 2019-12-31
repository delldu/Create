#/************************************************************************************
#***
#***	Copyright 2019 Dell(18588220928g@163.com), All Rights Reserved.
#***
#***	File Author: Dell, 2019-12-13 22:33:17
#***
#************************************************************************************/
#
INSTALL_DIR ?= "/usr/local/bin"

XSUBDIRS :=  \
	source

all:
	@for d in $(XSUBDIRS)  ; do \
		if [ -d $$d ] ; then \
			$(MAKE) -C $$d || exit 1; \
		fi \
	done	

install:
	@for d in $(XSUBDIRS)  ; do \
		if [ -d $$d ] ; then \
			$(MAKE) -C $$d install || exit 1; \
		fi \
	done	
	@echo "Install create to "$(INSTALL_DIR) "...... "
	@cp create $(INSTALL_DIR)
	@chmod +x $(INSTALL_DIR)/create
	@cp template $(INSTALL_DIR) -R

clean:
	@for d in $(XSUBDIRS) ; do \
		if [ -d $$d ] ; then \
			$(MAKE) -C $$d clean || exit 1; \
		fi \
	done	
