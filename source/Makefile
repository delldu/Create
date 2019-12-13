INSTALL_DIR ?= "/usr/local/bin"

all:
	go build main.go
	mv main create_from_template
	strip create_from_template


install:
	@echo "Install create_from_template to "$(INSTALL_DIR) "...... "
	@cp create_from_template $(INSTALL_DIR)
	@chmod +x $(INSTALL_DIR)/create_from_template

clean:
	rm create_from_template
