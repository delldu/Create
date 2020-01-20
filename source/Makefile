INSTALL_DIR ?= "/usr/local/bin"

all:
	go build main.go
	@mv main create_temp
	@strip create_temp


install:
	@echo "Install create_temp to "$(INSTALL_DIR) "...... "
	@cp create_temp $(INSTALL_DIR)
	@cp create_help $(INSTALL_DIR)
	@chmod +x $(INSTALL_DIR)/create_temp

clean:
	rm create_temp
