INSTALL_DIR ?= "/usr/local/bin"

all:
	go build main.go
	@mv main create_view
	@strip create_view


install:
	@echo "Install create_view to "$(INSTALL_DIR) "...... "
	@cp create_view $(INSTALL_DIR)
	@cp create_help $(INSTALL_DIR)
	@chmod +x $(INSTALL_DIR)/create_view

clean:
	rm create_view
