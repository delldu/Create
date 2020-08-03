INSTALL_DIR ?= "/usr/local/bin"

all:
	go build ni.go
	go build nv.go


install:
	@echo "Install create_view to "$(INSTALL_DIR) "...... "
	@cp ni $(INSTALL_DIR)
	@chmod +x $(INSTALL_DIR)/ni

clean:
	rm -rf ni nv
