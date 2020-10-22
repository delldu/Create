TARGET := \
	dist/matter.html \
	dist/image.html \
	dist/matter.js \
	dist/matter.css

TS_SOURCE_FILES := \
	src/canvas.ts \
	src/image.ts \
	src/message.ts \
	src/progress.ts \
	src/keyboard.ts \
	src/mouse.ts \
	src/panel.ts \
	src/uploader.ts \
	src/download.ts \

#	src/workspace.ts

all: ${TARGET}

dist/matter.js: src/matter.ts
	tsc src/matter.ts -lib es2015,dom --strict --removeComments --outDir dist

src/matter.ts: ${TS_SOURCE_FILES}
	cat $^ > $@

dist/image.html: src/image.html
	cp src/image.html dist/image.html

dist/matter.html: src/matter.html
	cp src/matter.html dist/matter.html

dist/matter.css: src/matter.css
	cp src/matter.css dist/matter.css

clean:
	rm -rf src/matter.ts dist/*.js dist/*.html dist/*.css
