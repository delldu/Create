TARGET := \
	dist/index.html \
	dist/index.js \
	dist/index.css

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

dist/index.js: src/index.ts
	tsc src/index.ts -lib es2015,dom --strict --removeComments --outDir dist

src/index.ts: ${TS_SOURCE_FILES}
	cat $^ > $@

dist/index.html: src/index.html
	cp src/index.html dist/index.html

dist/index.css: src/index.css
	cp src/index.css dist/index.css

clean:
	rm -rf src/index.ts dist/*.js dist/*.html dist/*.css


test:
	mkdir -p dist/upload
	cp src/image.html dist
	cp src/image.css dist
	cp src/image.py dist
