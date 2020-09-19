TARGET := \
	dist/index.html \
	dist/index.js \
	dist/index.css

TS_SOURCE_FILES := \
	src/shape.ts \
	src/message.ts \
	src/progress.ts \
	src/canvas.ts \
	src/uploader.ts \
	src/downloader.ts \

#	src/workspace.ts

all: ${TARGET}

dist/index.js: src/index.ts
	tsc src/index.ts --outDir dist

src/index.ts: ${TS_SOURCE_FILES}
	cat $^ > $@

dist/index.html: src/index.html
	cp src/index.html dist/index.html

dist/index.css: src/index.css
	cp src/index.css dist/index.css

clean:
	rm -rf dist/*.js dist/*.html dist/*.css
	