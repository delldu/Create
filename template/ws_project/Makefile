TARGET := \
	dist/index.html \
	dist/index.js \
	dist/index.css

TS_SOURCE_FILES := \
	src/websocket.ts

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
	rm -rf dist/*.js dist/*.html dist/*.css


	
