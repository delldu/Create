#!/usr/bin/env python
 
from gimpfu import gimp, register, main
 
def hello_world():
    gimp.message("Hello, GIMP world!\n")
 
register(
    "hello_world",
    'A simple Python-Fu "Hello, World" plug-in',
    'When run this plug-in prints "Hello, GIMP world!" in a dialog box.',
    "Tony Podlaski",
    "Tony Podlaski 2017. MIT License",
    "2017",
    "Hello World",
    "",
    [],
    [],
    hello_world,
    menu="<Image>/Filters/Samples",
)
 
main()
