"""Setup."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright {{create "author"}} {{create "date +%Y"}}, All Rights Reserved.
# ***
# ***    File Author: {{ create "author" }}, {{ bash "date" }}
# ***
# ************************************************************************************/
#

import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
     name='{{ . }}',  
     version='0.0.1',
     author='{{ create "author" }}',
     author_email="",
     description="{{ . }}",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url='https://github.com/{{ create "whoami" }}/{{ . }}',
     packages=['{{ . }}'],
     package_data={'{{ . }}': ['weights/*.pth']},
     include_package_data=True,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )
