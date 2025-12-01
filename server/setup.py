from setuptools import setup, Extension
import sys

if sys.platform == 'darwin':
    module = Extension(
        'fast_capture',
        sources=['fast_capture.cpp'],
        extra_link_args=['-framework', 'OpenGL'],
        extra_compile_args=['-O3', '-DGL_SILENCE_DEPRECATION']
    )
else:
    module = Extension(
        'fast_capture',
        sources=['fast_capture.cpp'],
        libraries=['GL'],
        extra_compile_args=['-O3']
    )

setup(
    name='fast_capture',
    version='1.0',
    ext_modules=[module]
)