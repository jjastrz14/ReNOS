from setuptools import setup, Extension
import pybind11, glob, os

cpp_files = [
    f for f in glob.glob('src/restart/**/*.cpp', recursive=True)
    if not os.path.basename(f) == 'main.cpp'
]

include_dirs = ['include', 'src/restart/include', pybind11.get_include()]

for root, dirs, files in os.walk('src/restart/src'):
    include_dirs.append(root)


ext_modules = [
    Extension(
        'nocsim',
        sources=cpp_files,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++17'],
    )
]

setup(
    name='nocsim',
    version='1.1.0',
    description='A NoC simulator with C++ core and pybind11 bindings',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6'],
    install_requires=['pybind11>=2.6'],
    zip_safe=False,
)
