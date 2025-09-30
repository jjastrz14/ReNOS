from setuptools import setup, Extension
import pybind11, glob, os, sys

# BookSim2 (nocsim) extension
cpp_files = [
    f for f in glob.glob('src/restart/**/*.cpp', recursive=True)
    if not os.path.basename(f) == 'main.cpp'
]

include_dirs = ['include', 'src/restart/include', pybind11.get_include()]

for root, dirs, files in os.walk('src/restart/src'):
    include_dirs.append(root)

# Analytical simulator extension
analytical_cpp_files = [
    f for f in glob.glob('src/analytical/**/*.cpp', recursive=True)
    if not os.path.basename(f) == 'main.cpp'
]

analytical_include_dirs = ['src/analytical/include', 'src/analytical/src', pybind11.get_include()]

for root, dirs, files in os.walk('src/analytical/src'):
    analytical_include_dirs.append(root)

# Fast analytical simulator extension
fast_analytical_cpp_files = [
    f for f in glob.glob('src/fast_analytical_model/**/*.cpp', recursive=True)
    if not os.path.basename(f) == 'main.cpp'
]

fast_analytical_include_dirs = ['src/fast_analytical_model', 'src/restart/include', pybind11.get_include()]

ext_modules = [
    # Original BookSim2 simulator
    Extension(
        'nocsim',
        sources=cpp_files,
        include_dirs=include_dirs + ['/opt/homebrew/Cellar/nlohmann-json/3.11.3/include'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
    # Analytical simulator
    Extension(
        'analytical_nocsim',
        sources=analytical_cpp_files,
        include_dirs=analytical_include_dirs + ['/opt/homebrew/Cellar/nlohmann-json/3.11.3/include'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
    # Fast analytical simulator
    Extension(
        'fast_nocsim',
        sources=fast_analytical_cpp_files,
        include_dirs=fast_analytical_include_dirs + ['/opt/homebrew/Cellar/nlohmann-json/3.11.3/include'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    )
]

setup(
    name='renos-simulators',
    version='1.3.0',
    description='ReNOS: NoC simulators with cycle-accurate (BookSim2), analytical, and fast analytical models',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.6'],
    install_requires=['pybind11>=2.6'],
    zip_safe=False,
)