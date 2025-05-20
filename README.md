# <span style="color:sandybrown"><b>Restart: an extension of Booksim 2.0, the Cycle accurate NoC simulator to host trace-like simulations.</b></span>

This is a simple extension of Booksim 2.0, the cycle accurate NoC simulator, to include the support for trace-like simulations. 
Booksim itself is an extremely flexible simulator: as the original authors of the code themselves specify, "most of the simulator’s components are designed to be modular so tasks such as adding a new routing algorithm, topology, or router microarchitecture should not require a complete redesign of the code".\
For these reasons, as well as the fact that the components of the simulated NoC are threated at a somewhat higher level of abstraction than other simulators, Booksim represents a good starting point for the main goal of this project.

## <b> 1.  <span style="color:sandybrown">Main objective of the code</b></span>

The primary goal is to implement a NoC simulator to use for processing application-specific workloads, in order to estimante the required resources - both in terms of power consumption and latency - for the correct and opimized execution of said application on a NoC-based architecture. The idea is to use the estimates produced by the simulation as data for optimization of the mapping of the application's tasks to the PEs represented by the nodes of the NoC

## <b> 2. <span style="color:sandybrown">Running the code </span> </b>

###  2.1.  How to compile C++ library for UNIX systems
The code can be complied with the following comands, in the main directory of the project:
```bash

$ mkdir build
$ cd build
$ cmake ..
$ make -j4

``` 
To clean the build directory, use the following command:
```bash
$ make clean
```

###  2.2.  How to install python dependencies
We recommand using python version == 3.11 and having intalled setuptools in your base environment

#### Interactive mode 
```python 
python install.py
```
#### Create a new environment
```python 
python install.py --create-env --env-name renos --python-version 3.11
```
#### Install in existing environment using conda
```python 
python install.py --env-name renos
```
#### Install using pip in an existing environment
```python 
python install.py --env-name renos --use-pip
```
# Install using pip without conda
```python
python install.py --use-pip
```

after installation activate created environment

```bash
$ conda activate name_of_environment
```

###  2.3.  Install NoC simulator as python library

if you are using MacOS it is recomended to build NoC simulator as python library using (run this is source library of ReNOS): 


```python
python -m pip install .
```

Unfortunelty software is not avaialbe for Windows

### 2.2. Dependencies

Two additional requirements are needed to run the code:

- <a href=https://github.com/catchorg/Catch2>Catch2</a>: a unit testing framework for C++ , which is included in the `test` directory of the project. The library is not needed for the main code to run. The compilation will still give errors if the library is not found: to avoid this, the `CMakeLists.txt` file should be modified to remove the lines that include the testing directory.

- <a href=https://github.com/nlohmann/json>JSON for Modern C++</a>: a JSON, header-only parser for C++ to allow the parsing of the configuration file (differently from the original Booksim, which defined a custom format for the configuration `.txt` file).

Both of these libraries can be installed using the most common package managers, such as `apt` for Ubuntu or `brew` for MacOS.
