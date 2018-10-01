### Notes on Pytorch
Source files and official installation instruction can be found at: https://github.com/pytorch/pytorch#from-source.  
Pay attention to python version, i.e. run "python3 setup.py install".  
pytorch_tut.py contains code that can be used as reference when coding in Pytorch.

### Notes on TensorFlow installation on Jetson
Prebuilt wheel files are available at https://devtalk.nvidia.com/default/topic/1031300/jetson-tx2/tensorflow-1-9-rc-wheel-with-jetpack-3-2-/4.  
Use pip to install:  
pip install [downloaded wheel file]  
pip3 install [downloaded wheel file]  

### Notes on C++
When compiling the program, the compiler needs the header files to compile the source code; the linker needs the libraries to resolve external references from other object files or libraries. The compiler and linker will not find the header/libraries unless you set the appropriate options.  
#### Libraries
A library is a collection of pre-compiled object files that can be linked into your programs via the linker.  
A static library has file extension ".a". When a program is linked against a static library, the machine code of external functions used is copied into the executable.  
A shared library has file extension "so". When a program is linked against a shared library, only a small table is created in the executable. Before the execuyable starts running, the OS loads the machine code needed for the external functions - a process known as dynamic linking. Dynamic linking makes execuyable files smaller and saved disk space, because ont copy of a library can be shared between multiple programs. Futhermore, most OS allows one copy of a shared library in memory to be used by all running programs, thus, saving memory.  
The linker searches the library paths for libraries needed to link the program into an executable. The library path is specified via -L \<dir\> option, or the environment variable LIBRARY_PATH. 

#### Header files
Header files typically contain only declarations not definitions/implementations. They relieve us of the burden of writing forward declaration for every function we use that lives in another file.  
Use of a header file in the source code is done through #include <filename> or #include "filename". The angled brackets tell the compiler that the header file is included with the compiler, so it should look for that header file in the standard system directories. The double-quotes tell the compiler that the header file is supplied by us, so it should look for that header file in the current directory containing the source code first. If it doesn’t find the header file there, it will check other specified include path, and then the standard system directories.
The additional include paths are specified via -I \<dir\> option, or the environment variable CPATH.  
Header guards are important because they prevent a given header file from being #included more than once in the same file.   

### Notes on CMake
* CMake variables are not environmental variables (unlike Makefile).  
* CMake is not a build system, it is a build system generator. CMake is all about targets and properties.
* Forget about those commands operating on the directory level (add_compile_option(), include_directories(), link_directories(), link_libraries()), because they affect all targets under the directory. It is much easier to stay at the target level.
* Use add_excutable(exe_name source_file) OR add_library(exe_name source_file) to declare modules.
* Use target_link_libraries(exe_name pack_name)to declare dependencies. Dependencies can be PUBLIC/transitve, PRIVATE(non-transitive) or INTERFACE (nothing to build). Header-only libraries can be declared as a INTERFACE library.
* Use find_package(pack_name version_num REQUIRED) to add external libraries, then target_link_libraries(exe_name pack_name) to link to them.
* Use a FindModule for third-party libraries that do not support clients to use CMake.
* Use add_subdirectory() to build hierarchies of your project. Subdirectories should have their own CMakeLists.txt files.
