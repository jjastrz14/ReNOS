
import sys
from dirs import PYTHON_MODULE_DIR
#import sysconfig
#print(sysconfig.get_config_vars())
import platform
import subprocess
import sysconfig

def get_comprehensive_compiler_info():
    print("Comprehensive Compiler Diagnostics")
    print("-" * 40)
    
    # Python details
    print("\nPython Details:")
    print("Executable:", sys.executable)
    print("Version:", sys.version)
    print("Compiler:", platform.python_compiler())
    
    # Compilation details
    print("\nCompilation Details:")
    print("Compiler:", sysconfig.get_config_var('CC'))
    print("Compiler Flags:", sysconfig.get_config_var('CFLAGS'))
    
    # System compiler checks
    print("\nSystem Compiler Versions:")
    try:
        # Clang version
        clang_version = subprocess.check_output(['clang', '--version'], 
                                                universal_newlines=True)
        print("Clang:", clang_version.split('\n')[0])
        
        # GCC version (if installed)
        try:
            gcc_version = subprocess.check_output(['gcc', '--version'], 
                                                  universal_newlines=True)
            print("GCC:", gcc_version.split('\n')[0])
        except:
            print("GCC not found in system path")
    except Exception as e:
        print("Could not retrieve system compiler versions:", str(e))
    
    # Additional platform info
    print("\nPlatform Details:")
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    print("Processor:", platform.processor())

if __name__ == "__main__":
    get_comprehensive_compiler_info()

    sys.path.append(PYTHON_MODULE_DIR)
    print("Just before import nocsim")
    import nocsim 
    print("Imported successfully")
