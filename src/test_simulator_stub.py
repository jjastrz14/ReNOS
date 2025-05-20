# the file should loop over all the files specified
# and execute them with the simulator stub to check
# if one of the files is not working properly

import simulator_stub as ss
from dirs import *

FILES_TO_RUN = ["config_files/dumps/dump{}.json".format(i) for i in range(0, 60)]

def test_files():
    # Create a SimulatorStub object
    stub = ss.SimulatorStub(EX_DIR)

    for file in FILES_TO_RUN:
        print("File: ", file)
        results, logger = stub.run_simulation(file, verbose = False)
        print("Results: ", results)
    
if __name__ == "__main__":
    test_files()