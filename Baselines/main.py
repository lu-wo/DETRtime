import sys
import time
import logging
from config import config, create_folder
from benchmark import benchmark


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

"""
Main entry of the program
Creates the logging files, loads the data and starts the benchmark.
All configurations (parameters) of this benchmark are specified in config.py
"""

def main():
    # Setting up logging
    create_folder()
    logging.basicConfig(filename=config['info_log'], level=logging.INFO)
    logging.info('Started the Logging')
    start_time = time.time()

    # For being able to see progress that some methods use verbose (for debugging purposes)
    f = open(config['model_dir'] + '/console.out', 'w')
    sys.stdout = Tee(sys.stdout, f)

    benchmark()

    logging.info("--- Runtime: %s seconds ---" % (time.time() - start_time))
    logging.info('Finished Logging')

if __name__=='__main__':
    main()
