import json
import random
import subprocess
import time

# should be read from DALiuGE
CONFIG_FILENAME = "config.json"

DEBUG = True
DEBUG_OUTPUT = "DEBUG OUTPUT"

def readConfig(filename):
    with open(CONFIG_FILENAME) as json_file:
        config = json.load(json_file)
    return config


def main():
    config = readConfig(CONFIG_FILENAME)
    #print(config)

    # build command line
    commandLine = ['LeapAccelerateCLI', '-f', config['filePath'], '-s', str(config['numStations']), '-d', str(config['directions'])]
    #print(str(commandLine))

    if DEBUG:
        time.sleep(random.uniform(5,10))
        print(DEBUG_OUTPUT)
    else:
        # call leap
        process = subprocess.call(commandLine)

if __name__ == "__main__":
    main()
