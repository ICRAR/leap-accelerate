import json
import random
import subprocess
import time

from dlg import droputils
from dlg.drop import BarrierAppDROP
from dlg.meta import dlg_int_param, dlg_float_param, dlg_string_param, \
    dlg_component, dlg_batch_input, dlg_batch_output, dlg_streaming_input

class CallLeap(BarrierAppDROP):
    """A BarrierAppDrop that reads a config file, generates a command line for the LeapAccelerateCLI application, and then executes the application"""
    compontent_meta = dlg_component('Call Leap', 'Call Leap.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])

    configFilename = dlg_string_param('config', '')

    # should be read from DALiuGE
    #CONFIG_FILENAME = "config.json"

    DEBUG = True
    DEBUG_OUTPUT = "DEBUG OUTPUT"


    def initialize(self, **kwargs):
        super(ProduceConfig, self).initialize(**kwargs)


    def run(self):
        config = _readConfig(configFilename)
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


    def _readConfig(filename):
        with open(configFilename) as json_file:
            config = json.load(json_file)
        return config
