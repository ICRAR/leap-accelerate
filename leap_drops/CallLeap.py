import json
import os
import random
import subprocess
import time

from dlg.droputils import DROPFile
from dlg.drop import BarrierAppDROP
from dlg.meta import dlg_int_param, dlg_float_param, dlg_string_param, \
    dlg_component, dlg_batch_input, dlg_batch_output, dlg_streaming_input

class CallLeap(BarrierAppDROP):
    """A BarrierAppDrop that reads a config file, generates a command line for the LeapAccelerateCLI application, and then executes the application"""
    compontent_meta = dlg_component('Call Leap', 'Call Leap.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])

    measurementSetFilename = dlg_string_param('measurementSetFilename', '')

    DEBUG = True
    DEBUG_OUTPUT = "DEBUG OUTPUT"


    def initialize(self, **kwargs):
        super(ProduceConfig, self).initialize(**kwargs)


    def run(self):
        # check number of inputs and outputs
        if len(self.outputs) != 1:
            raise Exception("One output is expected by this application")
        if len(self.inputs) != 1:
            raise Exception("One input is expected by this application")

        # check that measurement set file exists
        if not os.path.isfile(measurementSetFilename):
            raise Exception("Could not find measurement set file:" + measurementSetFilename)

        # read config from input
        config = self._readConfig(self.inputs[0])
        #print(config)

        # build command line
        commandLine = ['LeapAccelerateCLI', '-f', measurementSetFilename, '-s', str(config['numStations']), '-d', str(config['directions'])]
        #print(str(commandLine))

        if DEBUG:
            time.sleep(random.uniform(5,10))
            self.outputs[0].write(DEBUG_OUTPUT)
        else:
            # call leap
            result = subprocess.run(commandLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.outputs[0].write(result.stdout)

    def _readConfig(self, inDrop):
        with DROPFile(inDrop) as f:
            config = json.load(f)
        return config
