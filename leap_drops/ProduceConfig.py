import csv
import json
import math

from dlg.droputils import DROPFile
from dlg.drop import BarrierAppDROP
from dlg.meta import dlg_int_param, dlg_float_param, dlg_string_param, \
    dlg_component, dlg_batch_input, dlg_batch_output, dlg_streaming_input
from dlg.droputils import DROPFile


## Produce Config
# @brief Produce Config
# @details A BarrierAppDrop that produces multiple config files suitable for the CallLeap BarrierAppDrop
# @par EAGLE_START
# @param gitrepo $(GIT_REPO)
# @param version $(PROJECT_VERSION)
# @param category PythonApp
# @param[in] param/number_of_stations/1/Integer/readwrite
#     \~English The number of stations from the measurement set that should be processed\n
#     \~Chinese \n
#     \~
# @param[in] param/implementation/cpu/String/readwrite
#     \~English The implementation of the LEAP algorithm to use (cpu, casa, cuda)\n
#     \~Chinese \n
#     \~
# @param[in] param/auto_correlation/false/string/readwrite
#     \~English Enable auto correlation in the LEAP algorithm\n
#     \~Chinese \n
#     \~
# @param[in] param/appclass/leap_nodes.ProduceConfig.ProduceConfig/String/readonly
#     \~English The path to the class that implements this app\n
#     \~Chinese \n
#     \~
# @param[in] port/Directions
#     \~English A CSV file containing directions for calibration
#     \~Chinese \n
#     \~
# @param[out] port/Config
#     \~English A JSON config containing the specification for running an instance of LeapAccelerateCLI
#     \~Chinese \n
#     \~
# @par EAGLE_END

class ProduceConfig(BarrierAppDROP):
    """A BarrierAppDrop that produces multiple config files suitable for the CallLeap BarrierAppDrop"""
    compontent_meta = dlg_component('ProduceConfig', 'Produce Config.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])

    # read component parameters
    numStations = dlg_int_param('number of stations', 1)
    implementation = dlg_string_param('implementation', 'cpu')
    autoCorrelation = dlg_string_param('auto correlation', 'false')


    def initialize(self, **kwargs):
        super(ProduceConfig, self).initialize(**kwargs)


    def run(self):
        # check number of inputs and outputs
        if len(self.inputs) != 1:
            raise Exception("One input is expected by this application")

        # read directions from input 0
        directions = self._readDirections(self.inputs[0])

        # determine number of directions per instance
        numDirectionsPerInstance = float(len(directions)) / float(len(self.outputs))

        startDirectionIndex = 0
        endDirectionIndex = 0

        # split directions
        for i in range(len(self.outputs)):
            endDirectionIndex = int(math.floor((i+1)*numDirectionsPerInstance))

            # split directions
            partDirections = directions[startDirectionIndex:endDirectionIndex]

            # build config
            configJSON = self._createConfig(self.numStations, partDirections, self.implementation, self.autoCorrelation)

            # stringify config
            config = json.dumps(configJSON)

            # write config to output
            self.outputs[i].write(config)

            # continue from here in the next iteration
            startDirectionIndex = endDirectionIndex


    def _readDirections(self, inDrop):
        directions = []

        # NOTE: it appears csv.reader() can't use the DROPFile(inDrop) directly,
        #       since DROPFile is not a iterator. Instead, we read the whole
        #       inDrop to a string and pass that to csv.reader()
        with DROPFile(inDrop) as f:
            file_data = f.read()
            csvreader = csv.reader(file_data.split('\n'))
            for row in csvreader:
                # skip rows with incorrect number of values
                if len(row) is not 2:
                    continue

                x = float(row[0])
                y = float(row[1])
                directions.append([x,y])

        return directions


    def _createConfig(self, numStations, directions, implementation, autoCorrelation):
        return {
            'numStations': numStations,
            'directions': directions,
            'implementation': implementation,
            'autoCorrelation': autoCorrelation
        }
