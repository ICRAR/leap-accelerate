import csv
import json

from dlg import droputils
from dlg.drop import BarrierAppDROP
from dlg.meta import dlg_int_param, dlg_float_param, dlg_string_param, \
    dlg_component, dlg_batch_input, dlg_batch_output, dlg_streaming_input
from dlg.droputils import DROPFile


class ProduceConfig(BarrierAppDROP):
    """A BarrierAppDrop that produces multiple config files suitable for the CallLeap BarrierAppDrop"""
    compontent_meta = dlg_component('ProduceConfig', 'Produce Config.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])

    numCopies = dlg_int_param('number of copies', 1)
    numStations = dlg_int_param('number of stations', 1)
    implementation = dlg_string_param('eigen', '')

    # should be read from DALiuGE
    #NUMBER_OF_COPIES = 1
    #NUM_STATIONS = 126
    #DIRECTIONS_FILENAME = "directions.csv"
    #MEASUREMENT_SET_FILENAME = "/Users/james/working/leap-accelerate/testdata/1197638568-32.ms"
    #IMPLEMENTATION = 'eigen'


    def initialize(self, **kwargs):
        super(ProduceConfig, self).initialize(**kwargs)


    def run(self):
        # check number of inputs and outputs
        if len(self.outputs) != 1:
            raise Exception("One output is expected by this application")
        if len(self.inputs) != 1:
            raise Exception("One input is expected by this application")

        # read directions from input 0
        directions = _readDirections(self.inputs[0])

        # split directions
        for i in range(numCopies):
            # TODO: actually split directions according to num copies, currently
            #       we send all directions to all configs
            partDirections = directions

            # build config
            configJSON = _createConfig(numStations, partDirections, implementation)
            config = json.dumps(configJSON)

            # write config to output
            self.outputs[0].write(config)


    def _readDirections(inDrop):
        directions = []

        with DROPFile(inDrop) as f:
            csvreader = csv.reader(f, delimiter=',')
            for row in csvreader:
                x = float(row[0])
                y = float(row[1])
                directions.append([x,y])

        return directions


    def _createConfig(numStations, directions, implementation):
        return {
            'numStations': numStations,
            'directions': directions,
            'implementation': implementation
        }
