import csv
import json

from dlg import droputils
from dlg.drop import BarrierAppDROP
from dlg.meta import dlg_int_param, dlg_float_param, dlg_string_param, \
    dlg_component, dlg_batch_input, dlg_batch_output, dlg_streaming_input


class ProduceConfig(BarrierAppDROP):
    """A BarrierAppDrop that produces multiple config files suitable for the CallLeap BarrierAppDrop"""
    compontent_meta = dlg_component('ProduceConfig', 'Produce Config.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])

    numCopies = dlg_int_param('number of copies', 1)
    numStations = dlg_int_param('number of stations', 1)
    directionsFilename = dlg_string_param('directions', '')
    measurementSetFilename = dlg_string_param('measurementSet', '')
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
        directions = _readDirections()

        for i in range(numCopies):
            config = _createConfig(numStations, directions, implementation)
            print(json.dumps(config))


    def _readDirections():
        directions = []

        with open(directionsFilename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                x = float(row[0])
                y = float(row[1])
                directions.append([x,y])

        return directions


    def _createConfig(numStations, directions, implementation):
        return {
            'numStations': numStations,
            'filePath': measurementSetFilename,
            'directions': directions,
            'implementation': implementation
        }
