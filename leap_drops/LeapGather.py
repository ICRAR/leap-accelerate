import json

from dlg.droputils import DROPFile
from dlg.drop import BarrierAppDROP
from dlg.meta import dlg_int_param, dlg_float_param, dlg_string_param, \
    dlg_component, dlg_batch_input, dlg_batch_output, dlg_streaming_input

## Leap Gather
# @brief Leap Gather
# @details A BarrierAppDrop that gathers output from multiple instances of the LeapAccelerateCLI application, sorts it, and outputs it
# @par EAGLE_START
# @param gitrepo $(GIT_REPO)
# @param version $(PROJECT_VERSION)
# @param category PythonApp
# @param[in] port/Result
#     \~English The JSON output from an instance of LeapAccelerateCLI
#     \~Chinese 要读取的起始频率\n
#     \~
# @param[out] port/Result
#     \~English The combined output from many instances the LeapAccelerateCLI application (JSON)
#     \~Chinese 要读取的起始频率\n
#     \~
# @par EAGLE_END

class LeapGather(BarrierAppDROP):
    """A BarrierAppDrop that gathers output from multiple instances of the LeapAccelerateCLI application, sorts it, and outputs it"""
    compontent_meta = dlg_component('Leap Gather', 'Leap Gather.',
                                    [dlg_batch_input('binary/*', [])],
                                    [dlg_batch_output('binary/*', [])],
                                    [dlg_streaming_input('binary/*')])


    def initialize(self, **kwargs):
        super(LeapGather, self).initialize(**kwargs)


    def run(self):
        # check number of outputs
        if len(self.outputs) != 1:
            raise Exception("One output is expected by this application")

        # read from all inputs
        inputs = []
        for i in range(len(self.inputs)):
            with DROPFile(self.inputs[i]) as f:
                file_data = f.read()
                inputs.append(json.loads(file_data))

        # write to output
        self.outputs[0].write(json.dumps(inputs))
