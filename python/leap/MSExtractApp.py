
import overrides
import leap

from dlg.drop import BarrierAppDROP, BranchAppDrop, ContainerDROP
from dlg.meta import dlg_float_param, dlg_string_param
from dlg.meta import dlg_bool_param, dlg_int_param
from dlg.meta import dlg_component, dlg_batch_input
from dlg.meta import dlg_batch_output, dlg_streaming_input

##
# @brief MSExtractApp\n
# @details
# @par EAGLE_START
# @param gitrepo $(GIT_REPO)
# @param version $(PROJECT_VERSION)
# @param category PythonApp
# @param[in] param/appclass//String/readonly
#   \~English Application class\n
# @param[out]
# @par EAGLE_END
class MSExtractApp(BarrierAppDROP):
    """
    An extraction app drop that reads leap specific data
    from a measurement set.
    """

    component_meta = dlg_component('MSExtractApp', '',
                                   [dlg_batch_input('binary/*', [])],
                                   [dlg_batch_output('binary/*', [])],
                                   [dlg_streaming_input('binary/*')])

    @overrides
    def run(self):
        mspath = self.inputs[0]

        out_meta = self.outputs[0]
        out_uvws = self.outputs[1]
        out_vis = self.outputs[2]

        ms = leap.MeasurementSet(mspath)

        if out_meta:
            meta = leap.extract_meta(ms)
            out_meta.write(meta)
        
        if out_uvws:
            uvws = leap.extract_uvws(ms)
            out_uvws.write(uvws)

        if out_vis:
            vis = leap.extract_vis(ms)
            out_vis.write(vis)


