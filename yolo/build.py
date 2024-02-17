import argparse
import onnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name

from qonnx.transformation.base import Transformation
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.general import GiveUniqueNodeNames, RemoveUnusedTensors, SortGraph, GiveReadableTensorNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.remove import RemoveIdentityOps, remove_node_and_rewire
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import verify_step
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import os
import shutil
import numpy as np

from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline import Streamline, absorb, reorder, collapse_repeated as collapse, round_thresholds as round

#Delete previous run results if exist
# if os.path.exists(estimates_output_dir):
#     shutil.rmtree(estimates_output_dir)
#     print("Previous run results deleted!")

class MoveOpPastSPFF(Transformation):
    def apply(self, model):
        model, c1 = reorder.MoveMulPastFork().apply(model)
        model, c2 = reorder.MoveMulPastMaxPool().apply(model)
        model, c3 = reorder.MoveTransposePastFork().apply(model)
        model, c4 = reorder.MakeMaxPoolNHWC().apply(model)
        return model, any([c1, c2, c3, c4])
    
class MoveMulPastSPFF(Transformation):
    def apply(self, model):
        model, c1 = reorder.MoveMulPastFork().apply(model)
        model, c2 = reorder.MoveMulPastMaxPool().apply(model)
        return model, any([c1, c2])
    
class MoveTranspose(Transformation):
    def apply(self, model):
        model, c1 = reorder.MoveTransposePastFork().apply(model)
        model, c2 = reorder.MakeMaxPoolNHWC().apply(model)
        model, c3 = absorb.AbsorbTransposeIntoResize().apply(model)
        model, c4 = absorb.AbsorbTransposeIntoMultiThreshold().apply(model)
        return model, any([c1, c2, c3, c4])
    
class RemoveUnusedOuputs(Transformation):
    def apply(self, model):
        empty = [out for out in model.graph.output if model.find_producer(out.name) is None]
        for out in empty:
            model.graph.output.remove(out)
        return model, False

class MoveIdenticalOpPastConcat(Transformation):
    """
    Move identical operations on different branches past the common join node.
    This transformation assumes that the identical operations only change the
    data layout. For linear operations, see the transformation MoveLinearPastEltwiseAdd.
    Specifically, this transformation matches and transforms the following patterns:
    f(x) + f(y) -> f(x + y)
    where f(.) is currently only supporting 'Transpose', and an 'Add' node is
    the join node.
    """

    def __init__(self, identical_op_list):
        super().__init__()
        self.ops_to_move = identical_op_list

    def move_node(self, model: ModelWrapper, n: onnx.NodeProto, producers: list[onnx.NodeProto], prod_type: str):
        prod_inputs = [p.input[0] for p in producers]
        cat_in0 = n.input[0]
        cat_out = n.output[0]

        # Connect inputs of producers directly to inputs of concat
        for i in range(len(n.input)):
            n.input[i] = prod_inputs[i]
        
        # Reuse first input tensor as output
        if prod_type == "Transpose":
            perm = get_by_name(producers[0].attribute, "perm").ints
            new_shape = model.get_tensor_shape(cat_out)
            new_shape = [new_shape[i] for i in np.argsort(perm)]
            model.set_tensor_shape(tensor_name=cat_in0, tensor_shape=new_shape)
            
            axis = get_by_name(n.attribute, "axis")
            axis.i = perm[axis.i]
        else:
            new_shape = model.get_tensor_shape(cat_out)
            model.set_tensor_shape(tensor_name=cat_in0, tensor_shape=new_shape)

        n.output[0] = cat_in0
        producers[0].input[0] = cat_in0
        producers[0].output[0] = cat_out

        for i in range(1, len(producers)):
            model.graph.node.remove(producers[i])

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Concat" and model.is_join_node(n):
                if any(i is None for i in n.input):
                    continue

                producers = model.find_direct_predecessors(n)
                p0type = producers[0].op_type

                identical_op = all(p.op_type == p0type for p in producers)

                if identical_op and p0type in self.ops_to_move:
                    self.move_node(model, n, producers, p0type)
                    graph_modified = True

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)
            model = model.transform(InferDataLayouts(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)

class MoveTransposePastReshape(Transformation):
    def apply(self, model: ModelWrapper):
        graph_modified = False
        for n in model.graph.node:
            if n.op_type == "Reshape":
                if n.input[0] is None:
                    continue
                prod = model.find_producer(n.input[0])
                if prod is None or prod.op_type != "Transpose":
                    continue
                perm = get_by_name(prod.attribute, "perm")
                shape = model.get_initializer(n.input[1])
                if shape is None or perm is None or len(shape) != len(perm.ints):
                    continue

                inp = prod.input[0]
                mid = n.input[0]
                out = n.output[0]
                inp_shape = model.get_tensor_shape(inp)

                # Update Reshape with new transposed shape
                perm_inv = np.argsort(perm.ints)
                new_shape = np.asarray([shape[i] for i in perm_inv])
                model.set_initializer(n.input[1], new_shape)
                # Update shape between nodes
                new_mid_shape = list(np.zeros(inp_shape).reshape(new_shape).shape)
                model.set_tensor_shape(mid, new_mid_shape)
                # Rewire nodes
                n.input[0] = inp
                n.output[0] = mid
                prod.input[0] = mid
                prod.output[0] = out

        if graph_modified:
            model = model.transform(SortGraph(), make_deepcopy=False, cleanup=False)
            model = model.transform(InferDataLayouts(), make_deepcopy=False, cleanup=False)

        return (model, graph_modified)
        return (model, False)

class RemoveReshape(Transformation):
    def apply(self, model: ModelWrapper):
        for n in model.graph.node:
            if n.op_type == "Reshape":
                producer = model.find_producer(n.input[0])
                if producer is not None:
                    # wire output tensor to
                    # output of producer node
                    for i, p_out in enumerate(producer.output):
                        if p_out == n.input[0]:
                            producer.output[i] = n.output[0]
                else:
                    # node is first in graph
                    successors = model.find_direct_successors(n)
                    assert successors is not None, "Whole graph is one node."
                    for succ in successors:
                        for i, s_inp in enumerate(succ.input):
                            if s_inp == n.output[0]:
                                # rewire successor's input directly to graph input
                                succ.input[i] = n.input[0]
                # remove node
                model.graph.node.remove(n)
        return (model, False)

class MergeReshapeConcat(Transformation):
    def _remove(self, model, n):
        producer = model.find_producer(n.input[0])
        if producer is not None:
            # wire output tensor to output of producer node
            for i, p_out in enumerate(producer.output):
                if p_out == n.input[0]:
                    producer.output[i] = n.output[0]
        # remove node
        model.graph.node.remove(n)
    
    def apply(self, model: ModelWrapper):
        for n in model.graph.node:
            if n.op_type == "Concat":
                producers = [model.find_producer(inp) for inp in n.input]
                if not all([p.op_type == "Reshape" for p in producers]):
                    continue

                for prod in producers:
                    self._remove(model, prod)
                

        return (model, False)

class SetFpgaResourceTypes(Transformation):
    def apply(self, model: ModelWrapper):
        for n in model.graph.node:
            if n.op_type == "MatrixVectorActivation":
                node = getCustomOp(n)
                if node.get_nodeattr("binaryXnorMode") == 0:
                    node.set_nodeattr("resType", "dsp")
            if n.op_type == "ConvolutionInputGenerator":
                node = getCustomOp(n)
                node.set_nodeattr("ram_style", "block")
        return (model, False)

def my_preprocess(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model.set_tensor_layout(model.graph.input[0].name, data_layout.NCHW)
    model = model.transform(RemoveUnusedOuputs())

    # Remove first quantization so that UINT8 data can be used directly
    new_input_node = model.get_nodes_by_op_type("Mul")[0]
    new_input_tensor = model.get_tensor_valueinfo(new_input_node.input[0])
    old_input_tensor = model.graph.input[0]
    model.graph.input.remove(old_input_tensor)
    model.graph.input.append(new_input_tensor)
    new_input_index = model.get_node_index(new_input_node)
    del model.graph.node[0:new_input_index]

    # Discard everything after the first few layers (for faster testing)
    # new_output_node = model.get_nodes_by_op_type("Mul")[22]
    # new_output_tensor = model.get_tensor_valueinfo(new_output_node.output[0])
    # # old_output_tensor = model.graph.output[0]
    # # model.graph.output.remove(old_output_tensor)
    # del model.graph.output[:]
    # model.graph.output.append(new_output_tensor)
    # new_output_index = model.get_node_index(new_output_node)
    # del model.graph.node[new_output_index+1:-1]
    # model.graph.node.remove(model.get_nodes_by_op_type("Concat")[0])

    # remove redundant value_info for primary input/output
    # othwerwise, newer FINN versions will not accept the model
    if model.graph.input[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.input[0])
    if model.graph.output[0] in model.graph.value_info:
        model.graph.value_info.remove(model.graph.output[0])

    return model

def my_step_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """

    streamline_transformations = [
        absorb.AbsorbSignBiasIntoMultiThreshold(),
        reorder.MoveMulPastFork(),
        MoveMulPastSPFF(),
        MoveIdenticalOpPastConcat(["Mul"]),
        reorder.MoveScalarMulPastConv(),
        reorder.MoveScalarLinearPastInvariants(),
        MoveIdenticalOpPastConcat(["Mul"]),
        collapse.CollapseRepeatedMul(),
        absorb.AbsorbAddIntoMultiThreshold(),
        absorb.AbsorbMulIntoMultiThreshold(),
        InferDataTypes(),
        InferShapes(),
        InferDataLayouts(),
        LowerConvsToMatMul(),
        absorb.AbsorbTransposeIntoMultiThreshold(),

        MoveTranspose(),
        MoveIdenticalOpPastConcat(["Transpose"]),
        absorb.AbsorbConsecutiveTransposes(),
        MoveTransposePastReshape(),
        MoveIdenticalOpPastConcat(["Transpose"]),
        absorb.AbsorbConsecutiveTransposes(),

        round.RoundAndClipThresholds(),
        InferDataTypes(),
        InferShapes(),
        InferDataLayouts(),
        GiveUniqueNodeNames(),
        GiveReadableTensorNames(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)

    if build_cfg.VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)
    return model

def my_step_convert_to_hls(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Convert eligible nodes to `HLSCustomOp` subclasses that represent HLS
    layers. Which nodes and particular configurations can be converted to HLS
    is limited, see the source code of the `convert_to_hls` module for more."""

    mem_mode = cfg.default_mem_mode.value
    if cfg.standalone_thresholds:
        # doing this first causes all threshold layers to be standalone
        model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferAddStreamsLayer())
    model = model.transform(to_hls.InferUpsample())
    # needed for bipolar MatMul layers
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
    # TopK to LabelSelect
    model = model.transform(to_hls.InferLabelSelectLayer())
    # input quantization (if any) as standalone threshold
    model = model.transform(to_hls.InferThresholdingLayer())
    
    model = model.transform(MergeReshapeConcat())
    model = model.transform(to_hls.InferConcatLayer())
    model = model.transform(to_hls.InferDuplicateStreamsLayer())
    
    # needed for convolutions -- TODO always exec?
    need_conv = len(model.get_nodes_by_op_type("Im2Col")) > 0
    if need_conv:
        model = model.transform(to_hls.InferPool_Batch())
        if cfg.force_rtl_conv_inp_gen:
            model = model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
        else:
            model = model.transform(to_hls.InferConvInpGen())
        model = model.transform(RemoveCNVtoFCFlatten())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())

    model = model.transform(SetFpgaResourceTypes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    return model

# parser = argparse.ArgumentParser()
# parser.add_argument("--start", nargs=1, default=None)
# parser.add_argument("--end", nargs=1, default=None)
# args = parser.parse_args()

cfg_estimates = build.DataflowBuildConfig(
    enable_build_pdb_debug=False,
    verbose             = True,
    mvau_wwidth_max     = 1024,
    target_fps          = 60,
    synth_clk_period_ns = 5.0,
    rtlsim_batch_size   = 32,
    board               = "U280",
    shell_flow_type     = build_cfg.ShellFlowType.VITIS_ALVEO,

    split_large_fifos   = True,
    force_python_rtlsim = True,
    auto_fifo_depths    = True,
    # folding_config_file = "depth_config.json",
    # verify_save_rtlsim_waveforms = True,

    force_rtl_conv_inp_gen = True,

    # start_step          = "step_set_fifo_depths",
    # stop_step           = "step_set_fifo_depths",
    steps               = [
                            "step_qonnx_to_finn",
                            my_preprocess,
                            "step_tidy_up",
                            # "step_streamline",
                            my_step_streamline,
                            # "step_convert_to_hls",
                            my_step_convert_to_hls,
                            "step_create_dataflow_partition",
                            "step_target_fps_parallelization",
                            "step_apply_folding_config",
                            "step_minimize_bit_width",
                            "step_generate_estimate_reports",
                            "step_hls_codegen",
                            "step_hls_ipgen",
                            "step_set_fifo_depths",
                            "step_create_stitched_ip",
                            "step_measure_rtlsim_performance",
                            "step_out_of_context_synthesis",
                            "step_synthesize_bitfile",
                            "step_make_pynq_driver",
                            "step_deployment_package",
                          ],
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]
)
build.build_dataflow_cfg("models/quantized_yolo.onnx", cfg_estimates)
