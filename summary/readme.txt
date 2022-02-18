summary
1.refindet320 export - pretrained checkpoints with model architecture
with range fixes 
opset_version=11
torch.onnx.export(refinedet320, dummy_input, "/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx", verbose=True, input_names=input_names, output_names=output_names, opset_version=11)

python refinedet320_export_to_onnx.py
see the output results file in the folder

2.refinedet jit trace - pretrained checkpoints with model architecture
with range fixes 
python refinedet320_jit_trace.py
see the output results file in the folder

3.verify onnx - check model
 python verify_refinedet320_onnx.py
model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx")
see the output results file in the folder

4.OnnxInferenceSession test - for onnx validation
ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx")
python run_refindet320_onnx.py

runtime error
sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Load model from /home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx failed:Node (Range_4) Op (Range) [ShapeInferenceError] Input to 'Range' op should be scalars (Tensor with only one element and shape empty)
(check37) bmw@BMW:~/anaconda3/envs/check37/refinedet-pytorch$
see the output results file in the folder

5.onnx simplifier check
model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx")
python refindet320_qnnx_simplifier.py

same runtime error is coming 
sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL : Node (Range_4) Op (Range) [ShapeInferenceError] Input to 'Range' op should be scalars (Tensor with only one element and shape empty)
(check37) bmw@BMW:~/anaconda3/envs/check37/refinedet-pytorch$

see the output results file in the folder



extra testing
1. checked with different op set versions - 11 and 12
2. input shapes are not dynamic -  same (32, 3, 320, 320) shape is taking




