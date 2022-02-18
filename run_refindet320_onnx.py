import onnxruntime as ort

'''
import onnx


# Load the ONNX model
model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/opset12/refinedet320.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)
#ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/refinedet320.onnx")
#ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/export_params/refinedet320.onnx")

model= onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model,"/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/infered/refinedet320.onnx")
'''
#ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/rangefix/refinedet320.onnx")
#ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/refinedet320.onnx")
ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx")





'''
outputs = ort_session.run(
    None,
    {"input": torch.randn(32, 3, 320, 320)},
)
print(outputs[0])
'''