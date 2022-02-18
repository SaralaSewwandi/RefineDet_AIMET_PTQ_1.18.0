import onnxruntime as ort

ort_session = ort.InferenceSession("/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/refinedet512.onnx")


outputs = ort_session.run(
    None,
    {"input": np.random.randn(32, 3, 512, 512).astype(np.float32)},
)
print(outputs[0])