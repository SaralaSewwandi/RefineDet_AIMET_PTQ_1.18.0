import onnx

# Load the ONNX model
model = onnx.load("/home/bmw/anaconda3/envs/refinedetrepo2/refinedet.pytorch/refinedet512.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))