import onnx

# Load the ONNX model
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/original/refinedet320.onnx")
model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx")



# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))