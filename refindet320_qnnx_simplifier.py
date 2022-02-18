from onnxsim import simplify
import onnx


# load your predefined ONNX model
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/rangefix/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/rangefix/no_export_params/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/rangefix/no_export_params/opset11/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/original/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/refinedet320.onnx")
#model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/refinedet320.onnx")

model = onnx.load("/home/bmw/anaconda3/envs/check37/refinedet-pytorch/onnx/latest/refinedet320.onnx")




# convert model
model_simp, check = simplify(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model_simp.graph))

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object