import onnx
import onnx.helper

def update_attribute(attributes, name, value):
    attributes = [attr for attr in attributes if attr.name != name]
    attributes.append(onnx.helper.make_attribute(name, value))
    return attributes

def determine_padding(kernel_shape):
    if kernel_shape == [3, 3]:
        return [1, 1, 1, 1]
    elif kernel_shape == [1, 1]:
        return [0, 0, 0, 0]
    return [0, 0, 0, 0]

def update_conv_attribute(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    for node in graph.node:
        if node.op_type == "Conv":
            print(f"Updating attributes for node: {node.name}")

            kernel_shape = None
            updated_attributes = []

            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = list(attr.ints)
                updated_attributes.append(attr)

            updated_attributes = update_attribute(updated_attributes, "auto_pad", "NOTSET")
            pads = determine_padding(kernel_shape)
            updated_attributes = update_attribute(updated_attributes, "pads", pads)

            del node.attribute[:]
            node.attribute.extend(updated_attributes)

    onnx.save(model, output_path)
    print(f"Updated model saved to: {output_path}")


MODEL_NAME="yolov3.onnx"
update_conv_attribute(f"onnxsim{MODEL_NAME}", f"autopad/{MODEL_NAME}")
