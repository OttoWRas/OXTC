# Generate simple ONNX model for testing acceleration of "binary activation maps" technique
# Script output:
# - ONNX file
# - Test input and output vectors
# Usage examples: 
# - default parameters (in_size=5, out_size=3):
#   python3 generate_onnx_model.py
# - custom parameters (in_size=10, out_size=5):
#   python3 generate_onnx_model.py 10 5

import sys
import onnx
import numpy as np
import onnxruntime as ort
from onnx import helper, checker
from onnx import TensorProto

# Scale factor
S = 10 

def main(in_size, out_size):
    print(f"Creating model with {in_size=} and {out_size=}")

    # Create a tensor for the weights (we use random values for now)
    weights = (np.random.randn(in_size, out_size) * S).astype(np.int32)
    weight_tensor = helper.make_tensor(
        name="W",
        data_type=TensorProto.INT32,
        dims=[in_size, out_size],
        vals=weights.flatten()
    )

    # Declare input and output nodes
    # We have 2 outputs: the linear layer output and the binary activation output
    input_tensor = helper.make_tensor_value_info("q qsz", TensorProto.INT32, [1, in_size])
    output_tensor1 = helper.make_tensor_value_info("y", TensorProto.INT32, [1, out_size])
    output_tensor2 = helper.make_tensor_value_info("y_hat", TensorProto.INT32, [1, out_size])

    # Nodes for the operations
    # The op_type comes from the ONNX documentation: https://onnx.ai/onnx/operators/
    # The inputs and outputs connections come from the tensor declarations above (x, W, y, y_hat) 
    #   or are created on the fly (_y)
    # Since there is no "binary" activation function in ONNX, we use Relu and Sign to simulate it
    matmul_node = helper.make_node(
        op_type="MatMul",
        inputs=["x", "W"],
        outputs=["y"],
    )

    activation1_node = helper.make_node(
        op_type="Relu",
        inputs=["y"],
        outputs=["_y"],
    )
    activation2_node = helper.make_node(
        op_type="Sign",
        inputs=["_y"],
        outputs=["y_hat"],
    )

    # Create the graph
    # This connects the nodes and tensors together
    graph = helper.make_graph(
        nodes=[matmul_node, activation1_node, activation2_node],
        name="LinearLayer",
        inputs=[input_tensor],
        outputs=[output_tensor1, output_tensor2],
        initializer=[weight_tensor],
    )

    # Create the model and verify that everything is correct
    model = helper.make_model(graph)
    checker.check_model(model, full_check=True)
    print(f"Model successfully created.")

    # Save the model to a file
    # The filename is based on the input and output sizes
    model_path = f"./simple_model_{in_size}_{out_size}.onnx"
    onnx.save(model, model_path)
    print(f"Model saved to {model_path}")

    # Now we can run the model to get some test data
    # First we must load the ONNX model (the library for running the 
    #   model is different from the one for creating it)
    print()
    print("Running the model to get test data")
    print(f"Loading model from {model_path}")
    session = ort.InferenceSession(model_path)

    # Show the input and output information
    for _in in session.get_inputs():
        print(f"Input {_in.name}: shape={_in.shape}")
    for _out in session.get_outputs():
        print(f"Output {_out.name}: shape={_out.shape}")
    print()

    # Generate random input binary data of the correct size
    input_data = (np.random.randn(1, in_size) > 0).astype(np.int32)

    # Run the model
    # We specify the output names in a list, and we provide the 
    #   input tensor name and the data as a dictionary
    output_names = ["y", "y_hat"]
    output_data = session.run(output_names, {"x": input_data})

    # Print the weights, input and, corresponding output
    print(f"Weights:\n{weights}")
    print()
    print("Input:")
    print(f"x:\t {input_data}")
    print()
    print("Outputs:")
    for i, _out in enumerate(output_names):
        print(f"{_out}:\t {output_data[i]}")


if __name__ == "__main__":
    # Default parameters
    in_size, out_size = 5, 3
    if len(sys.argv) == 0:
        print(f"Using default parameters in_size={in_size} and out_size={out_size}")
    if len(sys.argv) > 1:
        in_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        out_size = int(sys.argv[2])
    # Run the main function
    main(in_size, out_size)
