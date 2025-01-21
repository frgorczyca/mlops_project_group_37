import onnxruntime as ort
import onnx
from transformers import AutoTokenizer

# Load the ONNX model with metadata
onnx_model = onnx.load("model_with_transformer_name.onnx")

# Retrieve the transformer model name from the metadata
transformer_model_name = None
for entry in onnx_model.metadata_props:
    if entry.key == "transformer_model_name":
        transformer_model_name = entry.value
        break

if not transformer_model_name:
    raise ValueError("Transformer model name metadata not found in the ONNX model.")

# Initialize the tokenizer based on the transformer model name
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

# Load the ONNX model session
onnx_session = ort.InferenceSession("model_with_transformer_name.onnx")

# Prepare input text
text = "This is a sample input."

# Tokenize input text
inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

# Prepare ONNX inputs in the right format
onnx_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

# Run inference with ONNX Runtime
onnx_output = onnx_session.run(None, onnx_inputs)

# Get the output (for example, the first output tensor)
output = onnx_output[0]

# Print the output
print(output)
