import gradio as gr
import onnxruntime as ort
from PIL import Image
import numpy as np

# Load ONNX model
try:
    session = ort.InferenceSession("generator.onnx")
    print("✅ ONNX model loaded successfully")
except Exception as e:
    print(f"⚠ Failed to load ONNX model: {e}")
    session = None

def preprocess_input(image_input):
    if isinstance(image_input, dict):
        if "composite" in image_input:
            image_array = np.array(image_input["composite"])
        elif "image" in image_input:
            image_array = np.array(image_input["image"])
        else:
            raise ValueError("Dict input missing 'composite' or 'image' key")
        
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        image_array = image_array[:, :, :3]  # drop alpha if present
        image = Image.fromarray(image_array)
    
    elif isinstance(image_input, np.ndarray):
        image_array = image_input
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        image_array = image_array[:, :, :3]
        image = Image.fromarray(image_array)
    
    elif isinstance(image_input, Image.Image):
        image = image_input
    
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")
    
    return image

def infer(image_input):
    if session is None:
        return "⚠ Error: Model not loaded."

    try:
        image = preprocess_input(image_input)
        img = image.convert("RGB").resize((256, 256))
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        img_np = np.expand_dims(np.transpose(img_np, (2, 0, 1)), axis=0)  # (1, 3, 256, 256)

        result = session.run(None, {"input": img_np})[0]

        if np.isnan(result).any() or result.shape[1:] != (3, 256, 256):
            print("⚠ Invalid result detected, returning fallback image")
            fallback_img = np.full((256, 256, 3), 128, dtype=np.uint8)
            return Image.fromarray(fallback_img)

        result_img = result.squeeze().transpose(1, 2, 0)  # (256, 256, 3)
        result_img = ((result_img + 1) * 127.5).clip(0, 255).astype(np.uint8)

        return Image.fromarray(result_img)
    except Exception as e:
        fallback_img = np.full((256, 256, 3), 255, dtype=np.uint8)
        return Image.fromarray(fallback_img)


custom_css = """
body { background-color: #111; color: #eee; }
footer { text-align: center; font-size: 0.8em; color: #888; margin-top: 30px; }
button:hover {
    box-shadow: 0 0 15px #0ff, 0 0 25px #0ff;
    transition: box-shadow 0.3s ease-in-out;
}
"""


# Gradio UI
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as demo:
    gr.Markdown(
        "<h1 style='text-align: center;'>👟 NeoStep</h1>"
        "<p style='text-align: center;'><i>Giving your sketches a futuristic upgrade.</i><br>"
        "<small>Powered by Antariksh</small></p>"
    )

    with gr.Tab("Upload Sketch"):
        upload_input = gr.Image(type="pil", label="Upload your sketch")
        upload_output = gr.Image(type="pil", label="Generated Image")
        upload_button = gr.Button("Generate")
        upload_button.click(fn=infer, inputs=upload_input, outputs=upload_output)

    with gr.Tab("Draw Sketch"):
        sketch_input = gr.Sketchpad(label="Draw your sketch", width=512, height=512)
        sketch_output = gr.Image(type="pil", label="Generated Image")
        sketch_button = gr.Button("Generate")
        sketch_button.click(fn=infer, inputs=sketch_input, outputs=sketch_output)

    gr.Markdown(
        "<footer>"
        "© 2025 NeoStep | Crafted with ❤️ by Antariksh<br>"
        "<a href='https://github.com/antariksh101' target='_blank' style='color:#0ff; text-decoration:none;'>GitHub</a> "
        "| "
        "<a href='https://www.linkedin.com/in/antariksh101/' target='_blank' style='color:#0ff; text-decoration:none;'>LinkedIn</a>"
        "</footer>"
    )

demo.launch()
