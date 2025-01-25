from flask import Flask, render_template, request, send_from_directory
import os
#import cv2
import numpy as np
#import onnxruntime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load your ONNX model
onnx_model_path = "your_model.onnx"  # Replace with your ONNX model path
ort_session = onnxruntime.InferenceSession(onnx_model_path)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def apply_onnx_model(image_path, output_path):
    return None
#    try:
#        img = cv2.imread(image_path)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure correct color format
#        img = cv2.resize(img, (224, 224)) # Resize to model input size (adjust if needed)
#        img = img.astype(np.float32) / 255.0  # Normalize
#        img = np.transpose(img, (2, 0, 1)) # Change HWC to CHW
#        img = np.expand_dims(img, axis=0) # Add batch dimension
#
#        ort_inputs = {input_name: img}
#        ort_outs = ort_session.run([output_name], ort_inputs)
#        predictions = ort_outs[0]
#
#        # Process predictions (example: get class with highest probability)
#        predicted_class = np.argmax(predictions)
#
#        # For demonstration, we'll just save a copy of the image.
#        cv2.imwrite(output_path, cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)) #Save the image
#        return predicted_class
#    except Exception as e:
#        print(f"Error processing image: {e}")
#        return None



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            output_filename = "output_" + filename
            output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            prediction = apply_onnx_model(filepath, output_filepath)
            if prediction is not None:
                return render_template('index.html', uploaded_filename=filename, processed_filename=output_filename, prediction=prediction)
            else:
                return render_template('index.html', error="Error processing image.")

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
