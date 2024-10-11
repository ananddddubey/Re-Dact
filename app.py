from flask import Flask, render_template, request, send_from_directory
import os
from redactrion_model import process_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REDACTED_FOLDER = 'redacted'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REDACTED_FOLDER'] = REDACTED_FOLDER

# Ensure upload and redacted directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REDACTED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        level = int(request.form['level'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Determine file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext in ['.txt']:
            file_type = 'text'
        elif file_ext in ['.csv']:
            file_type = 'csv'
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            file_type = 'image'
        elif file_ext in ['.pdf']:
            file_type = 'pdf'
        else:
            return "Unsupported file type", 400
        
        # Process file
        redacted_file_path = process_file(file_path, file_type, level)
        redacted_filename = os.path.basename(redacted_file_path)
        return render_template('index.html', download_link=f"/download/{redacted_filename}")

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['REDACTED_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port= 8080)
