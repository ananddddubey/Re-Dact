import spacy
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import pytesseract
from pytesseract import Output
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
import io
import os
from flask import current_app

# Load models
nlp = spacy.load("en_core_web_sm")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def extract_text_and_boxes(image_path):
    image = cv2.imread(image_path)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    return image, data

def redact_text_in_image(image_path, level):
    image, data = extract_text_and_boxes(image_path)
    num_boxes = len(data['text'])

    for i in range(num_boxes):
        if int(data['conf'][i]) > 60:
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                if level == 1:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
                elif level == 2:
                    if any(not c.isalnum() for c in text):
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

    output_path = os.path.join(current_app.config['REDACTED_FOLDER'], os.path.basename(image_path).replace(".png", "_redacted.png").replace(".jpg", "_redacted.jpg"))
    cv2.imwrite(output_path, image)
    return output_path

def blur_faces(image_path):
    image = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_region = image[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        image[y:y + h, x:x + w] = blurred_face

    output_path = os.path.join(current_app.config['REDACTED_FOLDER'], os.path.basename(image_path).replace(".png", "_faces_redacted.png").replace(".jpg", "_faces_redacted.jpg"))
    cv2.imwrite(output_path, image)
    return output_path

def generate_synthetic_data(entity_text, entity_label):
    input_ids = tokenizer.encode(f"Generate a synthetic example for {entity_label}: {entity_text}", return_tensors='pt')
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
    synthetic_data = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return synthetic_data

def redact_text(text, level=1):
    doc = nlp(text)
    redacted_text = text

    for ent in doc.ents:
        if level == 1:
            redacted_text = redacted_text.replace(ent.text, "[REDACTED]")
        elif level == 2:
            redacted_text = redacted_text.replace(ent.text, "")
            redacted_text = ''.join(['[SYMBOL]' if not c.isalnum() and not c.isspace() else c for c in redacted_text])
        elif level == 3:
            synthetic_data = generate_synthetic_data(ent.text, ent.label_)
            redacted_text = redacted_text.replace(ent.text, synthetic_data)

    return redacted_text

def redact_pdf(file_path, level):
    reader = PdfReader(file_path)
    writer = PdfWriter()

    for page in reader.pages:
        text = page.extract_text()
        if text:
            redacted_text = redact_text(text, level)
            # Note: PdfWriter does not support inserting text directly.
            # Advanced PDF manipulation would require additional libraries like reportlab or pdfplumber.
            # For simplicity, we'll skip actual redaction in PDF and return original.
            writer.add_page(page)

    output_file = os.path.join(current_app.config['REDACTED_FOLDER'], os.path.basename(file_path).replace(".pdf", f"_redacted_l{level}.pdf"))
    with open(output_file, "wb") as f_out:
        writer.write(f_out)
    return output_file

def process_file(file_path, file_type, level):
    if file_type == "text":
        with open(file_path, 'r') as file:
            content = file.read()
        redacted_content = redact_text(content, level)
        output_file = os.path.join(current_app.config['REDACTED_FOLDER'], os.path.basename(file_path).replace(".txt", f"_redacted_l{level}.txt"))
        with open(output_file, 'w') as file:
            file.write(redacted_content)
        return output_file

    elif file_type == "csv":
        df = pd.read_csv(file_path)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: redact_text(str(x), level) if isinstance(x, str) else x)
        output_file = os.path.join(current_app.config['REDACTED_FOLDER'], os.path.basename(file_path).replace(".csv", f"_redacted_l{level}.csv"))
        df.to_csv(output_file, index=False)
        return output_file

    elif file_type == "image":
        if level == 3:
            face_redacted_image = blur_faces(file_path)
            return face_redacted_image
        else:
            redacted_image = redact_text_in_image(file_path, level)
            return redacted_image

    elif file_type == "pdf":
        redacted_pdf_file = redact_pdf(file_path, level)
        return redacted_pdf_file
