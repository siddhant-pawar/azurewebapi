import os
import fitz  # PyMuPDF
import tempfile
from transformers import pipeline
import json
import io
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError

app = Flask(__name__)
client = OpenAI(api_key="sk-proj-EFqvn4ZKJDizxwgWRl1IT3BlbkFJHIZe3tLDA1VNJK7Gwd7X")
# Load the summarization pipeline using t5-small
summarizer = pipeline("summarization", model="t5-small")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def split_text_into_paragraphs(text):
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def summarize_paragraphs(paragraphs, max_length=150, min_length=30):
    summaries = []
    for paragraph in paragraphs:
        summary = summarizer(paragraph, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return summaries

def create_jsonl_content(original_paragraphs, summarized_paragraphs, system_role):
    jsonl_content = ""
    for original, summary in zip(original_paragraphs, summarized_paragraphs):
        json_line = {
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": original},
                {"role": "assistant", "content": summary}
            ]
        }
        jsonl_content += json.dumps(json_line) + '\n'
    return jsonl_content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print("Request received")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    print(f"File received: {file.filename}")

    model_type = request.form.get('model_type')
    system_role = request.form.get('system_role')
    if not model_type:
        return jsonify({"error": "Model type not provided"}), 400
    if not system_role:
        return jsonify({"error": "System role not provided"}), 400

    print(f"Model type selected by the user: {model_type}")
    print(f"System role provided by the user: {system_role}")

    try:
        # Save the uploaded file to a temporary location
        temp_file_path = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(temp_file_path)
        paragraphs = split_text_into_paragraphs(text)
        summaries = summarize_paragraphs(paragraphs)

        # Create JSONL content
        jsonl_content = create_jsonl_content(paragraphs, summaries, system_role)

        # Upload the JSONL content to OpenAI
        jsonl_bytes = io.BytesIO(jsonl_content.encode('utf-8'))
        print("Uploading JSONL content to OpenAI...")
        response = client.files.create(file=jsonl_bytes, purpose="fine-tune")
        print("File successfully uploaded to OpenAI.")

        # Schedule fine-tuning job
        fine_tuning_job = client.fine_tuning.jobs.create(
            model=model_type,
            training_file=response.id,
        )

        # Check the fine-tuning job status
        while True:
            job_status = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
            if job_status.status == "succeeded":
                print(f"Fine-tuning completed! Model ID: {job_status.fine_tuned_model}")
                break
            elif job_status.status == "failed":
                print("Fine-tuning failed. Check logs for details.")
                break

        # Delete the temporary file after processing
        if temp_file_path:
            os.remove(temp_file_path)

        return jsonify({"message": "Fine-tuning job created successfully", "job_id": fine_tuning_job.id}), 200

    except OpenAIError as e:
        print(f"An OpenAI error occurred: {e}")
        return jsonify({"error": f"An OpenAI error occurred: {e}"}), 500

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
