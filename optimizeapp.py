import os
import fitz  # PyMuPDF
import tempfile
from transformers import pipeline
import json
import io
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError

app = Flask(__name__)

# Securely load sensitive information from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the summarization pipeline using t5-small
summarizer = pipeline("summarization", model="t5-small")

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
    return text

def split_text_into_paragraphs(text):
    """Splits text into paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

def summarize_paragraphs(paragraphs, max_length=150, min_length=30):
    """Summarizes each paragraph."""
    summaries = [summarizer(p, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] for p in paragraphs]
    return summaries

def create_jsonl_content(original_paragraphs, summarized_paragraphs, system_role):
    """Creates JSONL content for fine-tuning."""
    json_lines = [
        json.dumps({
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": original},
                {"role": "assistant", "content": summary}
            ]
        })
        for original, summary in zip(original_paragraphs, summarized_paragraphs)
    ]
    return "\n".join(json_lines)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({"error": "No file part or file not selected"}), 400

    file = request.files['file']
    model_type = request.form.get('model_type')
    system_role = request.form.get('system_role')

    if not model_type:
        return jsonify({"error": "Model type not provided"}), 400
    if not system_role:
        return jsonify({"error": "System role not provided"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        text = extract_text_from_pdf(temp_file_path)
        paragraphs = split_text_into_paragraphs(text)
        summaries = summarize_paragraphs(paragraphs)

        jsonl_content = create_jsonl_content(paragraphs, summaries, system_role)
        jsonl_bytes = io.BytesIO(jsonl_content.encode('utf-8'))

        response = client.files.create(file=jsonl_bytes, purpose="fine-tune")

        fine_tuning_job = client.fine_tuning.jobs.create(
            model=model_type,
            training_file=response.id,
        )

        while True:
            job_status = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
            if job_status.status == "succeeded":
                model_id = job_status.fine_tuned_model
                break
            elif job_status.status == "failed":
                raise OpenAIError("Fine-tuning failed. Check logs for details.")

        os.remove(temp_file_path)

        return jsonify({"message": "Fine-tuning job created successfully", "job_id": fine_tuning_job.id, "model_id": model_id}), 200

    except OpenAIError as e:
        return jsonify({"error": f"An OpenAI error occurred: {e}"}), 500

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
