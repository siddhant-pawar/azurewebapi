# Azure WEB API:
This code is a Flask web application that allows users to upload a PDF file. The application extracts text from the PDF, summarizes the text using a pre-trained model (t5-small), creates a JSONL file suitable for fine-tuning a model on OpenAI's platform, and initiates a fine-tuning job.
### Imports and Initialization

```python
import os
import fitz  # PyMuPDF
import tempfile
from transformers import pipeline
import json
import io
from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError

```
- os: For interacting with the operating system, especially for environment variables.
- fitz (PyMuPDF): For reading PDF files.
- tempfile: For creating temporary files.
- pipeline from transformers: For using a pre-trained summarization model.
- json and io: For handling JSON data and input/output operations.
- flask: For creating the web application.
- openai: For interacting with the OpenAI API.


### Environment Setup
To securely store and load sensitive information, create a .env file with the following content:
```
# .env file
OPENAI_API_KEY=your_openai_api_key
```

```python
app = Flask(__name__)

# Securely load sensitive information from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the summarization pipeline using t5-small
summarizer = pipeline("summarization", model="t5-small")

```
- Initializes the Flask application.
- Loads the OpenAI API key from environment variables.
- Initializes the OpenAI client.
- Loads the t5-small model for summarization using the Hugging Face transformers library.
- 
## Utility Functions
### extract_text_from_pdf :- 
```python
def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
    return text
```
- Opens a PDF file and extracts text from each page.
- Uses fitz to read the PDF and concatenate the text from each page.

### split_text_into_paragraphs :- 
```python
def split_text_into_paragraphs(text):
    """Splits text into paragraphs."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

```
- Splits the extracted text into paragraphs by splitting on newline characters.
- Removes any leading or trailing whitespace from each paragraph and filters out empty paragraphs.
### split_text_into_paragraphs :- 
```python
def summarize_paragraphs(paragraphs, max_length=150, min_length=30):
    """Summarizes each paragraph."""
    summaries = [summarizer(p, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text'] for p in paragraphs]
    return summaries
```
- Uses the t5-small summarization model to summarize each paragraph.
- The max_length and min_length parameters control the length of the summaries.
- do_sample=False ensures deterministic outputs.
  
### create_jsonl_content -
```python
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
```
- Creates a JSONL (JSON Lines) formatted string for fine-tuning an OpenAI model.
- Each line is a JSON object containing a conversation between system, user, and assistant roles.
## Flask Routes
### Index Route :- 

```python
@app.route('/')
def index():
    return render_template('index.html')
```
- Renders the index.html template when the root URL is accessed.
### Upload Route :-
```python
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

```
- Handles file uploads via POST requests.
- Validates the presence of the file, model_type, and system_role.
- Saves the uploaded file to a temporary location.
- Extracts text from the PDF, splits it into paragraphs, and summarizes each paragraph.
- Creates JSONL content and uploads it to OpenAI for fine-tuning.
- Initiates a fine-tuning job and monitors its status until it succeeds or fails.
- Removes the temporary file after processing.
- Returns a JSON response with the status of the fine-tuning job.
### Running the Application
```python
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
```
- Runs the Flask application on all network interfaces (0.0.0.0) at port 8000.
