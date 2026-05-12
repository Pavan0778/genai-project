from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from transformers import pipeline
import pdfplumber
import os
import tempfile

app = Flask(__name__)
app.secret_key = "change-this-secret"

# Maximum upload size = 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Allowed file types
app.config["UPLOAD_EXTENSIONS"] = [".pdf"]

# Load summarization model
generator = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)


# Extract text from uploaded PDF
def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


# Home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Summarize PDF
@app.route("/summarize", methods=["POST"])
def summarize():

    uploaded_file = request.files.get("pdf_file")

    # Check if file uploaded
    if not uploaded_file or uploaded_file.filename == "":
        flash("Please upload a PDF file.")
        return redirect(url_for("index"))

    # Secure filename
    filename = secure_filename(uploaded_file.filename)

    # Check extension
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in app.config["UPLOAD_EXTENSIONS"]:
        flash("Only PDF files are allowed.")
        return redirect(url_for("index"))

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        uploaded_file.save(tmp.name)

        # Extract text
        pdf_text = extract_text_from_pdf(tmp.name)

    # Delete temp file
    os.unlink(tmp.name)

    # Empty PDF check
    if not pdf_text.strip():
        flash("No readable text found in the PDF.")
        return redirect(url_for("index"))

    # Limit text size for model
    pdf_text = pdf_text[:2000]

    try:
        # Generate summary
        result = generator(
            pdf_text,
            max_length=120,
            min_length=40,
            do_sample=False
        )

        summary = result[0]["summary_text"]

    except Exception as e:
        flash(f"Error while generating summary: {str(e)}")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        summary=summary,
        original_text=pdf_text,
        filename=filename
    )


# Run Flask app
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )