from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from transformers import pipeline
import pdfplumber
import os
import tempfile

app = Flask(__name__)
app.secret_key = "change-this-secret"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
app.config["UPLOAD_EXTENSIONS"] = [".pdf"]

# Load text generation model once at startup
generator = pipeline(
    "text-generation",
    model="gpt2",
)


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    uploaded_file = request.files.get("pdf_file")
    if not uploaded_file or uploaded_file.filename == "":
        flash("Please choose a PDF file to upload.")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded_file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in app.config["UPLOAD_EXTENSIONS"]:
        flash("Invalid file type. Please upload a PDF document.")
        return redirect(url_for("index"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        uploaded_file.save(tmp.name)
        pdf_text = extract_text_from_pdf(tmp.name)

    os.unlink(tmp.name)

    if not pdf_text.strip():
        flash("The uploaded PDF did not contain extractable text.")
        return redirect(url_for("index"))

    pdf_text = pdf_text[:1000]
    prompt = f"Summarize this text:\n{pdf_text}"

    result = generator(prompt, max_new_tokens=100, do_sample=False)
    summary = result[0]["generated_text"]

    return render_template(
        "index.html",
        summary=summary,
        original_text=pdf_text,
        filename=filename,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
