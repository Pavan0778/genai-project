from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pdfplumber
import os
import tempfile
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = Flask(__name__)
app.secret_key = "change-this-secret"

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_EXTENSIONS"] = [".pdf"]


def extract_text_from_pdf(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():

    uploaded_file = request.files.get("pdf_file")

    if not uploaded_file or uploaded_file.filename == "":
        flash("Please upload a PDF file.")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded_file.filename)

    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in app.config["UPLOAD_EXTENSIONS"]:
        flash("Only PDF files are allowed.")
        return redirect(url_for("index"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        uploaded_file.save(tmp.name)

        pdf_text = extract_text_from_pdf(tmp.name)

    os.unlink(tmp.name)

    if not pdf_text.strip():
        flash("No readable text found in the PDF.")
        return redirect(url_for("index"))

    pdf_text = pdf_text[:5000]

    parser = PlaintextParser.from_string(pdf_text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary_sentences = summarizer(parser.document, 3)

    summary = " ".join(str(sentence) for sentence in summary_sentences)

    return render_template(
        "index.html",
        summary=summary,
        original_text=pdf_text,
        filename=filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )