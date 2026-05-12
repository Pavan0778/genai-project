from transformers import pipeline
import pdfplumber

# Load text generation model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# PDF path
pdf_path = "sample.pdf"

# Read PDF
text = ""

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

# Reduce text size
text = text[:1000]

# Prompt
prompt = f"Summarize this text:\n{text}"

# Generate summary
result = generator(
    prompt,
    max_new_tokens=100,
    do_sample=False
)

# Print output
print("\nSUMMARY:\n")
print(result[0]['generated_text'])