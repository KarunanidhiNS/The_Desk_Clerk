import os
import re
from flask import Flask, render_template, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Global state
qa_pairs = []     
index = None      



def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


QA_PATTERN = re.compile(
    r"Q\s*[:.]\s*(?P<question>.*?)\s*"
    r"(?:Ans|A)\s*[:.]\s*(?P<answer>.*?)"
    r"(?=Q\s*[:.]|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_qa_pairs(text):

    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n+", "\n", normalized)

    pairs = []
    for match in QA_PATTERN.finditer(normalized):
        question = match.group("question").strip().replace("\n", " ")
        answer = match.group("answer").strip().replace("\n", " ")

        question = re.sub(r"\s{2,}", " ", question)
        answer = re.sub(r"\s{2,}", " ", answer)

        if question and answer:
            pairs.append({"question": question, "answer": answer})

    return pairs


def split_into_paragraphs(text):
    """
    Used only if the PDF does NOT follow a Q:/Ans: format.
    Falls back to paragraph-level chunks so search still works.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = [p.strip().replace("\n", " ") for p in paragraphs if len(p.strip()) > 30]
    return chunks


def create_vector_store(items):
    """
    items: list of strings to embed and search over.
    For Q&A mode we embed the QUESTIONS (much better match quality
    than embedding the whole chunk), and look up the matching answer
    by index afterwards.

    We L2-normalize embeddings and use an Inner-Product index, which
    makes the search score equal to COSINE SIMILARITY (range -1 to 1,
    1 = identical meaning, 0 = unrelated, negative = opposite).
    Cosine similarity gives a far more reliable "is this actually
    relevant?" signal than raw L2 distance.
    """
    global index

    embeddings = model.encode(items, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product = cosine sim after normalize
    index.add(embeddings)

    print("FAISS Index Created")
    print("Total items indexed:", len(items))



STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "to", "and", "or",
    "are", "was", "were", "for", "with", "does", "do", "did", "how",
    "who", "which", "when", "where", "why", "explain", "define", "tell",
    "me", "about", "this", "that", "be", "can", "you"
}


def keyword_overlap(query, text):
    """
    Returns True if the query shares at least one meaningful (non-stopword)
    word with the candidate question/text. This catches cases where the
    embedding model finds a 'closest available' match that is actually
    completely unrelated (e.g. asking about something not in the PDF at all).
    """
    query_words = {w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in STOPWORDS and len(w) > 2}
    text_words = {w for w in re.findall(r"[a-zA-Z]+", text.lower()) if w not in STOPWORDS and len(w) > 2}

    if not query_words:
        return True  

    return len(query_words & text_words) > 0


def search_answer(query, top_k=1, similarity_threshold=0.45):
    global qa_pairs, index

    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    similarities, indices = index.search(query_embedding, top_k)

    best_idx = indices[0][0]
    best_similarity = similarities[0][0]

    if best_idx < 0 or best_idx >= len(qa_pairs):
        return "Answer not found."

    matched_question = qa_pairs[best_idx]["question"]

    if best_similarity < similarity_threshold:
        return "Answer not found. This topic does not appear to be in the uploaded PDF."

    if not keyword_overlap(query, matched_question):
        return "Answer not found. This topic does not appear to be in the uploaded PDF."

    return qa_pairs[best_idx]["answer"]


@app.route("/")
def home():
    return render_template("index.html", answer="")



@app.route("/upload", methods=["POST"])
def upload_pdf():
    global qa_pairs

    if "pdf" not in request.files:
        return render_template("index.html", answer="No PDF uploaded.")

    pdf_file = request.files["pdf"]

    if pdf_file.filename == "":
        return render_template("index.html", answer="Please select a PDF.")

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        return render_template("index.html", answer="No readable text found in PDF.")

    parsed = parse_qa_pairs(text)

    if parsed:
        qa_pairs = parsed
        questions_only = [p["question"] for p in qa_pairs]
        create_vector_store(questions_only)
        return render_template(
            "index.html",
            answer=f"PDF uploaded successfully. Extracted {len(qa_pairs)} Q&A pairs."
        )

    chunks = split_into_paragraphs(text)
    if not chunks:
        return render_template("index.html", answer="Could not extract usable content from PDF.")

    qa_pairs = [{"question": c, "answer": c} for c in chunks]
    create_vector_store(chunks)

    return render_template(
        "index.html",
        answer=f"PDF uploaded successfully (no Q:/Ans: format detected). "
               f"Indexed {len(chunks)} paragraphs for search."
    )

@app.route("/ask", methods=["POST"])
def ask_question():
    global index

    question = request.form.get("question")

    if not question:
        return render_template("index.html", answer="Please enter a question.")

    if index is None:
        return render_template("index.html", answer="Please upload a PDF first.")

    answer = search_answer(question)

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=False)