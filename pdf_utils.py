from pypdf import PdfReader

def load_and_chunk_pdfs(paths, chunk_size=300, overlap=50):
    documents = []

    for path in paths:
        reader = PdfReader(path)

        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        chunks = chunk_text(text, chunk_size, overlap)
        documents.extend(chunks)

    return documents


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
