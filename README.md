# 🌸 RAGarden: Multimodal AI for Floriculture

**RAGarden** is a multimodal Retrieval-Augmented Generation (RAG) system that combines **text and image embeddings** using [ImageBind](https://github.com/facebookresearch/ImageBind), integrates with **KDB.ai** for vector storage & search, and uses **Google Gemini** to generate rich descriptions of flowers.  
It features a **Tkinter desktop UI** with theme switching, image display, and export options.

---

## Features

- **Multimodal Embeddings** – Extract embeddings from both flower images and text descriptions.
- **Vector Search with KDB.ai** – Store and search embeddings efficiently.
- **Gemini AI Integration** – Generate detailed flower descriptions.
- **Tkinter UI** – User-friendly interface with dark/light mode toggle.
- **Data Import** – Automatically fetch sample data from GitHub (images & text).
- **Export Options** – Copy or save AI-generated responses.
- **Response Logging** – Keep a record of all queries and responses.

---

## Project Structure

<pre>
RAGarden.py            # Main application file
requirements.txt        # Python dependencies
.env                    # Environment variables
data/
 ├── images/            # Flower images
 └── text/              # Text descriptions
rag_response_log.txt    # Saved responses
</pre>

---

## Environment Variables

Create a `.env` file in the project root:

```env
KDBAI_API_KEY=your_kdbai_api_key
KDBAI_ENDPOINT=your_kdbai_endpoint
GOOGLE_API_KEY=your_google_api_key
```

---
## Installation

# 1. Clone the repository
```bash
git clone https://github.com/whoravinder/RAGarden
```

# 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv 
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
# 3. Install dependencies
```python
pip install -r requirements.txt
```
---

## Libraries used
-Python 3.9+
-PyTorch – for running ImageBind embeddings
-ImageBind – multimodal embedding model
-KDB.ai – vector database
-Google Gemini API – text generation
-Tkinter – desktop UI
-Pillow – image processing
-Pandas / NumPy – data handling

---



