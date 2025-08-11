
import os
import re
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageTk
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data as imagebind_data
import google.generativeai as genai
from kdbai_client import Session
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
from datetime import datetime
import traceback

load_dotenv()

KDBAI_API_KEY = os.getenv("KDBAI_API_KEY")
KDBAI_ENDPOINT = os.getenv("KDBAI_ENDPOINT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not all([KDBAI_API_KEY, KDBAI_ENDPOINT, GOOGLE_API_KEY]):
    raise EnvironmentError("One or more API keys missing from .env")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval().to(device)


def extract_flower_name(query):
    patterns = [
        r"about\s+(\w+)",
        r"what\s+is\s+(\w+)",
        r"show\s+me\s+(\w+)",
        r"(\w+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    return ""

def read_text_from_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

@torch.no_grad()
def get_embedding_vector(inputs):
    embedding = model(inputs)
    for _, v in embedding.items():
        return v.reshape(-1).cpu().numpy()

def embed_text(text):
    inputs = {ModalityType.TEXT: imagebind_data.load_and_transform_text([text], device)}
    return get_embedding_vector(inputs)

def embed_image(path):
    inputs = {ModalityType.VISION: imagebind_data.load_and_transform_vision_data([path], device)}
    return get_embedding_vector(inputs)

def get_github_repo_contents(repo_owner, repo_name, branch, folder_path):
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}?ref={branch}"
    contents = requests.get(api_url).json()
    local_folder = f"./data/{folder_path.split('/')[-1]}"
    os.makedirs(local_folder, exist_ok=True)
    if isinstance(contents, list):
        for item in contents:
            url = item.get('download_url')
            if not url:
                continue
            file_path = os.path.join(local_folder, item['name'])
            try:
                if item['name'].lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = Image.open(requests.get(url, stream=True).raw)
                    img.save(file_path)
                else:
                    with open(file_path, 'wb') as f:
                        f.write(requests.get(url).content)
            except Exception as e:
                print(f"Failed to download {item['name']}: {e}")


if not os.path.exists("./data/text") or len(os.listdir("./data/text")) == 0:
    get_github_repo_contents("whoravinder", "RAGarden", "main", "data/images")
    get_github_repo_contents("whoravinder", "RAGarden", "main", "data/text")

df = pd.DataFrame(columns=["path", "media_type", "embeddings"])

for f in os.listdir("data/images"):
    path = os.path.join("data/images", f)
    try:
        vec = embed_image(path)
        assert len(vec) == 1024
        df.loc[len(df)] = [path, "image", vec]
    except Exception as e:
        print(f"Error embedding image {f}: {e}")

for f in os.listdir("data/text"):
    path = os.path.join("data/text", f)
    with open(path, "r") as file:
        content = file.read()
        if content.strip():
            try:
                vec = embed_text(content)
                assert len(vec) == 1024
                df.loc[len(df)] = [path, "text", vec]
            except Exception as e:
                print(f"Error embedding text {f}: {e}")

print("\n📊 Embedding Summary:")
print("🖼️ Images:", df[df["media_type"] == "image"].shape[0])
print("📝 Text  :", df[df["media_type"] == "text"].shape[0])

session = Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
try:
    db = session.database("myDatabase")
    print("📂 Using existing database")
except:
    db = session.create_database("myDatabase")
    print("🆕 Created database")

try:
    db.table("multi_modal_ImageBind").drop()
except:
    pass

table = db.create_table(
    table="multi_modal_ImageBind",
    schema=[
        {"name": "path", "type": "str"},
        {"name": "media_type", "type": "str"},
        {"name": "embeddings", "type": "float64s"},
    ],
    indexes=[
        {
            "type": "flat",
            "name": "flat_index",
            "column": "embeddings",
            "params": {"dims": 1024, "metric": "CS"},
        }
    ],
)
table.insert(df.to_dict(orient="records"))
print(f"✅ Inserted {len(df)} records into KDB.ai")

def rag_query(user_query):
    query_vec = embed_text(user_query).tolist()
    flower_keyword = extract_flower_name(user_query)

    text_res = table.search(
        vectors={"flat_index": [query_vec]}, n=1, filter=[("like", "media_type", "text")]
    )[0]

    try:
        text_path = text_res[0]["path"] if isinstance(text_res, list) and len(text_res) > 0 else None
    except:
        text_path = None

    selected_text = read_text_from_file(text_path) if text_path and os.path.exists(text_path) else ""

    image_path = None
    for fname in os.listdir("data/images"):
        if flower_keyword and flower_keyword.lower() in fname.lower():
            path = os.path.join("data/images", fname)
            if os.path.exists(path):
                image_path = path
                break

    prompt = (
        f"You are a flower expert.\n"
        f"Please describe the flower '{flower_keyword}' with the following details:\n"
        "- Scientific Name\n- Appearance\n- Habitat\n- Botanical traits\n- Symbolism\n- Fun Fact\n\n"
        f"Text description from database: {selected_text}"
    )

    response = gemini_model.generate_content([prompt])
    return prompt, response.text, image_path

def run_ui():
    global image_cache
    image_cache = None
    log_file_path = "ravinder_rag_response_log.txt"
    dark_mode = True

    def save_response_log(query, response):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== {datetime.now()} ===\n")
            f.write(f"Query: {query}\n")
            f.write(f"Response:\n{response}\n\n")

    def toggle_theme():
        nonlocal dark_mode
        dark_mode = not dark_mode
        apply_theme()

    def apply_theme():
        theme = dark_theme if dark_mode else light_theme
        root.configure(bg=theme["bg"])
        for w in themed_widgets:
            w.configure(bg=theme["entry_bg"], fg=theme["fg"])
        for b in buttons:
            b.configure(bg=theme["btn_bg"], fg=theme["btn_fg"])
        output_text.configure(bg=theme["text_bg"], fg=theme["fg"], insertbackground=theme["fg"])
        image_frame.configure(bg=theme["bg"])
        image_label.configure(bg=theme["bg"])

    def on_submit():
        global image_cache
        query = query_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a flower query.")
            return
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "🔍 Searching...\n")
        root.update_idletasks()
        try:
            prompt, result_text, image_path = rag_query(query)
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, result_text)
            save_response_log(query, result_text)
            if image_path and os.path.exists(image_path):
                pil_img = Image.open(image_path).resize((350, 350))
                image_cache = ImageTk.PhotoImage(pil_img)
                image_label.configure(image=image_cache, text="")
                image_label.image = image_cache
            else:
                image_label.configure(image="", text="🖼️ No image found", font=("Arial", 12), fg="gray")
        except Exception as e:
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}")

    def copy_output():
        root.clipboard_clear()
        root.clipboard_append(output_text.get("1.0", tk.END).strip())
        messagebox.showinfo("Copied", "Response copied to clipboard!")

    def download_output():
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(output_text.get("1.0", tk.END))
            messagebox.showinfo("Saved", f"Response saved to {file_path}")

    dark_theme = {
        "bg": "#121212", "fg": "#f8f8f2", "btn_bg": "#333", "btn_fg": "#fff",
        "entry_bg": "#1e1e1e", "text_bg": "#1e1e1e"
    }
    light_theme = {
        "bg": "#ffffff", "fg": "#212529", "btn_bg": "#0d6efd", "btn_fg": "#fff",
        "entry_bg": "#ffffff", "text_bg": "#ffffff"
    }

    root = tk.Tk()
    root.title("🌸 Flora-RAG 🌸")
    root.geometry("1200x720")
    root.resizable(False, False)

    themed_widgets, buttons = [], []

    header = tk.Label(root, text="🌸 Flora-RAG, Multimodal AI for Floriculture 🌸", font=("Helvetica", 16, "bold"))
    header.grid(row=0, column=0, columnspan=3, pady=(15, 5))
    themed_widgets.append(header)

    query_entry = tk.Entry(root, font=("Segoe UI", 10), width=60)
    query_entry.grid(row=1, column=0, padx=15, pady=10, sticky="w", columnspan=2)
    themed_widgets.append(query_entry)

    button_stack = tk.Frame(root)
    button_stack.grid(row=1, column=2, padx=10, pady=5, sticky="e")
    
    search_button = tk.Button(button_stack, text="Search 🌺", command=on_submit, font=("Segoe UI", 10), width=12)
    search_button.pack(side=tk.LEFT, padx=5)
    
    toggle_theme_button = tk.Button(button_stack, text="🌗 Theme", command=toggle_theme, font=("Segoe UI", 10), width=10)
    toggle_theme_button.pack(side=tk.LEFT)
    
    buttons.extend([search_button, toggle_theme_button])

    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 10), width=80, height=25, bd=2)
    output_text.grid(row=2, column=0, columnspan=2, padx=15, pady=10)

    button_frame = tk.Frame(root)
    button_frame.grid(row=3, column=0, columnspan=2, pady=5)

    copy_button = tk.Button(button_frame, text="📋 Copy", command=copy_output, font=("Segoe UI", 10), width=10)
    copy_button.pack(side=tk.LEFT, padx=10)
    download_button = tk.Button(button_frame, text="💾 Save", command=download_output, font=("Segoe UI", 10), width=10)
    download_button.pack(side=tk.LEFT, padx=10)

    buttons.extend([copy_button, download_button])

    image_frame = tk.Frame(root, width=350, height=350, bd=2, relief=tk.SOLID)
    image_frame.grid(row=2, column=2, padx=15, pady=10, sticky="n")

    image_label = tk.Label(image_frame, bg="gray", text="🖼️", font=("Arial", 14), anchor="center")
    image_label.place(relx=0.5, rely=0.5, anchor="center")
    themed_widgets.append(image_label)

    status_label = tk.Label(root, text="Created by Sayuri Shelley", font=("Helvetica", 10, "bold"))
    status_label.grid(row=4, column=0, columnspan=3, pady=(5, 10))
    themed_widgets.append(status_label)

    apply_theme()
    root.mainloop()

if __name__ == "__main__":
    run_ui()
