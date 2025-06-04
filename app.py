from flask import Flask, render_template, request
import re

app = Flask(__name__)

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

def setup_faiss_index():
    df = pd.read_csv("bias_rewrite_pairs.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Encode biased sentences
    biased_sentences = df["biased"].tolist()
    embeddings = model.encode(biased_sentences, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "bias_faiss.index")

    # Save metadata for lookups
    metadata = {
        "biased": df["biased"].tolist(),
        "neutral": df["neutral"].tolist(),
        "bias_type": df["bias_type"].tolist()
    }
    with open("bias_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
class_names = list(model.config.id2label.values())
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
explainer = LimeTextExplainer(class_names=class_names)

def detect_bias_type(text):
    result = pipe(text)

    explanation = explainer.explain_instance(
        text, 
        lambda x: np.array([ [label['score'] for label in pipe(xx)[0]] for xx in x ]),
        num_features=5, 
        top_labels=1
    )

    lime_html = explanation.as_html()
    
    top_label = max(result[0], key=lambda x: x['score'])['label']
    return top_label, lime_html

def get_bias_label(label):
    label_map = {
        "LABEL_0": "Masculine_Bias",
        "LABEL_1": "Feminine_Bias",
        "LABEL_2": "Racial_Bias",
        "LABEL_3": "Age_Bias",
        "LABEL_4": "Disability_Bias",
        "LABEL_5": "LGBTQ_Bias"
    }
    return label_map.get(label, "Unknown")

def retrieve_example(bias_type, original_text):
    index = faiss.read_index("bias_faiss.index")
    with open("bias_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([original_text], show_progress_bar=True).astype("float32")

    D, I = index.search(query_embedding, 5)  # Retrieve top 5 similar examples
    examples = []
    
    for i in range(len(I[0])):
        if I[0][i] != -1:  # Ensure valid index
            example = {
                "biased": metadata["biased"][I[0][i]],
                "neutral": metadata["neutral"][I[0][i]],
                "bias_type": metadata["bias_type"][I[0][i]]
            }
            examples.append(example)

    return examples

def build_gemini_rewrite_prompt(examples, original_text):
    """
    Build a few-shot prompt for Gemini to rewrite a biased job description in an unbiased way.
    :param examples: List of dicts with 'biased', 'neutral', and 'bias_type' keys.
    :param original_text: The user's job description.
    :return: Formatted prompt string.
    """
    prompt = "You are an expert at detecting and rewriting biased job descriptions to be unbiased and inclusive.\n"
    prompt += "Below are some examples:\n\n"
    for ex in examples:
        prompt += f"Biased: {ex['biased']}\n"
        prompt += f"Unbiased: {ex['neutral']}\n"
        prompt += f"Bias Type: {ex['bias_type']}\n\n"
    prompt += "Now, rewrite the following job description to be unbiased and inclusive. If no bias is found, reply: 'No major bias detected.'\n"
    prompt += f"Job Description: {original_text}\n"
    prompt += "Unbiased:"
    return prompt

from google import genai

def call_gemini_api(prompt):
    """
    Call the Gemini API with the provided prompt.
    :param prompt: The formatted prompt string.
    :return: The response from the Gemini API.
    """
    
    client = genai.Client(api_key="")
    result = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return result.text

@app.route("/", methods=["GET", "POST"])
def index():
    bias_type = None
    original_text = ""

    if request.method == "POST":
        original_text = request.form["jobdesc"]
        raw_label, lime_html = detect_bias_type(original_text)
        bias_type = get_bias_label(raw_label)
        examples = retrieve_example(bias_type, original_text)
        
        print(f"Bias Type: {bias_type}, Examples: {examples}")
        
        prompt = build_gemini_rewrite_prompt(examples, original_text)
        suggestions = call_gemini_api(prompt)
        print(f"Suggestions: {suggestions}")

    return render_template(
        "index.html",
        original=original_text,
        bias_type=bias_type,
        lime_html=lime_html if request.method == "POST" else None,
        suggestions=suggestions if request.method == "POST" else None,
    )

if __name__ == "__main__":
    # setup_faiss_index()
    app.run(debug=True)
