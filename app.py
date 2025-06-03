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

# Feminine-coded and masculine-coded words
feminine_coded_words = [
    "agree", "affectionate", "child", "cheer", "collab", "commit", "communal",
    "compassion", "connect", "considerate", "cooperat", "co-operat", "depend",
    "emotiona", "empath", "feel", "flatterable", "gentle", "honest",
    "interpersonal", "interdependen", "interpersona", "inter-personal",
    "inter-dependen", "inter-persona", "kind", "kinship", "loyal", "modesty",
    "nag", "nurtur", "pleasant", "polite", "quiet", "respon", "sensitiv",
    "submissive", "support", "sympath", "tender", "together", "trust",
    "understand", "warm", "whin", "enthusias", "inclusive", "yield", "shar"
]

masculine_coded_words = [
    "active", "adventurous", "aggress", "ambitio", "analy", "assert",
    "athlet", "autonom", "battle", "boast", "challeng", "champion",
    "compet", "confident", "courag", "decid", "decision", "decisive",
    "defend", "determin", "domina", "dominant", "driven", "fearless",
    "fight", "force", "greedy", "head-strong", "headstrong", "hierarch",
    "hostil", "implusive", "independen", "individual", "intellect", "lead",
    "logic", "objective", "opinion", "outspoken", "persist", "principle",
    "reckless", "self-confiden", "self-relian", "self-sufficien",
    "selfconfiden", "selfrelian", "selfsufficien", "stubborn", "superior",
    "unreasonab"
]

gender_specific_words = [
    "female", "females", "lady", "ladies", "woman", "women", "girl", "girls",
    "male", "males", "man", "men", "boy", "boys"
]

marital_status_words = [
    "single", "not married", "unmarried", "married"
]

physical_appearance_words = [
    "height", "weight", "face", "body", "physically", "physical",
    "good looking", "good appearance", "proportional", "attractive"
]

age_bias_words = [
    "age", "years old", "year old", "aged", "young", "youthful", "old", "elderly"
]

religion_bias_words = [
    "religion", "religious", "muslim", "islam", "christian", "catholic",
    "protestant", "hindu", "buddhist", "jewish", "atheist", "non-religious",
    "hijab", "headscarf", "headscarves", "niqab", "veil", "modest", "islamic"
]

# WPT patterns with named labels
wpt_patterns = {
    "We are family": r"\bwe (are|’re|'re) (like )?(a )?family\b",
    "Young and energetic": r"\byoung (and )?(dynamic|energetic|team)\b",
    "Must be attractive": r"\bmust be (physically )?(attractive|good looking)\b",
    "Single preferred": r"\b(single (candidates )?preferred|must be single)\b",
    "Work hard play hard": r"\b(work hard,? play hard)\b",
    "Always on call": r"\b(24/7|always available|round the clock)\b",
    "No 9 to 5": r"\bnot (just )?(a )?9[- ]?to[- ]?5\b",
    "High-pressure environment": r"\b(high[- ]pressure|stressful|fast[- ]paced) (job|environment|role)\b",
    "Fit in culturally": r"\b(cultural (fit|match)|fit in( well)? with (the )?team)\b",

    # Age Bias
    "Recent graduate preferred": r"\b(recent (graduate|grad)|fresh out of (college|university))\b",
    "Digital native": r"\bdigital native(s)?\b",

    # Gender Bias
    "Looking for a strong man": r"\b(looking for|must be) (a )?(strong|tough) man\b",
    "Female-only roles": r"\b(female(s)? (only|preferred)|women (only|preferred))\b",

    # Disability Bias
    "Must be able-bodied": r"\bmust be (able[- ]bodied|physically fit|no physical limitations)\b",

    # Family/Marital Status Bias
    "No family obligations": r"\b(no (family|personal) obligations|required to travel frequently)\b",

    # Religious Bias
    "Sunday availability required": r"\b(must work on Sundays|no religious holidays)\b",

    # Cultural Stereotyping / Personality Fit
    "Beer Fridays culture": r"\b(beer|booze) (fridays|culture|after work)\b",
    "Ninja/Rockstar": r"\b(code|sales|marketing)? ?(ninja|rockstar|guru|wizard)\b",

    # Socioeconomic Bias
    "Own vehicle a must": r"\bmust (have|own) (a )?(car|vehicle)\b",

    # Time Commitment / Overwork Expectation
    "Burning the midnight oil": r"\b(burn(ing)? the midnight oil|work long hours)\b",

    # Language/National Origin Bias
    "Native English speaker only": r"\b(native (english )?speaker (only|required))\b"
}


# Additional WPT patterns (regex based) for implicit age filters
wpt_age_patterns = [
    r'\bage\s+\d{1,2}\s*(?:yo|years?\s*old)?\b',
    r'\bage\s+(?:between|above|below|under|over)?\s*\d{1,2}\s*(?:yo|years?\s*old)?\b',
    r'\bunder\s+\d{1,2}\s*(?:yo|years?\s*old)?\b',
    r'\bover\s+\d{1,2}\s*(?:yo|years?\s*old)?\b'
]

def detect_bias_words(text):
    feminine_matches = []
    masculine_matches = []
    gender_matches = []
    marital_matches = []
    appearance_matches = []
    age_matches = []
    religion_matches = []

    for word in feminine_coded_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            feminine_matches.append(word)

    for word in masculine_coded_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            masculine_matches.append(word)

    for word in gender_specific_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            gender_matches.append(word)

    for word in marital_status_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            marital_matches.append(word)

    for word in physical_appearance_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            appearance_matches.append(word)

    for word in age_bias_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            age_matches.append(word)

    for word in religion_bias_words:
        if re.search(rf'\b{word}\w*\b', text, re.IGNORECASE):
            religion_matches.append(word)

    return (
        feminine_matches, masculine_matches,
        gender_matches, marital_matches,
        appearance_matches, age_matches,
        religion_matches
    )

def detect_wpt_patterns(text):
    matched_patterns = []

    # Named WPT patterns
    for label, pattern in wpt_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            matched_patterns.append(label)

    # Regex-only WPT patterns (e.g., implicit age filters)
    for pattern in wpt_age_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            matched_patterns.append("Age-related requirement")

    return list(set(matched_patterns))

def highlight_bias(text, feminine, masculine, gender, marital, appearance, age, religion):
    highlighted = text

    for word in set(feminine):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:lightpink;">\1</mark>', highlighted, flags=re.IGNORECASE)

    for word in set(masculine):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:lightblue;">\1</mark>', highlighted, flags=re.IGNORECASE)

    for word in set(gender):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:#FFD700;">\1</mark>', highlighted, flags=re.IGNORECASE)

    for word in set(marital):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:#90EE90;">\1</mark>', highlighted, flags=re.IGNORECASE)

    for word in set(appearance):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:#FFA07A;">\1</mark>', highlighted, flags=re.IGNORECASE)

    for word in set(age):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:#E6E6FA;">\1</mark>', highlighted, flags=re.IGNORECASE)

    for word in set(religion):
        highlighted = re.sub(rf'({word}\w*)', r'<mark style="background-color:#D8BFD8;">\1</mark>', highlighted, flags=re.IGNORECASE)

    return highlighted

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

def detect_bias_type(text):
    model_path = "./model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    class_names = list(model.config.id2label.values())
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    explainer = LimeTextExplainer(class_names=class_names)

    result = pipe(text)

    explanation = explainer.explain_instance(
        text, 
        lambda x: np.array([ [label['score'] for label in pipe(xx)[0]] for xx in x ]),
        num_features=5, 
        top_labels=1
    )
    explanation.save_to_file("lime_explanation.html")
    
    with open("lime_explanation.html", "r") as f:
        lime_html = f.read()
    
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
    
    client = genai.Client(api_key="AIzaSyB0HHC3_4eOLe5sgCvd7tOn6rlWP9VZnPo")
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
