from flask import Flask, render_template, request, jsonify

from Dbias.text_debiasing import *;
from Dbias.bias_classification import *;
from Dbias.bias_recognition import *;

app = Flask(__name__)

# Mock rule-based dictionary for rewrite suggestions
rewrite_dict = {
    "young": "energetic",
    "strong": "capable",
    "manpower": "staff",
    "recent graduate": "entry-level candidate"
}

def llm_analyze_bias(text):
    classi_out = classify(text)
    debiased_res = run(text)
    biased_words = recognizer(text)
    
    biased_words_list = []
    for id in range(0, len(biased_words)):
        biased_words_list.append(biased_words[id]['entity'])
    
    highlighted_text = text
    for word in biased_words_list:
        highlighted_text = highlighted_text.replace(word, f"<span class='highlight'>{word}</span>")
    
    suggestions = "No suggestions available."
    if debiased_res != None:
      all_suggestions = []
      for sent in debiased_res[0:3]:
        all_suggestions.append(sent['Sentence'])
      suggestions = "\n\n".join(all_suggestions)

    return {
        classi_out[0]['label']: int(classi_out[0]['score'] * 100),
        }, highlighted_text, suggestions

# Mock analysis logic
def rule_based_analyze_bias(text):
    biases = {
        "gender": 78,
        "age": 65,
        "disability": 40
    }

    highlights = {
        "young": "<span class='highlight'>young</span>",
        "strong": "<span class='highlight'>strong</span>",
        "manpower": "<span class='highlight'>strong</span>",
        "recent graduate": "<span class='highlight'>entry-level candidate</span>"
    }

    suggestion_list = []
    for word, replacement in rewrite_dict.items():
        if word in text:
            suggestion_list.append(f"<li class='list-group-item'>{word} → <span class='suggestion'>{replacement}</span></li>")
    suggestions = "".join(suggestion_list)

    highlighted_text = text
    for word, markup in highlights.items():
        highlighted_text = highlighted_text.replace(word, markup)

    return biases, highlighted_text, suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = ""
    analyzer = request.form.get('analyzer', 'rule-based')

    # Check if a file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            text = file.read().decode('utf-8')
    else:
        text = request.form.get('text', '')

    # Switch-case logic for analyzer
    if analyzer == "rule-based":
        bias_scores, highlighted_text, suggestions = rule_based_analyze_bias(text)
    elif analyzer == "llm":
        # Placeholder for LLM-based analysis
        bias_scores, highlighted_text, suggestions = llm_analyze_bias(text)
    elif analyzer == "ml":
        # Placeholder for ML-based analysis
        print("ML analysis not implemented yet.")
    else:
        # Default fallback
        bias_scores, highlighted_text, suggestions = rule_based_analyze_bias(text)

    return jsonify({
        "bias_scores": bias_scores,
        "highlighted_text": highlighted_text,
        "suggestions": suggestions
    })

import os
from urllib.request import urlretrieve
import zipfile
import site
import shutil

def manual_install_wheel(wheel_url_or_path):
    print("Manually installing a broken wheel for the DBias library")
    if wheel_url_or_path.startswith("http://") or wheel_url_or_path.startswith("https://"):
        local_whl = os.path.join(os.path.basename(wheel_url_or_path))
        print(f"Downloading wheel from {wheel_url_or_path}...")
        urlretrieve(wheel_url_or_path, local_whl)
    else:
        local_whl = wheel_url_or_path
        if not os.path.exists(local_whl): raise FileNotFoundError(f"No such file: {local_whl}")
    print(f"Unpacking {local_whl}...")
    unpack_dir = local_whl.replace(".whl", "_unpacked")
    with zipfile.ZipFile(local_whl, 'r') as zf: zf.extractall(unpack_dir)
    site_packages_dirs = site.getsitepackages()
    if not site_packages_dirs: site_packages_dirs = [site.getusersitepackages()]
    site_packages = site_packages_dirs[0]
    print(f"Copying to site-packages at: {site_packages}")
    for item in os.listdir(unpack_dir):
        src_path = os.path.join(unpack_dir, item)
        dst_path = os.path.join(site_packages, item)
        if os.path.exists(dst_path):
            print(f"Overwriting existing: {dst_path}")
            if os.path.isdir(dst_path): shutil.rmtree(dst_path)
            else: os.remove(dst_path)
        if os.path.isdir(src_path): shutil.copytree(src_path, dst_path)
        else: shutil.copy2(src_path, dst_path)
    print(f"Installed {local_whl} manually into site-packages.")


if __name__ == '__main__':
    wheel_url = "https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl"
    wheel_filename = os.path.basename(wheel_url)
    unpack_dir = wheel_filename.replace(".whl", "_unpacked")
    site_packages_dirs = site.getsitepackages()
    if not site_packages_dirs:
        site_packages_dirs = [site.getusersitepackages()]
    site_packages = site_packages_dirs[0]
    # Check if the main package directory exists in site-packages
    already_installed = os.path.exists(os.path.join(site_packages, "en_pipeline"))
    if not already_installed:
        # Download wheel only if not already downloaded
        if not os.path.exists(wheel_filename):
            print(f"Downloading wheel from {wheel_url}...")
            urlretrieve(wheel_url, wheel_filename)
        # Unpack and install only if not already unpacked
        if not os.path.exists(unpack_dir):
            manual_install_wheel(wheel_filename)
        else:
            # Check if installed in site-packages
            print(f"Unpacked directory {unpack_dir} already exists. Checking installation...")
            if not already_installed:
                manual_install_wheel(wheel_filename)
            else:
                print("Wheel already installed in site-packages.")
    else:
        print("Wheel/package already installed. Skipping manual installation.")
    app.run(debug=True)