from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Mock rule-based dictionary for rewrite suggestions
rewrite_dict = {
    "young": "energetic",
    "strong": "capable",
    "manpower": "staff",
    "recent graduate": "entry-level candidate"
}

# Mock analysis logic
def analyze_bias(text):
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

    suggestions = []
    for word, replacement in rewrite_dict.items():
        if word in text:
            suggestions.append({
                "original": word,
                "suggestion": replacement
            })

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

    # Check if a file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            text = file.read().decode('utf-8')  # Read content directly without saving
    else:
        text = request.form.get('text', '')

    bias_scores, highlighted_text, suggestions = analyze_bias(text)
    return jsonify({
        "bias_scores": bias_scores,
        "highlighted_text": highlighted_text,
        "suggestions": suggestions
    })
if __name__ == '__main__':
    app.run(debug=True)