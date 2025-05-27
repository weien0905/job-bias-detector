from flask import Flask, render_template, request
import re

app = Flask(__name__)

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

@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    highlighted_text = ""
    feminine = []
    masculine = []
    gender = []
    marital = []
    appearance = []
    age = []
    religion = []
    wpt_matches = []

    if request.method == "POST":
        original_text = request.form["jobdesc"]
        (
            feminine, masculine,
            gender, marital,
            appearance, age,
            religion
        ) = detect_bias_words(original_text)

        wpt_matches = detect_wpt_patterns(original_text)

        highlighted_text = highlight_bias(
            original_text, feminine, masculine,
            gender, marital, appearance,
            age, religion
        )

    return render_template(
        "index.html",
        original=original_text,
        highlighted=highlighted_text,
        feminine=feminine,
        masculine=masculine,
        gender=gender,
        marital=marital,
        appearance=appearance,
        age=age,
        religion=religion,
        wpt=wpt_matches
    )

if __name__ == "__main__":
    app.run(debug=True)
