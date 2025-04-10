from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF

# Correct Flask paths based on src/ structure
app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static'
)

# Function to extract text from uploaded PDF resume
def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')
@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/matcher')
def matcher():
    return render_template('matcher.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recruiter')
def recruiter():
    return render_template('recruiter.html')

@app.route('/job-details1')
def job_details1():
    return render_template('job-details1.html')

@app.route('/job-details2')
def job_details2():
    return render_template('job-details2.html')

# Resume analysis endpoint
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    file = request.files['resume']
    jd = request.form['jd']

    resume_text = extract_pdf_text(file)
    documents = [jd, resume_text]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    return jsonify({
        "summary": resume_text[:300] + "...",
        "match_score": round(score * 100, 2)
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)