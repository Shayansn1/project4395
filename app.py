from flask import Flask, request, render_template, jsonify
import os
import PyPDF2
from predict_match import predict_match
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GPT_TOKEN = os.environ.get("OPENAI_API_KEY")
GPT_ENDPOINT = "https://models.github.ai/inference"
GPT_MODEL = "openai/gpt-4.1-mini"

def get_gpt_feedback(score, job, resume):
    prompt = f"""
    The resume currently scores {score}% in matching this job description.

    Job Description:
    {job}

    Candidate Resume:
    {resume}

    Suggest clear and actionable improvements to increase the accuracy of this resume for the given job in 3 sentences.
    """

    client = OpenAI(
        base_url=GPT_ENDPOINT,
        api_key=GPT_TOKEN,
    )

    try:
        response = client.chat.completions.create(
            messages = [
                {"role": "system", "content": "You are an expert career advisor specialized in optimizing resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            top_p=1.0,
            model = GPT_MODEL
        )
        res = response.choices[0].message.content
        return res.strip()
    except Exception as e:
        print("GPT Feedback Error:", str(e))
        return "Could not generate feedback. Please try again."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    resume_file = request.files['resume']
    job_description = request.form['job_description']
    resume_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
    resume_file.save(resume_path)

    ext = os.path.splitext(resume_path)[1].lower()
    if ext == '.pdf':
        with open(resume_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            resume_text = ' '.join(page.extract_text() or '' for page in reader.pages)
    elif ext == '.txt':
        with open(resume_path, 'r', encoding='utf-8') as f:
            resume_text = f.read()
    else:
        return jsonify({'error': 'Unsupported file format'}), 400

    score = predict_match(resume_text, job_description)

    feedback = get_gpt_feedback(score, job_description, resume_text)
    
    if os.path.exists(resume_path):
        os.remove(resume_path)

    return jsonify({
        'match_score': score,
        'feedback': feedback
    })

if __name__ == '__main__':
    app.run()
