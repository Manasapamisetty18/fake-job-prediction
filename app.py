from flask import Flask, request, render_template
import pandas as pd
from scipy.sparse import hstack
import pickle
import requests
from bs4 import BeautifulSoup


app = Flask(__name__)


# Load ML models and vectorizer

tfidfs = pickle.load( open('vectorizers.pickle', 'rb') )
Passives = pickle.load( open('classifiers.pickle', 'rb') )
mlp = pickle.load(open('mlp.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
Passive = pickle.load(open('Passive.pkl', 'rb'))
Gradient = pickle.load(open('Gradient.pkl', 'rb'))

# Internshala Scraper
def scrape_internshala(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        blocks = soup.find_all('div', {'class': ['text-container', 'text-container additional_detail']})
        if blocks:
            return " ".join([b.get_text(" ", strip=True) for b in blocks])
    except Exception as e:
        print(f"Internshala scraping error: {e}")
    return None



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
    
@app.route('/liveprediction')
def liveprediction():
    return render_template('liveprediction.html')
    
@app.route('/livepredict', methods=['POST'])
def livepredict():
    if request.method == 'POST':
        url = request.form['url'].strip()
        # model_name = request.form['Model']

        if 'internshala.com' in url:
            description = scrape_internshala(url)
      
        else:
            return "Unsupported job portal."

        if not description:
            return "Failed to extract job description. Please check the URL."

        try:
            # TF-IDF
            tfidf_vec = tfidfs.transform([description])

           

            # Combine
            full_input = hstack([tfidf_vec])

           
            pred = Passives.predict(full_input)
           

            result = "Fraudulent" if pred[0] == 1 else "Legitimate"

            return render_template('liveresult.html',
                                   prediction_text=result,
                                   description=url)
        except Exception as e:
            return f"Prediction error: {str(e)}"

@app.route('/liveresult')
def liveresult():
    return render_template('liveresult.html')


@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        description = str(request.form['news'])
        Telecommuting = int(request.form['Telecommuting'])
        Has_company_logo = int(request.form['Has_company_logo'])
        Has_questions = int(request.form['Has_questions'])
        Employment_type = int(request.form['Employment_type'])
        Required_experience = int(request.form['Required_experience'])
        Required_education = int(request.form['Required_education'])
        Function = int(request.form['Function'])
        model = request.form['Model']

        # Map form inputs to readable strings
        Telecommutings = 'Yes' if Telecommuting == 1 else 'No'
        Has_company_logos = 'Yes' if Has_company_logo == 1 else 'No'
        Has_questionss = 'Yes' if Has_questions == 1 else 'No'

        employment_map = {
            1: 'Full-time', 2: 'Part-time', 3: 'Contract',
            4: 'Temporary', 5: 'Other', 6: 'Not Mentioned'
        }
        Employment_types = employment_map.get(Employment_type, 'Unknown')

        experience_map = {
            1: 'Mid-Senior level', 2: 'Executive', 3: 'Entry level', 4: 'Associate',
            5: 'Not Applicable', 6: 'Director', 7: 'Internship', 8: 'Not Mentioned'
        }
        Required_experiences = experience_map.get(Required_experience, 'Unknown')

        education_map = {
            1: 'Master Degree', 2: 'Bachelor Degree', 3: 'Unspecified', 4: 'High School or equivalent',
            5: 'Associate Degree', 6: 'Vocational', 7: 'Vocational - HS Diploma',
            8: 'Professional', 9: 'Some High School Coursework', 10: 'Some College Coursework Completed',
            11: 'Certification', 12: 'Doctorate', 13: 'Vocational - Degree',
            14: 'Not Mentioned required_education'
        }
        Required_educations = education_map.get(Required_education, 'Unknown')

        function_map = {
            1: 'Marketing', 2: 'Customer Service', 3: 'Information Technology', 4: 'Sales',
            5: 'Health Care Provider', 6: 'Management', 7: 'Other', 8: 'Engineering',
            9: 'Administrative', 10: 'Design', 11: 'Production', 12: 'Education',
            13: 'Supply Chain', 14: 'Business Development', 15: 'Product Management',
            16: 'Financial Analyst', 17: 'Consulting', 18: 'Human Resources', 22: 'Project Management',
            23: 'Manufacturing', 24: 'Public Relations', 25: 'Strategy/Planning',
            26: 'Advertising', 27: 'Finance', 28: 'General Business', 29: 'Research',
            30: 'Accounting/Auditing', 31: 'Art/Creative', 32: 'Quality Assurance',
            33: 'Data Analyst', 34: 'Business Analyst', 35: 'Writing/Editing',
            36: 'Distribution', 37: 'Science', 38: 'Training', 39: 'Purchasing',
            40: 'Legal', 41: 'Not Mentioned function'
        }
        Functions = function_map.get(Function, 'Unknown')

        # Prepare data for prediction
        sample = {
            'description': [description],
            'Telecommuting': [Telecommuting],
            'Has_company_logo': [Has_company_logo],
            'Has_questions': [Has_questions],
            'Employment_type': [Employment_type],
            'Required_experience': [Required_experience],
            'Required_education': [Required_education],
            'Function': [Function]
        }
        sample_df = pd.DataFrame(sample)

        sample_text_tfidf = tfidf.transform(sample_df['description'])
        sample_numeric = sample_df.drop('description', axis=1).values
        sample_combined = hstack([sample_text_tfidf, sample_numeric])

        # Predict
        if model == "MLPClassifier":
            RESULT = mlp.predict(sample_combined)
        elif model == "PassiveAggressiveClassifier":
            RESULT = Passive.predict(sample_combined)
        elif model == "GradientBoostingClassifier":
            RESULT = Gradient.predict(sample_combined)
        else:
            RESULT = [0]  # Default to legitimate if model not found

        result = "Fraudulent" if RESULT[0] == 1 else "Legitimate"

        return render_template('result.html',
                               prediction_text=result,
                               model=model,
                               description=description,
                               Telecommutings=Telecommutings,
                               Has_company_logo=Has_company_logos,
                               Has_questionss=Has_questionss,
                               Employment_types=Employment_types,
                               Required_experiences=Required_experiences,
                               Required_educations=Required_educations,
                               Functions=Functions)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
