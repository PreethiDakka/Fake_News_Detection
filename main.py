from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

model= pickle.load(open('svm_model.pkl', 'rb'))  # Load the trained model
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # Load the vectorizer

@app.route('/')
def home():
    return render_template('index.html')


def clean_text(text):
    import re
    import string
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    # text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        text = request.form['text']

        # Combine and clean the input
        combined_input = clean_text(f"{title} {text}")

        # Vectorize the cleaned input
        tfidf_features = vectorizer.transform([combined_input]).toarray()  # shape: (1, num_features)

        # Predict using the trained model
        prediction = model.predict(tfidf_features)[0]
        label = "True News" if prediction == 1 else "Fake News"

        return render_template('index.html', prediction_text=f"Prediction: {label}")
if __name__ == '__main__':
    app.run(debug=True)
