import pickle
import re

import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure the required NLTK resources are downloaded
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('.venv/clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('.venv/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Extract emojis
    emojis = emoticon_pattern.findall(text)

    # Remove non-word characters and lowercase
    text = re.sub(r'\W+', ' ', text.lower()) + ' '.join(emojis).replace('-', '')

    # Stem and remove stopwords
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
