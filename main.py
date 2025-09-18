import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os # To handle file paths

# --- Configuration ---
# Set page title and favicon
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="centered"
)

# --- Custom Styles (Minimal to keep it simple and clean, but match your alerts) ---
st.markdown("""
    <style>
    /* Streamlit's main content wrapper for top margin */
    .css-1d3f8gv {
        padding-top: 2rem; /* Adjusted for simpler spacing */
    }
    .stAlert {
        border-radius: 0.25rem;
        padding: 0.75rem 1.25rem;
        margin-top: 1.5rem; /* Added margin for separation */
        border: 1px solid transparent;
        font-size: 1.1em;
        font-weight: 600;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stAlert.stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    .stAlert.stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-color: #ffeeba;
    }
    .stAlert.stError { /* Streamlit uses stError for 'danger' */
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    .stAlert.stInfo { /* For unknown sentiment, using stInfo for secondary */
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
    }
    </style>
""", unsafe_allow_html=True)

# --- NLTK Downloads (Run only once) ---
# Use st.cache_data to ensure NLTK downloads are handled efficiently
@st.cache_data(show_spinner=False)
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        with st.spinner("Downloading NLTK stopwords..."):
            nltk.download('stopwords', quiet=True)
    return True

download_nltk_data()

# --- Load Model and TF-IDF Vectorizer ---
# Adjust path for Streamlit Cloud - assume models are in the same directory as app.py
# Or, if they are in .venv, make sure .venv is packaged correctly or adjust paths
# For simplicity, let's assume clf.pkl and tfidf.pkl are in the same directory as this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'clf.pkl')
TFIDF_PATH = os.path.join(os.path.dirname(__file__), 'tfidf.pkl')

@st.cache_resource # Cache the model loading for efficiency
def load_resources():
    try:
        with open(MODEL_PATH, 'rb') as f:
            clf_loaded = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f:
            tfidf_loaded = pickle.load(f)
        return clf_loaded, tfidf_loaded
    except FileNotFoundError:
        st.error(f"Error: Model or TF-IDF file not found. "
                 f"Please ensure '{MODEL_PATH}' and '{TFIDF_PATH}' exist in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or TF-IDF: {e}")
        st.stop()

clf, tfidf = load_resources()

# --- Global NLTK objects (initialized once) ---
stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')
porter = PorterStemmer()

# --- Preprocessing Function ---
def preprocessing(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Extract emojis (simplified as they are typically removed/ignored by tokenizers)
    emojis = emoticon_pattern.findall(text)

    # Remove non-word characters and lowercase
    text = re.sub(r'\W+', ' ', text.lower()) + ' '.join(emojis).replace('-', '')

    # Stem and remove stopwords
    processed_words = [porter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(processed_words)

# --- Streamlit UI ---
st.title("Sentiment Analysis")

st.write("Enter your comment below and click 'Analyze Sentiment' to see its emotional tone.")

comment = st.text_area(
    "Your Comment:",
    height=150,
    placeholder="e.g., 'This movie was absolutely fantastic, I loved every minute of it!'"
)

if st.button("Analyze Sentiment"):
    if comment.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        # Display result
        if sentiment == 2:
            st.success("😊 Positive comment!")
        elif sentiment == 1:
            st.warning("😐 Neutral comment!")
        elif sentiment == 0:
            st.error("☹️ Negative comment!")
        else:
            st.info("⚠️ Unknown sentiment!")

st.markdown("---")
