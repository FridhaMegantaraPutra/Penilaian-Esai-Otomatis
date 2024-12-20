import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def calculate_cosine_similarity(reference_essay, student_essay):
    reference_essay = preprocess_text(reference_essay)
    student_essay = preprocess_text(student_essay)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([reference_essay, student_essay])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return similarity[0][0]


def convert_similarity_to_score(similarity):
    return round(similarity * 100, 2)
