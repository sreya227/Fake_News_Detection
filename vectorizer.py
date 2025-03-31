import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Example text data to train the vectorizer (replace with your data)
corpus = [
    "This is a sample news article.",
    "Another example of a news article.",
    "This news article talks about politics."
]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the corpus
vectorizer.fit(corpus)

# Save the vectorizer to a file
with open('vectorizer.sav', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Vectorizer saved successfully as 'vectorizer.sav'")
