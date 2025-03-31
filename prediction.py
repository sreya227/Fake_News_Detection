import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Get user input
var = input("Please enter the news text you want to verify: ")
print("You entered:", var)

# Function for making the prediction
def detecting_fake_news(var):
    # Correct file paths (adjust if needed)
    model_path = r"C:\Users\sreya\Fake_News_Detection\final_model.sav"
    vectorizer_path = r"C:\Users\sreya\Fake_News_Detection\vectorizer.sav"

    # Load the model and vectorizer
    with open(model_path, 'rb') as model_file:
        load_model = pickle.load(model_file)

    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Preprocess the input (vectorize it)
    var_vectorized = vectorizer.transform([var])

    # Predict the class and probability
    prediction = load_model.predict(var_vectorized)
    prob = load_model.predict_proba(var_vectorized)

    # Output the result
    print("The given statement is:", prediction[0])
    print("The truth probability score is:", prob[0][1])

# Run the function
if __name__ == '__main__':
    detecting_fake_news(var)
