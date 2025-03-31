import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Get user input
var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))

# Function for making the prediction
def detecting_fake_news(var):
    # Load the model and the vectorizer
    load_model = pickle.load(open('path_to_final_model/final_model.sav', 'rb'))
    vectorizer = pickle.load(open('path_to_vectorizer/vectorizer.sav', 'rb'))
  # Adjust path if necessary

    # Preprocess the input (vectorize it)
    var_vectorized = vectorizer.transform([var])

    # Predict the class and the probability
    prediction = load_model.predict(var_vectorized)
    prob = load_model.predict_proba(var_vectorized)

    # Output the result
    print("The given statement is:", prediction[0])
    print("The truth probability score is:", prob[0][1])

# Run the function
if __name__ == '__main__':
    detecting_fake_news(var)
