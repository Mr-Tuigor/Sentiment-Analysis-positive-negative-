import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Set up the Streamlit interface
st.title("Movie Review Sentiment Analyzer 🍿")
st.write("Enter a movie review below to see if it's positive or negative!")

# Text input box for the user
user_input = st.text_area("Write your review here:")

# Button to trigger the prediction
if st.button("Analyze Sentiment"):
    if user_input:
        # Convert the user's text into numbers using our saved vectorizer
        vec_input = vectorizer.transform([user_input])
        
        # Make the prediction
        prediction = model.predict(vec_input)[0]
        
        # Display the results
        if prediction == 1:
            st.success("This review is Positive! 😄")
        else:
            st.error("This review is Negative! 😞")
    else:
        st.warning("Please enter a review first.")
