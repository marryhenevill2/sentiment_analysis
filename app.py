import streamlit as st
import joblib

# create page title
st.title("Customer Feedback Sentiment Analyzer App")
st.write("Enter customer feedback below to analyze its sentiment.")

# load the trained model
@st.cache_resource
def load_model():
    return joblib.load("customer_feedback_sentiment_pipeline.joblib")

try: 
    pipeline = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'customer_feedback_sentiment_pipeline.joblib' is in the correct directory.")
    st.stop()

# User input for customer feedback
feedback = st.text_area("Customer Feedback", placeholder="Type your feedback here...", height=150)    

# predict sentiment
if st.button("Analyze Sentiment"):
    if not feedback.strip():
        st.warning("Please enter some feedback to analyze.")
    else:
        prediction = pipeline.predict([feedback])[0]
        st.success (f"The sentiment of the feedback is: {prediction.capitalize()}")

        if hasattr(pipeline, 'predict_proba'):
            probabilities = pipeline.predict_proba([feedback])[0]
            st.write(f"Probability of Positive Sentiment: {probabilities[1]:.2f}")
            st.write(f"Probability of Negative Sentiment: {probabilities[0]:.2f}")
            confidence = max(probabilities) * 100
            st.write(f"Confidence Level: {confidence:.2f}")





