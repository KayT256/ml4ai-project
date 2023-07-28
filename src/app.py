import streamlit as st
from sentiment_analysis import predict_sentiment

# Set up the Streamlit app
st.title('Sentiment Analysis App')
st.write('Enter a paragraph to get its sentiment score (1-5):')

# Get user input
text_input = st.text_area('Input text here:', height=200)

# Perform sentiment analysis and display the result
if st.button('Get sentiment score'):
    sentiment_score = predict_sentiment(text_input, 'data/models/sentiment_analysis_model/')
    st.write(f'Sentiment score: {sentiment_score}')
