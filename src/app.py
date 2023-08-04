import streamlit as st
from sentiment_analysis import predict_sentiment, map_sentiment_score

# Set up the Streamlit app
st.title('Sentiment Analysis App')
st.write('Enter a paragraph to get its sentiment:')

# Get user input
text_input = st.text_area('Input text here:', height=200)

# Perform sentiment analysis and display the result
if st.button('Get sentiment score'):
    sentiment_score = predict_sentiment(text_input, 'data/models/sentiment_analysis_model/')
    score = map_sentiment_score(sentiment_score)
    st.write(f'Sentiment: {score}')
    
