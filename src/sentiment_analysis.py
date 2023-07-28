import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def predict_sentiment(text, model_load_path):
    """Perform sentiment analysis on input text using the trained model."""
    # Load the pre-trained BERT model and tokenizer
    model = TFAutoModelForSequenceClassification.from_pretrained(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the input text
    tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='tf')

    # Make a prediction
    logits = model(tokenized_text)[0]
    probabilities = tf.nn.softmax(logits, axis=1)
    sentiment_score = tf.argmax(probabilities, axis=1).numpy()[0] + 1

    return sentiment_score
