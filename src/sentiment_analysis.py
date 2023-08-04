import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def predict_sentiment(text, model_load_path):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    tokenized_text = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='tf')

    logits = model(tokenized_text)[0]
    probabilities = tf.nn.softmax(logits, axis=1)
    sentiment_score = tf.argmax(probabilities, axis=1).numpy()[0] + 1

    return probabilities[0, 1]

def map_sentiment_score(score):
    if score < 0.2:
        return 'Negative'
    elif score < 0.4:
        return 'Fairly negative'
    elif score < 0.6:
        return 'neutral'
    elif score < 0.8:
        return 'Fairlypositive'
    else:
        return 'Positive'
