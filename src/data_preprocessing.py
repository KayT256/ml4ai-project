import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
    data.columns = data.iloc[0]
    return data[1:]

def preprocess_data(data):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenized_data = tokenizer(data['sentence'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='tf')
    labels = tf.keras.utils.to_categorical(data['label'], num_classes=2, dtype='float32')
    return tokenized_data, labels
