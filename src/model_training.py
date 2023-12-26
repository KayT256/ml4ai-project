import tensorflow as tf
import os
from transformers import TFAutoModelForSequenceClassification
from data_preprocessing import load_data, preprocess_data

# Set the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def train_model(train_file_path, val_file_path, model_save_path):
    train_data = load_data(train_file_path)
    val_data = load_data(val_file_path)
    train_tokenized, train_labels = preprocess_data(train_data)
    val_tokenized, val_labels = preprocess_data(val_data)

    # Pre-trained BERT model
    model = TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.CategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    history = model.fit(dict(train_tokenized), train_labels, validation_data=(dict(val_tokenized), val_labels), batch_size=32, epochs=1)

    model.save_pretrained(model_save_path)

train_model('../data/SST-2/train.tsv', '../data/SST-2/dev.tsv', '../data/models/sentiment_analysis_model')