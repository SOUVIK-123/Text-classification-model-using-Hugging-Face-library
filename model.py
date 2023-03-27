import pandas as pd
import numpy as  np

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.datasets import fetch_20newsgroups
# Load the newsgroups dataset
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# Convert data to pandas dataframes
train_df = pd.DataFrame({'text': newsgroups_train.data, 'label':
newsgroups_train.target})
test_df = pd.DataFrame({'text': newsgroups_test.data, 'label':
newsgroups_test.target})
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Tokenize the data and create encodings
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)
# Convert category labels to integers
le = LabelEncoder()
train_labels = le.fit_transform(train_df['label'])
test_labels = le.transform(test_df['label'])
# Load BERT model and set number of labels
model =
TFAutoModelForSequenceClassification.from_pretrained('bert-base-uncased',
num_labels=len(newsgroups_train.target_names))
# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# Prepare training and test data as tf.data.Dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings['input_ids'],
train_labels)).shuffle(len(train_df)).batch(4)
test_dataset = tf.data.Dataset.from_tensor_slices((test_encodings['input_ids'],
test_labels)).shuffle(len(test_df)).batch(4)
# Train the model for three epochs
model.fit(train_dataset, epochs=3, validation_data=test_dataset)
# Use the model to predict categories for the test data
predictions = model.predict(test_dataset)
# Convert predicted labels back to category names
predicted_categories = np.argmax(predictions.logits, axis=-1)
predicted_category_names = le.inverse_transform(predicted_categories)
# Print classification report
print(classification_report(test_df['label'], predicted_category_names))
