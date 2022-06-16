import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras import layers
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import re
import string

print(tf.__version__)


!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"


train_data = pd.read_csv(train_file_path, sep="\t", names=['type', 'message'])
test_data = pd.read_csv(test_file_path, sep="\t", names=['type', 'message'])

print(train_data)
print(test_data)


train_data["type"] = train_data["type"].map({"ham": 0, "spam": 1})
test_data["type"] = test_data["type"].map({"ham": 0, "spam": 1})

train_labels = train_data["type"]
test_labels = test_data["type"]

train_data = train_data["message"].values.flatten()
test_data = test_data["message"].values.flatten()


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=300
)

vectorize_layer.adapt(train_data)


model = keras.Sequential()

model.add(vectorize_layer)
model.add(layers.Embedding(10000, 10))
model.add(layers.Dropout(0.2))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))


model.compile(loss=keras.losses.binary_crossentropy, 
              optimizer=tf.optimizers.Adam(learning_rate=0.001),
              metrics="accuracy")
              
              
model.fit(train_data, train_labels, validation_split=0.2, epochs=20, verbose=1)


def return_class_name(key):
  return {0: "ham", 1: "spam"}[key]
  
  
def predict_message(pred_text):
  result = model.predict([pred_text])[0][0]
  print(result)
  prediction = [result, return_class_name(round(result))]

  return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)


def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
