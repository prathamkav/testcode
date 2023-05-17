from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def make_dataset(dataframe, lookup, batch_size, is_train=True):
    labels = tf.ragged.constant(dataframe["teams"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["processed_body"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


def invert_multi_hot(vocab, encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)


def make_model(lookup_size):
    shallow_mlp_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(lookup_size, activation="sigmoid"),
        ]  # More on why "sigmoid" has been used here in a moment.
    )
    return shallow_mlp_model

def preprocess(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into string
    text = ' '.join(filtered_tokens)
    return text
