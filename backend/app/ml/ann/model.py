"""
ANN Model Definition.
Defines the Neural Network architecture for classification/prediction.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

class ChronicRiskANN:
    """Neural Network model for chronic disease risk prediction."""
    
    def __init__(self, input_dim=8, dropout_rate=0.2):
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(units=16, activation="relu", input_shape=(self.input_dim,)))
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=8, activation="relu"))
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def fit(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=None, verbose=1):
        """Trains the underlying Keras model."""
        return self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data, returning (loss, accuracy)."""
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict(self, X):
        """Generates risk probabilities for the input samples."""
        return self.model.predict(X)

    def save(self, filepath):
        """Saves the Keras model to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.model.save(filepath)

    def load(self, filepath):
        """Loads the Keras model from a file."""
        self.model = load_model(filepath)
