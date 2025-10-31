# Human Activity Recognition System using UCI HAR Dataset

# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 2: Load the dataset
def load_data():
    INPUT_SIGNAL_TYPES = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
    DATA_PATH = "UCI HAR Dataset/"
    X_train = []
    X_test = []

    for signal in INPUT_SIGNAL_TYPES:
        X_train.append(np.loadtxt(DATA_PATH + "train/Inertial Signals/" + signal + "_train.txt"))
        X_test.append(np.loadtxt(DATA_PATH + "test/Inertial Signals/" + signal + "_test.txt"))

    X_train = np.transpose(np.array(X_train), (1, 2, 0))
    X_test = np.transpose(np.array(X_test), (1, 2, 0))

    y_train = pd.read_csv(DATA_PATH + "train/y_train.txt", header=None)[0].values
    y_test = pd.read_csv(DATA_PATH + "test/y_test.txt", header=None)[0].values

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# Step 3: Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))  # 6 activity classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Step 6: Visualize training
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Step 7: Classification Report
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_true_labels, y_pred_labels))

# Step 8: Predict on a single sample
activity_labels = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

sample = X_test[0].reshape(1, 128, 9)
prediction = model.predict(sample)
print("Predicted Activity:", activity_labels[np.argmax(prediction)])
