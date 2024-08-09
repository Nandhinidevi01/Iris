import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

df = pd.read_csv('IRIS.csv')

df.head()

species = df['species'].value_counts()
species

df.dtypes

df.shape

df.describe()

df.isnull().sum()

plt.figure(figsize=(8,8))
plt.pie(species, labels = species.index, autopct='%1.1f%%')
plt.legend(loc='lower left')
plt.show()

sns.FacetGrid(df, hue ='species').map(plt.scatter,"petal_length","sepal_width").add_legend()
plt.show()

sns.countplot(x='species', data=df)
plt.show()

X = df.drop('species', axis = 1) # grabs everything else but 'species'

# Create target variable
y = df['species'] # y is the column we're trying to predict


# Standardize the features
scaler = StandardScaler().fit(X)
x_transform = scaler.transform(X)


# Encode the target variable
lb_encoder = LabelEncoder()
y_encoded = lb_encoder.fit_transform(y)
# One-hot encode the target labels
y_onehot = tf.keras.utils.to_categorical(y_encoded)
#spliting the data in training and testing
X_train, X_test, y_train, y_test = train_test_split(x_transform, y_onehot, test_size = 0.20, random_state = 42)


#Building a Fully connected neural network (FCN) mode

model = Sequential()

#1st hidden layer
model.add(Dense(50, input_dim = 4, activation = 'relu'))

#2nd Layer
model.add(Dense(100, activation = 'relu'))

#3rd Layer
#softmax activation function is use for Categorial Data
# the output values represent probabilities of each category
model.add(Dense(3, activation = 'softmax'))

#compiling the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()
Model: "sequential"
    
model.fit(X_train, y_train, epochs = 30, batch_size = 10)

loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Binarize the output
y_test_bin = label_binarize(y_test.argmax(axis=1), classes=[0, 1, 2])
y_pred_proba = model.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Initialize lists to store training and validation loss and accuracy
train_loss = []
val_loss = []
train_acc = []
val_acc = []

# Fit the model and record the metrics for each epoch
for epoch in range(30):
  history = model.fit(X_train, y_train, epochs=1, batch_size=10, validation_data=(X_test, y_test), verbose=0)
  train_loss.append(history.history['loss'][0])
  val_loss.append(history.history['val_loss'][0])
  train_acc.append(history.history['accuracy'][0])
  val_acc.append(history.history['val_accuracy'][0])

# Plot the learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=lb_encoder.classes_, yticklabels=lb_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true_classes, y_pred_classes, target_names=lb_encoder.classes_))

