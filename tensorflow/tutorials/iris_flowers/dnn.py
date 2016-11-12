import tensorflow as tf
import numpy as np


# Data sets path
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load data sets
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,
                                                                   features_dtype=np.float, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST, features_dtype=np.float, target_dtype=np.int)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build a 3 layers DNN with 10, 20, 10 nuerons respectively, 3 target classes
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3, model_dir="/tmp/iris_model")

# Fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# Evaluate model
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Prediction
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))