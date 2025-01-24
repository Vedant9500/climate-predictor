import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize TPU if available
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except ValueError:
    strategy = tf.distribute.get_strategy()
    print("Running on CPU/GPU")

# Load the combined data
data = pd.read_csv('/workspaces/climate-predictor/combined_hourly_data.csv')

# Handle missing values by filling them with the mean of the column
data.fillna(data.mean(), inplace=True)

# Encode the 'city' column using one-hot encoding
data = pd.get_dummies(data, columns=['city'])

# Separate features and target variable
X = data.drop(columns=['temperature_2m'])
y = data['temperature_2m']

# Normalize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to TensorFlow format
def convert_to_tfrecord(X, y, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(y)):
            feature = {
                'features': tf.train.Feature(float_list=tf.train.FloatList(value=X[i])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=[y[i]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

# Save in TFRecord format for TPU
convert_to_tfrecord(X_train, y_train, '/workspaces/climate-predictor/train.tfrecord')
convert_to_tfrecord(X_test, y_test, '/workspaces/climate-predictor/test.tfrecord')

# Also save the scaler for later use
import joblib
joblib.dump(scaler, '/workspaces/climate-predictor/scaler.save')

print("Preprocessed data saved in TFRecord format for TPU training")
