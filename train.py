from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten

import tensorflow as tf
import os
import pandas as pd
import numpy as np

dirname = os.path.dirname(__file__)
bal_train_folder = os.path.join(dirname, 'DataSet/train')
eval_folder = os.path.join(dirname, 'DataSet/val')
mapping_csv = os.path.join(dirname, 'DataSet/class_labels_indices.csv')

audioset_label_count = 528

labels = pd.read_csv(mapping_csv)['mid'].tolist()


def get_record(folder, file_names):
  files_glob = []
  for name in file_names:
    files_glob.append("{}/{}".format(folder, name))

  return files_glob


def get_bal_record(file_names):
  return get_record(bal_train_folder, file_names)


def get_eval_record(file_names):
  return get_record(eval_folder, file_names)


def fetch_model():
  new_model = Sequential()
  new_model.add(BatchNormalization(input_shape=(10, 128)))  # The input shape excludes batch
  new_model.add(Flatten())
  new_model.add(Dense(2048, activation="relu"))
  new_model.add(Dense(audioset_label_count, activation="sigmoid"))
  new_model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
  return new_model


def parser(record, training=True, total_label_count=528):
  context_features = {
    "start_time_seconds": tf.FixedLenFeature([], dtype=tf.float32),
    "end_time_seconds": tf.FixedLenFeature([], dtype=tf.float32),
    "video_id": tf.FixedLenFeature([], dtype=tf.string),
  }
  sequence_features = {
    "audio_embedding": tf.FixedLenSequenceFeature([], dtype=tf.string)
  }

  # In training mode labels will be returned, otherwise they won't be
  if training:
    context_features["labels"] = tf.VarLenFeature(tf.int64)

  context_parsed, sequence_parsed = tf.parse_single_sequence_example(record, context_features, sequence_features)

  x = sequence_parsed['audio_embedding']
  if training:
    y = tf.sparse_to_dense(context_parsed["labels"].values, [total_label_count], 1)
    return x, y
  else:
    return x


def make_dataset_provider(tf_records, repeats=1000, num_parallel_calls=12, batch_size=32, total_label_count=100):
  def my_parser(record): return parser(record, total_label_count=total_label_count)

  dataset = tf.data.TFRecordDataset(tf_records)
  dataset = dataset.map(map_func=my_parser, num_parallel_calls=num_parallel_calls)
  dataset = dataset.repeat(repeats)

  dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.batch(batch_size)
  d_iter = dataset.make_one_shot_iterator()
  return d_iter


def data_generator(tf_records, batch_size=1, repeats=1000, num_parallel_calls=12, total_label_count=528):
  """
  :return: Data in shape (batch_size, n_frames=10, 128 features - 1 byte each)
  """
  tf_provider = make_dataset_provider(tf_records, repeats=repeats, num_parallel_calls=num_parallel_calls, batch_size=batch_size,
                                      total_label_count=total_label_count)
  sess = tf.Session()

  next_el = tf_provider.get_next()
  max_frames = 10
  while True:
    try:
      raw_x, y = sess.run(next_el)  # returns (batch_size, n_frames, 128)
      x = []
      for entry in raw_x:
        n_frames = entry.shape[0]  # Entry has a shape (n_frames, )
        audio_frame = []
        for i_frame in range(n_frames):
          frame = np.frombuffer(entry[i_frame], np.uint8).astype(np.float32)
          # print("trigger appending a frame of size {}".format(len(float_frames)))
          audio_frame.append(frame)

        if n_frames < max_frames:
          pad = [np.zeros([128], np.float32) for i in range(max_frames-n_frames)]
          audio_frame += pad

        x.append(audio_frame)

      print("audio_frame.shape=({}, {}, {}). y.shape()={}".format(len(x), len(x[0]), len(x[0][0]), len(y)))
      # print("Input shape")
      for i in range(len(x)):
        if len(x[i]) != 10:
          print("ERROR-1")
        for j in range(len(x[i])):
          if len(x[i][j]) != 128:
            print("ERROR-2")

      yield np.array(x), np.array(y)
    except tf.errors.OutOfRangeError:
      print("Iterations exhausted")
      break


train_data = []
eval_data = []
batch_size = 3

for file in os.listdir(bal_train_folder):
  if file.endswith(".tfrecord"):
    train_data.append(os.path.join(bal_train_folder, file))

for file in os.listdir(eval_folder):
  if file.endswith(".tfrecord"):
    eval_data.append(os.path.join(eval_folder, file))


train_generator = data_generator(train_data, batch_size=batch_size, num_parallel_calls=1)
validation_generator = data_generator(eval_data, batch_size=batch_size, num_parallel_calls=1)
model = fetch_model()
model.summary()

model.fit_generator(train_generator,
                    steps_per_epoch=len(train_data),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=20)