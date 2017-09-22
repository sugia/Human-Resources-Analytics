import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
from tensorflow.python.framework import ops 
import csv
import os

# normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
  col_max = m.max(axis=0)
  col_min = m.min(axis=0)
  return 1.0 * (m - col_min) / (col_max - col_min)

# define variable functions (weights and bias)
def init_weight(shape, st_dev):
  weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
  return(weight)

def init_bias(shape, st_dev):
  bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
  return(bias)

# create a fully connected layer:
def fully_connected(input_layer, weights, biases):
  layer = tf.matmul(input_layer, weights) + biases
  return(tf.nn.relu(layer))

if __name__ == '__main__':
  data_file = 'data/HR_comma_sep.csv'
  raw = []
  sales_left = {}
  salary_left = {}
  with open(data_file, 'r') as infile:
    reader = csv.reader(infile)
    reader.next()
    for row in reader:
      if row[8] not in sales_left:
        sales_left[row[8]] = [int(row[6])]
      else:
        sales_left[row[8]].append(int(row[6]))

      if row[9] not in salary_left:
        salary_left[row[9]] = [int(row[6])]
      else:
        salary_left[row[9]].append(int(row[6]))

      raw.append(row)

  # for sales_left
  vec = []
  for key in sales_left:
    vec.append((key, len(sales_left[key]) - sum(sales_left[key])))

  vec.sort(key = lambda x : x[1])

  # row[8]
  sales_rank = {}
  rank_sales = {}
  for i in range(len(vec)):
    sales_rank[vec[i][0]] = i 
    rank_sales[i] = vec[i][0]

  # transform row[8] in raw data
  for i in range(len(raw)):
    raw[i][8] = sales_rank[raw[i][8]]

  # for salary_left
  vec = []
  for key in salary_left:
    vec.append((key, len(salary_left[key]) - sum(salary_left[key])))

  vec.sort(key = lambda x : x[1])

  # row[9]
  salary_rank = {}
  rank_salary = {}
  for i in range(len(vec)):
    salary_rank[vec[i][0]] = i 
    rank_salary[i] = vec[i][0]

  # transform row[9] in raw data
  for i in range(len(raw)):
    raw[i][9] = salary_rank[raw[i][9]]


  x_vals = []
  y_vals = []
  for line in raw:
    x_vals.append(line[:6] + line[7:])
    y_vals.append(int(line[6]))

  # transform str in x_vals into float
  for i in range(len(x_vals)):
    for j in range(len(x_vals[i])):
      x_vals[i][j] = float(x_vals[i][j])

  x_vals = np.array(x_vals)
  y_vals = np.array(y_vals)
  

  ops.reset_default_graph()
  sess = tf.Session()

  batch_size = 100

  train_indices = np.random.choice(len(x_vals), int(len(x_vals) * 0.8), replace = False)
  test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
  x_vals_train = np.nan_to_num(normalize_cols(x_vals[train_indices]))
  # x_vals_train = x_vals[train_indices]
  y_vals_train = y_vals[train_indices]
  x_vals_test = np.nan_to_num(normalize_cols(x_vals[test_indices]))
  # x_vals_test = x_vals[test_indices]
  y_vals_test = y_vals[test_indices]

  num_a = x_vals.shape[1]
  num_b = 128
  num_c = 64
  num_d = 1 # y_vals.shape[1], though it could be empty

  x_data = tf.placeholder(shape=[None, num_a], dtype=tf.float32)
  y_target = tf.placeholder(shape=[None, num_d], dtype=tf.float32)

  # create the first layer
  weight_1 = init_weight(shape=[num_a, num_b], st_dev=10.0)
  bias_1 = init_bias(shape=[num_b], st_dev=10.0)
  layer_1 = fully_connected(x_data, weight_1, bias_1)

  weight_2 = init_weight(shape=[num_b, num_c], st_dev=10.0)
  bias_2 = init_bias(shape=[num_c], st_dev=10.0)
  layer_2 = fully_connected(layer_1, weight_2, bias_2)

  weight_3 = init_weight(shape=[num_c, num_d], st_dev=10.0)
  bias_3 = init_bias(shape=[num_d], st_dev=10.0)
  final_output = fully_connected(layer_2, weight_3, bias_3)

  # add classification loss (cross entropy)
  xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y_target))

  my_opt = tf.train.AdamOptimizer(0.001)
  train_step = my_opt.minimize(xentropy)
  init = tf.global_variables_initializer()
  sess.run(init)

  prediction = tf.round(tf.sigmoid(final_output))
  predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
  accuracy = tf.reduce_mean(predictions_correct)

  loss_vec = []
  train_acc = []
  test_acc = []
  for i in range(1000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict = {x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(xentropy, feed_dict = {x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_acc.append(temp_acc_train)

    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_acc.append(temp_acc_test)
    if (i+1) % 100 == 0:
      print('Step ' + str(i+1) + ': Loss = ' + str(temp_loss))


  # Plot loss over time
  plt.plot(loss_vec, 'k-')
  plt.title('Cross Entropy Loss per Generation')
  plt.xlabel('Generation')
  plt.ylabel('Cross Entropy Loss')
  plt.show()

  # Plot train and test accuracy
  plt.plot(train_acc, 'k-', label='Train Set Accuracy')
  plt.plot(test_acc, 'r--', label='Test Set Accuracy')
  plt.title('Train and Test Accuracy')
  plt.xlabel('Generation')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.show()
