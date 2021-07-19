#!/usr/bin/python
'''It's an implementation of BERT_flow's component `flow`,
based on an embedding input text file.'''

from flow.glow_1x1 import AttrDict, Glow
from flow.glow_init_hook import GlowInitHook

import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS
import horovod.tensorflow as hvd
from optimization import AdamWeightDecayOptimizer

import os
import json
import numpy as np
import collections
import re

flags.DEFINE_float("flow_learning_rate", 4e-5, "Initial learning rate for Adam.")
flags.DEFINE_string("flow_model_config", "config_l3_d3_w32", None)
flags.DEFINE_boolean("do_train", None, "if train flow's parameter")
flags.DEFINE_string("data_dir", None, None)
flags.DEFINE_boolean("do_predict", None, None)
flags.DEFINE_string("test_file", None, None)
flags.DEFINE_string("init_checkpoint", None, None)
flags.DEFINE_integer("data_repeat_times", 1, None)
flags.DEFINE_integer("batch_size", 1024, None)
flags.DEFINE_integer("num_train_steps", 2000, None)
flags.DEFINE_integer("num_warmup_steps", 300, None)

emb_dim = 256

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps,
                     optimizer="adamw", poly_power=1.0, start_warmup_step=0):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=poly_power,
      cycle=False)

  # Implements linear warmup. I.e., if global_step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step)
                    + ", for " + str(num_warmup_steps) + " steps ++++++")
    global_steps_int = tf.cast(global_step, tf.int32)
    start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
    global_steps_int = global_steps_int - start_warm_int
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is OK that you use this optimizer for finetuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
  # As report in the Bert pulic github, the learning rate for SQuAD 1.1 finetune
  # is 3e-5, 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a
  # batch size of 64 in the finetune.
  if optimizer == "adamw":
    tf.logging.info("using adamw")
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate * hvd.size(),
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer == "lamb":
    tf.logging.info("using lamb")
    optimizer = lamb_optimizer.LAMBOptimizer(
        learning_rate=learning_rate * hvd.size(),
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    raise ValueError("Not supported optimizer: ", optimizer)

  optimizer = hvd.DistributedOptimizer(optimizer)
  #tvars = tf.trainable_variables()
  #grads = tf.gradients(loss, tvars)
  #grads = optimizer.compute_gradients(loss, tvars)
  

  # This is how the model was pre-trained.
  #(grads, tvars) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  #grads_and_vars = [(tf.clip_by_norm(grad, clip_norm=1.0), var) for (grad, var) in grads]
  #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

  #train_op = optimizer.apply_gradients(
  #    list(zip(grads, tvars)), global_step=global_step)
  train_op = optimizer.minimize(loss=loss,
    global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, neither `AdamWeightDecayOptimizer` nor `LAMBOptimizer` do this.
  # But if you use a different optimizer, you should probably take this line
  # out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def map2embedding_fn(string_line):
  def line_fn(string_line):
    embedding = np.zeros((FLAGS.batch_size, emb_dim), dtype=np.float32)
    for bid, line in enumerate(string_line):
      line = line.strip('\n')
      line_seg = line.split('\t')
      embedding[bid] = np.asarray(list(float(x) for x in line_seg[7].split(' ')))
    return embedding
  
  embedding = tf.py_func(line_fn, [string_line], tf.float32, stateful=False)
  embedding.set_shape([FLAGS.batch_size, emb_dim])
  pass
  return embedding

# read embedding from file, and makes it a tensoflow variable
def input_embedding(gpu_size=1, gpu_id=0, mode=tf.estimator.ModeKeys.PREDICT):
  try:
    file_list = os.listdir(FLAGS.data_dir)
    file_dir = FLAGS.data_dir
  except:
    file_list = [FLAGS.data_dir.split('/')[-1]]
    file_dir = '/'.join(FLAGS.data_dir.split('/')[:-1])
  files = []
  for file_id, file_name in enumerate(file_list):
    file_name = os.path.join(file_dir, file_name)
    if file_id % gpu_size == gpu_id:
      files.append(file_name)
  if mode == tf.estimator.ModeKeys.PREDICT:
    files=[FLAGS.test_file]
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    train_data_set = tf.data.TextLineDataset(files)\
                              .repeat(FLAGS.data_repeat_times)\
                              .shuffle(100*FLAGS.batch_size,reshuffle_each_iteration=True)\
                              .batch(FLAGS.batch_size)
  if mode == tf.estimator.ModeKeys.PREDICT:
    train_data_set = tf.data.TextLineDataset(files)\
                              .batch(FLAGS.batch_size, drop_remainder = False)
  train_data_set = train_data_set.map(map2embedding_fn, num_parallel_calls=4)
  iterator = train_data_set.make_one_shot_iterator()
  batch_data = iterator.get_next()
  return (batch_data, None)

def get_flow_loss(embedding, is_training):
  with open(os.path.join("./flow/config", FLAGS.flow_model_config + ".json"), 'r')as jp:
    flow_model_config = AttrDict(json.load(jp))
  flow_model_config.is_training = is_training
  flow_model = Glow(flow_model_config)
  flow_loss_example = flow_model.body(embedding, is_training)  # no l2 normalization here any more
  flow_loss_batch = tf.reduce_mean(flow_loss_example)
  embedding = tf.identity(tf.squeeze(flow_model.z, [1,2]))

  return embedding, flow_loss_example, flow_loss_batch

def create_model(features, labels, mode, params):
  input_embeddings = features

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  with tf.variable_scope("flow") as scope:
    embeddings, flow_loss_example, flow_loss_batch = get_flow_loss(input_embeddings, True)

  tvars = tf.trainable_variables()
  initialized_variable_names = {}
  if FLAGS.init_checkpoint:
    (assignment_map, initialized_variable_names
    ) = get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
    tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
  
  tf.logging.info("****** Trainable Variables ******")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)

  if is_training:
    return embeddings, flow_loss_batch 
  if mode == tf.estimator.ModeKeys.PREDICT:
    output_spec = tf.estimator.EstimatorSpec(
      mode = mode,
      predictions = {"embeddings": embeddings})
    return output_spec
  

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  hvd.init()
  
  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(hvd.local_rank())
  config.gpu_options.allow_growth = True
  if hvd.rank() != 0:
    model_dir = None
  else:
    model_dir = './results'

  run_config = tf.estimator.RunConfig(
    session_config=config,
    model_dir=model_dir)
  estimator = tf.estimator.Estimator(
    model_fn=create_model,
    config=run_config)

  if FLAGS.do_predict:
    tf.logging.info("****** Predict using flow ******")
    results = estimator.predict(input_fn = input_embedding)

    output_predict_file = os.path.join(model_dir, FLAGS.test_file.split('/')[-1].replace(".emb.dat", ".post.emb.dat"))
    with open(FLAGS.test_file) as f:
      predict_examples = f.readlines()

    with tf.gfile.GFile(output_predict_file, "w") as pred_writer:
      num_written_lines = 0
      tf.logging.info("****** Predicting ******")
      for (i, (example, prediction)) in enumerate(zip(predict_examples, results)):
        ss = prediction["embeddings"]
        pred_writer.write('\t'.join(example.strip().split('\t')[:-1]) + '\t' + ' '.join(str(x) for x in ss) + '\n')
        num_written_lines += 1

  if FLAGS.do_train:
    tf.logging.info("****** Training Flow ******")
    embeddings, _ = input_embedding(hvd.size(), hvd.rank(), tf.estimator.ModeKeys.TRAIN)
    _, flow_loss = create_model(embeddings, None, tf.estimator.ModeKeys.TRAIN, None)

    learning_rate = FLAGS.flow_learning_rate
    global_step = tf.train.get_or_create_global_step()

    train_op = create_optimizer(flow_loss, learning_rate, FLAGS.num_train_steps, FLAGS.num_warmup_steps)

    hooks=[hvd.BroadcastGlobalVariablesHook(0), tf.train.StopAtStepHook(last_step=FLAGS.num_train_steps // hvd.size())]

    with tf.train.MonitoredTrainingSession(checkpoint_dir=model_dir, hooks=hooks, config=config,
                      save_checkpoint_steps=300, save_summaries_steps=30) as ms:
      while not ms.should_stop():
        _, mgs = ms.run([train_op, global_step])
  
  return
        
if __name__ == '__main__':
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("do_train")
  tf.app.run()
