# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
import os

from learner import Learner
from valuerl import ValueRL
from worldmodel import DeterministicWorldModel

class ValueRLLearner(Learner):
  """
  ValueRL-specific training loop details.
  """

  def learner_name(self): return "valuerl"

  def make_loader_placeholders(self):
    self.obs_loader = tf.compat.v1.placeholder(tf.float32, [self.learner_config["batch_size"], np.prod(self.env_config["obs_dims"])])
    self.next_obs_loader = tf.compat.v1.placeholder(tf.float32, [self.learner_config["batch_size"], np.prod(self.env_config["obs_dims"])])
    self.action_loader = tf.compat.v1.placeholder(tf.float32, [self.learner_config["batch_size"], self.env_config["action_dim"]])
    self.next_action_loader = tf.compat.v1.placeholder(tf.float32, [self.learner_config["batch_size"], self.env_config["action_dim"]])
    self.reward_loader = tf.compat.v1.placeholder(tf.float32, [self.learner_config["batch_size"]])
    self.done_loader = tf.compat.v1.placeholder(tf.float32, [self.learner_config["batch_size"]])
    self.datasize_loader = tf.compat.v1.placeholder(tf.float64, [])
    self.train_obs_loader = tf.compat.v1.placeholder(tf.float32, [1, np.prod(self.env_config["obs_dims"])])
    self.eval_obs_loader = tf.compat.v1.placeholder(tf.float32, [1, np.prod(self.env_config["obs_dims"])])
    return [self.obs_loader, self.next_obs_loader, self.action_loader, self.next_action_loader, self.reward_loader, self.done_loader, self.datasize_loader]

  def make_core_model(self):
    if self.config["model_config"] is not False:
        if self.multiprocessing:
            self.worldmodel = DeterministicWorldModel(self.config["name"], self.env_config, self.config["model_config"], self.config["seed"],self.original_config,self.learn_done_fn,self.multiprocessing)
        else:
            self.worldmodel = self.bonus_kwargs["model"]
    else:
        self.worldmodel = None

    valuerl = ValueRL(self.config["name"], self.env_config,  self.learner_config,self.config["seed"],self.original_config,multiprocessing=self.multiprocessing)
    (policy_loss, Q_loss), inspect_losses = valuerl.build_training_graph(*self.current_batch, worldmodel=self.worldmodel)

    policy_optimizer = tf.compat.v1.train.AdamOptimizer(self.learner_config["policy_lr"])
    policy_gvs = policy_optimizer.compute_gradients(policy_loss, var_list=valuerl.policy_params)
    capped_policy_gvs = policy_gvs
    policy_train_op = policy_optimizer.apply_gradients(capped_policy_gvs)

    Q_optimizer = tf.compat.v1.train.AdamOptimizer(self.learner_config["value_lr"])
    Q_gvs = Q_optimizer.compute_gradients(Q_loss, var_list=valuerl.Q_params)
    capped_Q_gvs = Q_gvs
    Q_train_op = Q_optimizer.apply_gradients(capped_Q_gvs)

    return valuerl, (policy_loss, Q_loss), (policy_train_op, Q_train_op), inspect_losses

  ## Optional functions to override
  def initialize(self, decay):
      if (not self.multiprocessing):
          self.policy_actions = self.core.build_evalution_graph(self.train_obs_loader, mode="explore")
          self.eval_actions = self.core.build_evalution_graph(self.eval_obs_loader, mode="exploit")
          np.random.seed(self.config["seed"])
      if self.multiprocessing and self.config["model_config"] is not False:
          while not self.load_worldmodel(): pass

  def resume_from_checkpoint(self, epoch):
      if self.config["model_config"] is not False:
          if self.multiprocessing:
              with self.bonus_kwargs["model_lock"]: self.worldmodel.load(self.sess, self.save_path, epoch)
          else:
              self.worldmodel.load(self.sess, self.save_path, epoch)

  def checkpoint(self,decay=False):
      self.core.copy_to_old(self.sess,decay=decay)  # copy Q to old Q (and policy to old policy if implemented)
      if self.config["model_config"] is not False:
          self.load_worldmodel()                    # update worldmodel

  def backup(self): pass

  # Other functions
  def load_worldmodel(self):
      if not os.path.exists("%s/%s.params.index" % (self.save_path, self.worldmodel.saveid)): return False
      if self.multiprocessing:
          with self.bonus_kwargs["model_lock"]: self.worldmodel.load(self.sess, self.save_path)
          return True
      else:
          # uncomment to print debugging messages (make sure model is loaded)
          # test = tf.Print([],[self.worldmodel.reward_predictor.weights[0]])
          # self.sess.run(test)
          self.worldmodel.load(self.sess, self.save_path) # ,minimal=True)
          # self.sess.run(test)
          return True

  # copied from agent for non-mulitprocessing eval
  def get_action(self,obs,eval=False):
      if eval:
          all_actions = self.sess.run(self.eval_actions, feed_dict={self.eval_obs_loader: np.expand_dims(obs,0)})
      else:
          if len(self.replay_buffer) < self.config["agent_config"]["full_random_n"]:
              return self.get_random_action(obs)
          all_actions = self.sess.run(self.policy_actions, feed_dict={self.train_obs_loader: np.expand_dims(obs,0)})
      all_actions = np.clip(all_actions, -1., 1.)
      return all_actions[:1]

  def get_random_action(self, *args, **kwargs):
    return np.random.random(self.env_config["action_dim"]) * 2 - 1
