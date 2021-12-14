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
import nn
import tensorflow_probability as tfp
from termcolor import cprint

from learner import CoreModel

class DeterministicWorldModel(CoreModel):
  """
  A simple feed-forward neural network world model, with an option for an ensemble.
  """

  @property
  def saveid(self):
    return "worldmodel"

  def create_params(self, env_config, learner_config):
    self.obs_dim = np.prod(env_config["obs_dims"])
    self.action_dim = env_config["action_dim"]
    self.reward_scale = env_config["reward_scale"]
    self.discount = env_config["discount"]

    self.aux_hidden_dim = self.learner_config["aux_hidden_dim"]
    self.transition_hidden_dim = self.learner_config["transition_hidden_dim"]
    self.bayesian_config = self.learner_config["bayesian"]

    # pulled out variables to modify them
    self.modAux_inputs = False
    model_layers = self.learner_config['layers'][0]
    aux_layers = self.learner_config['layers'][1]
    if self.original_config:
      aux_inputs = self.obs_dim + self.obs_dim + self.action_dim
      ## uncomment next two lines for orig_modAux_inputs config
      # aux_inputs = self.obs_dim + self.action_dim
      # self.modAux_inputs = True
    else:
      aux_inputs = self.obs_dim + self.action_dim
      self.modAux_inputs = True

    cprint(['model layers, aux layers, aux inputs',model_layers,aux_layers,aux_inputs],'cyan')
    # make nets
    with tf.compat.v1.variable_scope(self.name):
      if self.bayesian_config:
        self.transition_predictor = nn.EnsembleFeedForwardNet('transition_predictor',self.obs_dim + self.action_dim, [self.obs_dim],
            layers=model_layers, hidden_dim=self.transition_hidden_dim, get_uncertainty=True,
            ensemble_size=self.bayesian_config["transition"]["ensemble_size"],
            train_sample_count=self.bayesian_config["transition"]["train_sample_count"],
            eval_sample_count=self.bayesian_config["transition"]["eval_sample_count"])
        if self.learn_done_fn:
            self.done_predictor = nn.EnsembleFeedForwardNet('done_predictor', aux_inputs, [],
                layers=aux_layers, hidden_dim=self.aux_hidden_dim,
                get_uncertainty=True, ensemble_size=self.bayesian_config["transition"]["ensemble_size"],
                train_sample_count=self.bayesian_config["transition"]["train_sample_count"],
                eval_sample_count=self.bayesian_config["transition"]["eval_sample_count"])
        self.reward_predictor = nn.EnsembleFeedForwardNet('reward_predictor', aux_inputs, [],
            layers=aux_layers, hidden_dim=self.aux_hidden_dim, get_uncertainty=True,
            ensemble_size=self.bayesian_config["reward"]["ensemble_size"],
            train_sample_count=self.bayesian_config["reward"]["train_sample_count"],
            eval_sample_count=self.bayesian_config["reward"]["eval_sample_count"])
      else:
        self.transition_predictor = nn.FeedForwardNet('transition_predictor',self.obs_dim + self.action_dim, [self.obs_dim],
            layers=model_layers, hidden_dim=self.transition_hidden_dim, get_uncertainty=True)
        if self.learn_done_fn:
            self.done_predictor = nn.FeedForwardNet('done_predictor', aux_inputs, [],
                layers=aux_layers, hidden_dim=self.aux_hidden_dim, get_uncertainty=True)
        self.reward_predictor = nn.FeedForwardNet('reward_predictor',aux_inputs, [],
            layers=aux_layers, hidden_dim=self.aux_hidden_dim, get_uncertainty=True)


    self.policy_params = [v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name) if [("transition_predictor" in v.name) or ("done_predictor" in v.name) or ("reward_predictor" in v.name) or ("reward_predictor" in v.name)]]

  def get_ensemble_idx_info(self):
    if self.bayesian_config is not False:
      ensemble_idxs = tf.random.shuffle(tf.range(self.transition_predictor.ensemble_size))
      transition_ensemble_sample_n = self.transition_predictor.eval_sample_count
      reward_ensemble_sample_n = self.reward_predictor.eval_sample_count
      ensemble_idxs = ensemble_idxs[:transition_ensemble_sample_n]
      return ensemble_idxs, transition_ensemble_sample_n, reward_ensemble_sample_n
    else:
      return None, 1, 1

  def build_training_graph(self, obs, next_obs, actions, next_actions, rewards, dones, data_size):
    # placeholders
    done_loss = tf.constant(0.)
    reg_loss = tf.constant(0.)

    # transition losses
    info = tf.concat([obs, actions], -1)
    predicted_next_obs = self.transition_predictor(info, is_eval=False, reduce_mode="random") + obs
    next_obs_losses = .5 * tf.reduce_sum(tf.square(next_obs - predicted_next_obs), -1)
    next_obs_loss = tf.reduce_mean(next_obs_losses)
    reg_loss += .0001 * self.transition_predictor.l2_loss()

    # reward losses
    if self.modAux_inputs: next_info = info
    else:                   next_info = tf.concat([next_obs, info], -1)
    predicted_rewards = self.reward_predictor(next_info, is_eval=False, reduce_mode="random")
    # if not(self.modAux_inputs): # self.original_config or
    reward_losses = .5 * tf.square(rewards - predicted_rewards)
    # else:
    #   _lam=0.95
    #   next_vals_info = tf.concat([next_obs, next_actions], -1)
    #   next_vals = self.reward_predictor(next_vals_info, is_eval=False, reduce_mode="random")
    #   reward_losses = .5 * tf.square(tf.stop_gradient(rewards + _lam*(1-dones)*next_vals) - predicted_rewards)
    reward_loss = tf.reduce_mean(reward_losses)
    reg_loss += .0001 * self.reward_predictor.l2_loss()

    # done losses
    if self.learn_done_fn:
      predicted_dones = self.done_predictor(next_info, is_eval=False, reduce_mode="random")
      done_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=dones, logits=predicted_dones)
      done_loss = tf.reduce_mean(done_losses)
      reg_loss += .0001 * self.done_predictor.l2_loss()

    total_loss = done_loss + reward_loss + next_obs_loss + reg_loss

    inspect = (total_loss, done_loss, reward_loss, next_obs_loss, reg_loss)

    return total_loss, inspect

  def init_extra_info(self, obs):
    return tf.zeros_like(obs)

  def transition(self, obs, action, extra_info, ensemble_idxs=None, pre_expanded=None):
    info = tf.concat([obs, action], -1)
    next_obs_delta = self.transition_predictor(info, reduce_mode="none", ensemble_idxs=ensemble_idxs, pre_expanded=pre_expanded)
    if ensemble_idxs is None:
      next_obs = tf.expand_dims(obs,-2) + next_obs_delta
      next_info = tf.concat([next_obs, tf.expand_dims(info,-2)], -1)
    else:
      next_obs = obs + next_obs_delta
      next_info = tf.concat([next_obs, info], -1)
    if self.learn_done_fn:
      if self.modAux_inputs:
        done = tf.nn.sigmoid(self.done_predictor(info, reduce_mode="none", ensemble_idxs=ensemble_idxs, pre_expanded=pre_expanded))
      else:
        done = tf.nn.sigmoid(self.done_predictor(next_info, reduce_mode="none", ensemble_idxs=ensemble_idxs, pre_expanded=True))
    else:
      done = None
    extra_info = tf.zeros_like(obs)
    return next_obs, done, extra_info

  def get_rewards(self, obs, action, next_obs):
    if self.modAux_inputs:
      next_info = tf.concat([obs, action], -1)
    else:
      next_info = tf.concat([next_obs, obs, action], -1)
    reward = self.reward_predictor(next_info, reduce_mode="none")
    return reward
