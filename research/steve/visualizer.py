from __future__ import print_function
from builtins import range
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

import numpy as np
import tensorflow as tf
# import moviepy.editor as mpy
import time, os, traceback, multiprocessing, portalocker, sys

import envwrap
import util
import valuerl
from worldmodel import DeterministicWorldModel
from config import config

base_path = "%s/%s/%s/seed_%d" % (config["output_root"], config["name"], config["env"]["name"], config["seed"])
LOG_PATH = util.create_directory("%s/%s" % (base_path, config["log_path"])) + "/%s" % config["name"]
LOAD_PATH = util.create_directory("%s/%s" % (base_path,config["save_model_path"]))
print(LOG_PATH,LOAD_PATH)
OBS_DIM = np.prod(config["env"]["obs_dims"])
MODEL_AUGMENTED = config["model_config"] is not False
if MODEL_AUGMENTED: MODEL_BAYESIAN_CONFIG = config["model_config"]["bayesian"]
ROLLOUT_LEN = config["policy_config"]["value_expansion"]["rollout_len"]
MODEL_ENSEMBLING = config["name"] == "steve"
MULTIPROCESSING = False

FILENAME = LOAD_PATH

if __name__ == '__main__':
    oprl = valuerl.ValueRL(config["name"], config["env"], config["policy_config"], config["seed"],multiprocessing=multiprocessing)

    obs_loader = tf.compat.v1.placeholder(tf.float32, [1, OBS_DIM])
    policy_actions = oprl.build_evalution_graph(obs_loader, mode="exploit")

    if MODEL_AUGMENTED:
        next_obs_loader = tf.compat.v1.placeholder(tf.float32, [1, OBS_DIM])
        reward_loader = tf.compat.v1.placeholder(tf.float32, [1])
        done_loader = tf.compat.v1.placeholder(tf.float32, [1])
        worldmodel = DeterministicWorldModel(config["name"], config["env"], config["model_config"], config["seed"],config["original_config"],multiprocessing=multiprocessing)
        _, confidence, _ , _ = oprl.build_Q_expansion_graph(next_obs_loader, reward_loader, done_loader, worldmodel, rollout_len=ROLLOUT_LEN, model_ensembling=MODEL_ENSEMBLING)
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    oprl.load(sess, FILENAME)
    if MODEL_AUGMENTED: worldmodel.load(sess, FILENAME)

    env = envwrap.get_env(config["env"]['name'])

    hist = np.zeros([4, 10])
    for _ in range(10):
        ts = 0
        rgb_frames = []
        obs, reward, done, reset = env.reset(), 0, False, False
        while not reset:
            env.internal_env.render()
            # rgb_frames.append(env.internal_env.render(mode='rgb_array'))
            # action = env.action_space.sample()
            all_actions = sess.run(policy_actions, feed_dict={obs_loader: np.array([obs])})
            all_actions = np.clip(all_actions, -1., 1.)
            action = all_actions[0]
            obs, _reward, done, reset = env.step(action)

            if MODEL_AUGMENTED:
                _confidences = sess.run(confidence, feed_dict={next_obs_loader: np.expand_dims(obs,0),
                                                               reward_loader: np.expand_dims(_reward,0),
                                                               done_loader: np.expand_dims(done,0)})
                # print "%.02f %.02f %.02f %.02f" % tuple(_confidences[0,0])
                for h in range(4):
                    bucket = int((_confidences[0,0,h]-1e-5)*10)
                    hist[h,bucket] += 1

            reward += _reward
            ts += 1
            # print ts, _reward, reward
        print(ts, reward)
    hist /= np.sum(hist, axis=1, keepdims=True)
    for row in reversed(hist.T): print(' '.join(["%.02f"] * 4) % tuple(row))

    #clip = mpy.ImageSequenceClip(rgb_frames, fps=100)
    #clip.write_videofile(FILENAME + "/movie.mp4")
