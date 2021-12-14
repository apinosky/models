from builtins import str
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

import multiprocessing
import os, sys, time

from config import config, log_config
import util
import learner, agent, valuerl_learner
import psutil

AGENT_COUNT = config["agent_config"]["count"]
EVALUATOR_COUNT = config["evaluator_config"]["count"]
MODEL_AUGMENTED = config["model_config"] is not False
args = log_config()
if MODEL_AUGMENTED: import worldmodel_learner
GPU = int(args.root_gpu)
MAX_FRAMES = 100000

if __name__ == '__main__':
  all_procs = set([])
  interaction_procs = set([])

  # lock
  policy_lock = multiprocessing.Lock()
  model_lock = multiprocessing.Lock() if MODEL_AUGMENTED else None

  # queue
  stop_queue = multiprocessing.Queue(1)
  policy_replay_frame_queue = multiprocessing.Queue(1)
  model_replay_frame_queue = multiprocessing.Queue(1) if MODEL_AUGMENTED else None

  # interactors
  for interact_proc_i in range(AGENT_COUNT):
    interact_proc = multiprocessing.Process(target=agent.main, args=(interact_proc_i, False, policy_replay_frame_queue, model_replay_frame_queue, policy_lock, config, MAX_FRAMES))
    all_procs.add(interact_proc)
    interaction_procs.add(interact_proc)

  # evaluators
  for interact_proc_i in range(EVALUATOR_COUNT):
    interact_proc = multiprocessing.Process(target=agent.main, args=(interact_proc_i, True, policy_replay_frame_queue, model_replay_frame_queue, policy_lock, config, MAX_FRAMES))
    all_procs.add(interact_proc)
    interaction_procs.add(interact_proc)

  # policy training
  train_policy_proc = multiprocessing.Process(target=learner.run_learner, args=(valuerl_learner.ValueRLLearner, policy_replay_frame_queue, policy_lock, config, config["env"], config["policy_config"], MAX_FRAMES), kwargs={"model_lock": model_lock,"stop_queue":stop_queue})
  all_procs.add(train_policy_proc)

  # model training
  if MODEL_AUGMENTED:
    train_model_proc = multiprocessing.Process(target=learner.run_learner, args=(worldmodel_learner.WorldmodelLearner, model_replay_frame_queue, model_lock, config, config["env"], config["model_config"], MAX_FRAMES))
    all_procs.add(train_model_proc)

  # start all policies
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  for i, proc in enumerate(interaction_procs):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    proc.start()

  os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
  train_policy_proc.start()

  if MODEL_AUGMENTED:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1+GPU)
    train_model_proc.start()

  while True:
    if stop_queue.get():
        print('got signal to kill master')
        for proc in all_procs:
            # first kill children
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):  # or parent.children() for recursive=False
                child.kill()
                print('killed child process #',child.pid)
            # then kill process
            proc.terminate()
            proc.join()
            print('killed master pid #',proc.pid)
        break
    # try:
    #   pass
    # except:
    #   for proc in all_procs: proc.join()
