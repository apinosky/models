from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
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

import traceback, threading, time, warnings
import tensorflow as tf
import numpy as np

import util
from replay import ReplayBuffer
import gc
from termcolor import cprint

class Learner(object):
    """
    Generic object which runs the main training loop of anything that trains using
    a replay buffer. Handles updating, logging, saving/loading, batching, etc.
    """
    def __init__(self, interactor_queue, lock, config, env_config, learner_config, **bonus_kwargs):
        self.seed = config["seed"]
        tf.compat.v1.set_random_seed(config["seed"])

        self.learner_name = self.learner_name()
        if lock is None:
            self.multiprocessing = False
        else:
            self.multiprocessing = True
            self.interactor_queue = interactor_queue
            self.learner_lock = lock
            self.kill_threads = False
            self.permit_desync = False
            self.need_frames_notification = threading.Condition()
        self.config = config
        self.env_config = env_config
        self.learner_config = learner_config
        self.bonus_kwargs = bonus_kwargs
        self.original_config = config["original_config"]
        self._reset_inspections()
        self.total_frames = 0

        # params to test different learning configs
        if self.multiprocessing or self.original_config:
            self.decay = False
            self.update_model_every_iter = False
        else: # alp manual mods for testing
            self.decay = False
            self.update_model_every_iter = False
        self.learn_done_fn = not(self.config["no_done"])

        base_path = "%s/%s/%s/seed_%d" % (self.config["output_root"], self.config["name"], self.config["env"]["name"], self.config["seed"])
        self.save_path = util.create_directory("%s/%s" % (base_path, self.config["save_model_path"]))
        self.log_path = util.create_directory("%s/%s" % (base_path, self.config["log_path"])) + "/%s.log" % self.learner_name

        if self.multiprocessing:
            # replay buffer to store data
            self.replay_buffer_lock = threading.RLock()
            self.replay_buffer = ReplayBuffer(self.learner_config["replay_size"],
                                              np.prod(self.env_config["obs_dims"]),
                                              self.env_config["action_dim"],
                                              seed=config["seed"])
        else:
            self.replay_buffer = bonus_kwargs["replay_buffer"]

        # data loaders pull data from the replay buffer and put it into the tfqueue for model usage
        self.data_loaders = self.make_loader_placeholders()
        queue_capacity = np.ceil(1./self.learner_config["frames_per_update"]) if self.learner_config["frames_per_update"] else 100
        self.tf_queue = tf.queue.FIFOQueue(capacity=queue_capacity, dtypes=[dl.dtype for dl in self.data_loaders])
        self.enqueue_op = self.tf_queue.enqueue(self.data_loaders)
        self.current_batch = self.tf_queue.dequeue()
        self.cycles_since_last_update = 0

        # build the TF graph for the actual model to train
        self.core, self.train_losses, self.train_ops, self.inspect_losses = self.make_core_model()
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25, allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        print('learn_done, decay, update_model_every_iter',self.learn_done_fn,self.decay,self.update_model_every_iter)

    ## Mandatory functions to override
    def learner_name(self): raise Exception('unimplemented: learner_name')
    def make_loader_placeholders(self): raise Exception('unimplemented: make_loader_placeholders')
    def make_core_model(self): raise Exception('unimplemented: make_core_model')

    ## Optional functions to override
    def initialize(self): warnings.warn('unimplemented: initialize')
    def resume_from_checkpoint(self, epoch): warnings.warn('unimplemented: resume_from_checkpoint')
    def checkpoint(self): warnings.warn('unimplemented: checkpoint')
    def backup(self): warnings.warn('unimplemented: backup')

    ## Internal functions
    def _start(self):
        # fetch data from the interactors to pre-fill the replay buffer
        self.prefetch_thread = threading.Thread(target=self._poll_interactors, args=(True, self.learner_config["frames_before_learning"],))
        self.prefetch_thread.start()
        self.prefetch_thread.join()

        # start the interactor and data loader
        self.data_load_thread = threading.Thread(target=self._run_enqueue_data)
        self.data_load_thread.start()

        # initialize the learner, pretraining if needed
        if self.config["resume"]: self._resume_from_checkpoint()
        else:                     self._initialize()

        # re-sync everything, and start up interactions with the environment
        self.interactor_poll_thread = threading.Thread(target=self._poll_interactors)
        self.interactor_poll_thread.start()

        # start the clock
        self._last_checkpoint_time = time.time()

    def _learn(self, permit_desync=False, log=True, checkpoint=True, backup=True):
        # this is to keep the frames/update synced properly
        if self.learner_config["frames_per_update"] is not False and not permit_desync:
            if not self.multiprocessing: self.total_frames = self.replay_buffer.count
            if not self._have_enough_frames():
                if self.multiprocessing:
                    with self.need_frames_notification:
                        self.need_frames_notification.notify()
                return

        # this is to run the frames-per-update iterations when not multiprocessing
        if self.multiprocessing or permit_desync: iters = 1
        else: iters = int(np.ceil(1/self.learner_config["frames_per_update"]))

        if self.learner_config["frames_per_update"] > 1:
            if self.cycles_since_last_update < self.learner_config["frames_per_update"]-1:
                self.cycles_since_last_update += 1
                return
            else:
                self.cycles_since_last_update = 0

        for _ in range(iters):
            # load the data buffer
            if not self.multiprocessing:
                data = self.replay_buffer.random_batch(self.learner_config["batch_size"])
                self.sess.run(self.enqueue_op, feed_dict=dict(list(zip(self.data_loaders, data))))

            # log
            if log and (self.update_i + 1) % self.learner_config["log_every_n"] == 0:
                self._log() # just writes status to log files and prints to screen

            # checkpoint
            if checkpoint and (self.update_i + 1) % self.learner_config["epoch_every_n"] == 0:
                self._checkpoint() # copies Q to old Q and loads / saves model (if applicable)

            # backup
            if backup and (self.update_i + 1) % self.learner_config["backup_every_n"] == 0:
                self._backup() # saves replay buffer

            if self.decay:
                self.checkpoint(decay=True) # copy params to old params

            # train
            self._training_step()

        if self.update_model_every_iter and not(permit_desync) and self.learner_name == "worldmodel":
            self.core.save(self.sess, self.save_path) # ,minimal=True)

    def _have_enough_frames(self):
        gathered_frames = self.total_frames - self.learner_config["frames_before_learning"]
        return gathered_frames > self.learner_config["frames_per_update"] * self.update_i

    def _initialize(self):
        cprint('initializing ' + self.learner_name,'magenta')
        self.epoch = 0
        self.update_i = 0
        self.hours = 0
        self._last_checkpoint_time = time.time()

        self.initialize(self.decay)
        if self.multiprocessing and self.learner_config["pretrain_n"]: self._pretrain()
        if self.multiprocessing: self._checkpoint()

    def _pretrain(self):
        cprint('starting ' + self.learner_name + ' pretrain','green')
        for _ in range(self.learner_config["pretrain_n"]):
            self._learn(permit_desync=True, checkpoint=False, backup=False)
        self.epoch = 0
        self.update_i = 0
        cprint('finished ' + self.learner_name + ' pretrain','green')

    def _resume_from_checkpoint(self):
        epoch = util.get_largest_epoch_in_dir(self.save_path, self.core.saveid)
        if not self.config['keep_all_replay_buffers']: util.wipe_all_but_largest_epoch_in_dir(self.save_path, self.core.saveid)
        if epoch is False:
            raise Exception("Tried to reload but no model found")
        with self.learner_lock:
            self.core.load(self.sess, self.save_path, epoch)
            self.epoch, self.update_i, self.total_frames, self.hours = self.sess.run([self.core.epoch_n, self.core.update_n, self.core.frame_n, self.core.hours])
        with self.replay_buffer_lock:
            self.replay_buffer.load(self.save_path, '%09d_%s' % (epoch, self.learner_name))
        self.resume_from_checkpoint(epoch)

    def _log(self):
        if self.denom > 0:
            logstring = "(%3.2f sec) h%-8.2f e%-8d s%-8d f%-8d\t" % (time.time() - self._log_time, self.hours, self.epoch, self.update_i + 1, self.total_frames) + ', '.join(["%8f" % x for x in (self.running_total / self.denom).tolist()])
            print("%s\t%s" % (self.learner_name, logstring))
            with open(self.log_path, "a") as f: f.write(logstring + "\n")
        self._reset_inspections()

    def _reset_inspections(self):
        self.running_total = 0.
        self.denom = 0.
        self._log_time = time.time()

    def _checkpoint(self):
        if not self.decay:
            self.checkpoint()
        self.epoch += 1
        self.hours += (time.time() - self._last_checkpoint_time) / 3600.
        self._last_checkpoint_time = time.time()
        self.core.update_epoch(self.sess, self.epoch, self.update_i, self.total_frames, self.hours)
        if self.multiprocessing:
            with self.learner_lock: self.core.save(self.sess, self.save_path)
        else:
            if not(self.update_model_every_iter) and self.learner_name == "worldmodel":
                self.core.save(self.sess, self.save_path)

    def _backup(self):
        self.backup() # not implemented (doesn't actually do anything)
        if self.multiprocessing: # save replay buffer and learner
            if not self.learner_config['keep_all_replay_buffers']: util.wipe_all_but_largest_epoch_in_dir(self.save_path, self.core.saveid)
            with self.learner_lock:
                self.core.save(self.sess, self.save_path, self.epoch)
            with self.replay_buffer_lock:
                self.replay_buffer.save(self.save_path, '%09d_%s' % (self.epoch, self.learner_name))

    def _training_step(self):
        train_ops = tuple([op for op, loss in zip(self.train_ops,
                                                  self.train_losses)
                           if loss is not None])
        outs = self.sess.run(train_ops + self.inspect_losses)
        self.running_total += np.array(outs[len(train_ops):])
        self.denom += 1.
        self.update_i += 1

    def _poll_interactors(self, continuous_poll=False, frames_before_terminate=None):
        # poll the interactors for new frames.
        # the synced_condition semaphore prevents this from consuming too much CPU
        while not self.kill_threads:
            if self.learner_config["frames_per_update"] is not False and not continuous_poll:
                with self.need_frames_notification: self.need_frames_notification.wait()
            while not self.interactor_queue.empty():
                new_frames = self.interactor_queue.get()
                self._add_frames(new_frames)
                if frames_before_terminate and self.total_frames >= frames_before_terminate: return

    def _add_frames(self, frames):
        with self.replay_buffer_lock:
            for frame in frames:
                self.replay_buffer.add_replay(*frame)
            self.total_frames = self.replay_buffer.count
        return self.total_frames

    def _run_enqueue_data(self):
        while not self.kill_threads:
            try:
                data = self.replay_buffer.random_batch(self.learner_config["batch_size"])
                self.sess.run(self.enqueue_op, feed_dict=dict(list(zip(self.data_loaders, data))))
            except:
                self.kill_threads

    def _kill_threads(self):
        self.kill_threads = True

    def close(self):
      self._log() # save final log
      self.core.save(self.sess, self.save_path) # save final model
      print('saved learner')

      self.sess.close()
      self.tf_queue.close()
      tf.compat.v1.reset_default_graph()
      tf.contrib.keras.backend.clear_session()
      gc.collect()
      try:
          self.bonus_kwargs['stop_queue'].put(True)
          print('stopping master.py')
      except:
          pass


class CoreModel(object):
    """The base class for the "core" of learners."""
    def __init__(self, name, env_config, learner_config, seed, original_config, learn_done=True, multiprocessing=True):
        self.name = self.saveid + "/" + name
        self.env_config = env_config
        self.learner_config = learner_config
        # new configuration variables
        self.seed = seed
        self.original_config = original_config
        self.learn_done_fn = learn_done
        self.multiprocessing = multiprocessing
        self.set_seed(seed)

        with tf.compat.v1.variable_scope(self.name):
            self.epoch_n = tf.compat.v1.get_variable('epoch_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.update_n = tf.compat.v1.get_variable('update_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.frame_n = tf.compat.v1.get_variable('frame_n', [], initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            self.hours = tf.compat.v1.get_variable('hours', [], initializer=tf.constant_initializer(0.), dtype=tf.float64, trainable=False)
            self.epoch_n_placeholder = tf.compat.v1.placeholder(tf.int64, [])
            self.update_n_placeholder = tf.compat.v1.placeholder(tf.int64, [])
            self.frame_n_placeholder = tf.compat.v1.placeholder(tf.int64, [])
            self.hours_placeholder = tf.compat.v1.placeholder(tf.float64, [])
        self.assign_epoch_op = [tf.compat.v1.assign(self.epoch_n, self.epoch_n_placeholder), tf.compat.v1.assign(self.update_n, self.update_n_placeholder), tf.compat.v1.assign(self.frame_n, self.frame_n_placeholder), tf.compat.v1.assign(self.hours, self.hours_placeholder)]

        self.create_params(env_config, learner_config)
        self.model_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.saver = tf.compat.v1.train.Saver(self.model_params)
        self.minimal_saver = tf.compat.v1.train.Saver(self.policy_params)

    @property
    def saveid(self):
        raise Exception("specify a save ID")

    def create_params(self, env_config, learner_config):
        raise Exception("unimplemented")

    def update_epoch(self, sess, epoch, updates, frames, hours):
        sess.run(self.assign_epoch_op, feed_dict={self.epoch_n_placeholder: int(epoch), self.update_n_placeholder: int(updates), self.frame_n_placeholder: int(frames), self.hours_placeholder: float(hours)})

    def save(self, sess, path, epoch=None, minimal=False):
        if minimal:
            self.minimal_saver.save(sess, path + "/%s.params" % self.saveid)
        else:
            if epoch is None:  self.saver.save(sess, path + "/%s.params" % self.saveid)
            else:              self.saver.save(sess, path + "/%09d_%s.params" % (epoch, self.saveid))

    def load(self, sess, path, epoch=None, minimal=False):
        if minimal:
            self.minimal_saver.restore(sess, path + "/%s.params" % self.saveid)
        else:
            if epoch is None:  self.saver.restore(sess, path + "/%s.params" % self.saveid)
            else:              self.saver.restore(sess, path + "/%09d_%s.params" % (epoch, self.saveid))

    def set_seed(self,seed):
        tf.compat.v1.set_random_seed(seed)


# for multiprocessing
def run_learner(learner_subclass, queue, lock, config, env_config, learner_config, max_frames, **bonus_kwargs):
    learner = learner_subclass(queue, lock, config, env_config, learner_config, **bonus_kwargs)

    # added to save when exiting
    from signal import signal, SIGINT
    from sys import exit
    def close_learner():
        # Handle any cleanup here
        learner._kill_threads()
        # then save stuff
        print('learner max frames reached')
        learner.close()
    def handler(signal_received, frame):
        close_learner()
        exit(0)
    signal(SIGINT, handler) # Tell Python to run the handler() function when SIGINT is recieved

    try:
        learner._start()
        # while True: learner._learn()
        while learner.total_frames < max_frames : learner._learn() # run for fixed time horizon
        close_learner()

    except Exception as e:
        print('Caught exception in learner process')
        traceback.print_exc()
        learner._kill_threads()
        print()
        raise e
