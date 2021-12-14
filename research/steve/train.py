from builtins import str, range
import multiprocessing
import os, sys, time
import tensorflow as tf
from termcolor import cprint
import numpy as np
from datetime import datetime
import pickle

# local imports
from config import config, log_config
import util
import learner, valuerl_learner
from replay import ReplayBuffer
import envwrap # agent
import gc

# test config
AGENT_COUNT = config["agent_config"]["count"]
EVALUATOR_COUNT = config["evaluator_config"]["count"]
MODEL_AUGMENTED = config["model_config"] is not False
args = log_config()
if MODEL_AUGMENTED: import worldmodel_learner
GPU = args.root_gpu
MAX_FRAMES = 100000 # 50000 #
MAX_STEPS = config["env"]["max_frames"]
RENDER = args.render
FRAME_SKIP = 1
cprint(FRAME_SKIP,'green')

# computer config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

# added to save when exiting
from signal import signal, SIGINT
from sys import exit

def end_test():
    try:
        env.close()
    except:
        pass
    print('saving final data set')
    pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
    eval_rew, eval_steps = eval()
    eval_rewards.append([frame_idx, eval_rew, ep_num, eval_steps])
    pickle.dump(eval_rewards, open(path + 'eval_reward_data' + '.pkl', 'wb'))

    end = datetime.now()
    date_str = end.strftime("%Y-%m-%d_%H-%M-%S/")

    duration = end-now
    duration_in_s = duration.total_seconds()
    days = divmod(duration_in_s, 64*60*60)[0]
    hours = divmod(duration_in_s, 60*60)[0]
    minutes, seconds = divmod(duration_in_s, 60)
    duration_str = 'DD:HH:MM:SS {:d}:{:d}:{:d}:{:d}'.format(int(days),int(hours),int(minutes % 60),int(seconds))

    # save config
    with open(path + "/../config.txt","a") as f:
        f.write('End Time\n')
        f.write('\t'+ date_str + '\n')
        f.write('Duration\n')
        f.write('\t'+ duration_str + '\n')
        f.close()

    # close everything else
    if MODEL_AUGMENTED:
        model.close()
    policy.close()

    gc.collect()

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected.')
    end_test()
    print('Exiting gracefully')
    exit(0)

# eval function
def eval():
    obs = env.reset()
    # get action
    action = policy.get_action(obs.copy(),eval=True)
    step = 0
    done = False
    episode_reward = 0
    while step < MAX_STEPS:
        for _ in range(FRAME_SKIP):
            obs, reward, done, _ = env.step(action.copy())
        action = policy.get_action(obs.copy(),eval=True)
        episode_reward += reward
        step += 1
        if done:
            break
    cprint('eval: {} {}'.format(episode_reward, step),'cyan')
    return episode_reward, step

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    # set up seeds
    np.random.seed(config["seed"])
    tf.compat.v1.set_random_seed(config["seed"])

    # load enviroment (agent)
    env = envwrap.get_env(config["env"]["name"])
    env.seed(config["seed"])

    # load model
    if MODEL_AUGMENTED:
        model_replay_buffer = ReplayBuffer(config["model_config"]["replay_size"],
                                        np.prod(config["env"]["obs_dims"]),
                                        config["env"]["action_dim"],
                                        seed=config["seed"])
        kwargs = {'replay_buffer': model_replay_buffer, 'multiprocessing': False}
        model = worldmodel_learner.WorldmodelLearner(None, None, config, config["env"], config["model_config"],**kwargs)
        worldmodel = model.core
        pretraining = (config["model_config"]["pretrain_n"] > 0)
        PRETRAIN_FRAMES = config["model_config"]["pretrain_n"]
    else:
        model_replay_buffer = []
        worldmodel = None

    # load policy
    replay_buffer = ReplayBuffer(config["policy_config"]["replay_size"],
                                    np.prod(config["env"]["obs_dims"]),
                                    config["env"]["action_dim"],
                                    seed=config["seed"])

    kwargs={'model_lock': None, 'stop_queue': None, 'replay_buffer': replay_buffer, 'model': worldmodel, 'multiprocessing': False}
    policy = valuerl_learner.ValueRLLearner(None, None, config, config["env"], config["policy_config"],**kwargs)

    # set up log
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
    dir_name = 'seed_{}/'.format(str(config["seed"]))
    path = './' + config["output_root"] + '/' + config["name"] + '/' + config["env"]["name"] + '/' + dir_name
    cprint(path,'red')
    cprint([env.action_space.shape[0],env.observation_space.shape[0]],'cyan')
    if os.path.exists(path) == False:
        os.makedirs(path)
    # save config
    with open(path + "/../config.txt","a") as f:
        f.write('\nStart Time\n')
        f.write('\t'+ date_str )
        f.write('\nConfig\n')
        f.write('\t' + str(config) + '\n')
        f.close()

    # check assertions
    if config["model_config"] is not False:
        assert config["model_config"]["batch_size"] == config["policy_config"]["batch_size"], 'batch sizes should be equal'
        assert config["model_config"]["replay_size"] == config["policy_config"]["replay_size"], 'replay_size sizes should be equal'
    batch_size = config["policy_config"]["batch_size"]

    # initialize polcies
    policy._initialize()
    PRETRAIN_FRAMES = 0
    if MODEL_AUGMENTED:
        model._initialize()
        PRETRAIN_FRAMES = config["model_config"]["pretrain_n"]
        if PRETRAIN_FRAMES > 0:
            cprint('collecting data for model pretrain','green')

    # main loop
    ready_to_eval = False
    frame_idx = 0
    ep_num = 0
    rewards = []
    eval_rewards = []
    while frame_idx < MAX_FRAMES:
        obs = env.reset()
        # get action
        action = policy.get_action(obs.copy())
        step = 0
        done = False
        episode_reward = 0
        while step < MAX_STEPS:
            for _ in range(FRAME_SKIP):
                # get next state
                next_obs, reward, done, reset = env.step(action.copy())
            # get next action
            next_action = policy.get_action(next_obs.copy())
            # fill replay buffers and learn
            if MODEL_AUGMENTED:
                model_replay_buffer.add_replay(obs, next_obs, action, next_action, reward, done)
                model._learn()
            replay_buffer.add_replay(obs, next_obs, action, next_action, reward, done)
            policy._learn()
            # update for next iter
            obs = next_obs
            action = next_action
            episode_reward += reward
            frame_idx += 1
            step += 1
            # do pretraining (if needed)
            if MODEL_AUGMENTED and (PRETRAIN_FRAMES > 0) and (frame_idx >= PRETRAIN_FRAMES):
                model._pretrain()
                PRETRAIN_FRAMES = 0

            if RENDER:
                env.render(mode="human")

            if frame_idx % (MAX_FRAMES//10) == 0:
                ready_to_eval = True
                last_reward = rewards[-1][1] if len(rewards)>0 else 0
                print(
                    'frame : {}/{}, \t last rew: {}'.format(
                        frame_idx, MAX_FRAMES, last_reward
                    )
                )
                print('saving reward log')
                pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                pickle.dump(eval_rewards, open(path + 'eval_reward_data' + '.pkl', 'wb'))
            if done:
                break
        if (len(replay_buffer) > config["policy_config"]["frames_before_learning"]):
            print('ep rew', ep_num, episode_reward, frame_idx, step)
        rewards.append([frame_idx, episode_reward,ep_num, step])
        ep_num += 1
        if ready_to_eval:
        # if (frame_idx >= config["policy_config"]["frames_before_learning"]) and (ep_num % 15 == 0):
            for _ in range(1): # 10
                eval_rew, eval_steps = eval()
                eval_rewards.append([frame_idx, eval_rew, ep_num, eval_steps])
                ready_to_eval = False
    end_test()
