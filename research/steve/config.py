from __future__ import print_function
from builtins import str
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

import argparse, json, util, traceback

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("root_gpu", type=int)
parser.add_argument("seed", type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--render", action="store_true")
parser.add_argument("--orig", action="store_true")
parser.add_argument("--no_done", action="store_true")
parser.add_argument('--overwrite_root', type=str, const=None, help='modify root path from command line')
args = parser.parse_args()
print(args)

config_loc = args.config
config = util.ConfigDict(config_loc)

config["name"] = config_loc.split("/")[-1][:-5]
config["resume"] = args.resume
config["seed"] = args.seed
config["original_config"] = args.orig
config["no_done"] = args.no_done

if args.overwrite_root is not None:
    print('overriding output_root',config["output_root"],'with',args.overwrite_root)
    config["output_root"] = args.overwrite_root
    print('new root',args.overwrite_root)

cstr = str(config)

def log_config():
  base_path = "%s/%s/%s/seed_%d" % (config["output_root"], config["name"], config["env"]["name"], config["seed"])
  if not config["resume"]:
    checkpoints = base_path + "/%s" % (config["save_model_path"])
    util.create_and_wipe_directory(checkpoints)
  log_path = base_path + "/%s" % (config["log_path"])
  HPS_PATH = util.create_directory(log_path) + "/hps.json"

  print("ROOT GPU: " + str(args.root_gpu) + "\n" + str(cstr))
  with open(HPS_PATH, "w") as f:
    f.write("ROOT GPU: " + str(args.root_gpu) + "\n" + str(cstr))
  return args
