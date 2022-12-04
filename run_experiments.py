from ast import arg
import json
import os
from pathlib import  Path
#from test_class import  test
import re
import sys
# append src folder to import path
from src.train_cfgs import Train_cfgs
import argparse
import wandb


def get_config(json_file,json_dir):
    """Read default config and update files based on config file

    Args:
        json_path (dir): directory of jsons
    """
    #print("Loading Json file", json_file)
    with open(str(json_dir)+"/"+json_file, 'r') as j:
        contents = json.loads(j.read())

    default_config_path = Path(__file__).parent/"cfgs/default_cfg.json"
    with open(default_config_path, 'r') as t:
        default_dict = json.loads(t.read())

    default_dict.update(contents)
    print(f"{json_file}loaded")
    #print("Updated default directory with values", contents, "resulting in new dict", default_dict)
    return default_dict




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Loop')
    # cfg dir
    parser.add_argument('--cfg_dir', default=Path(__file__).parent/"cfgs", type=str, help='cfg dir')
    # use augmented data
    parser.add_argument('--aug', default=False, type=bool, help='use augmented data')
    args = parser.parse_args()

    json_dir = args.cfg_dir
    json_files = [pos_json for pos_json in os.listdir(json_dir) if pos_json.endswith('.json') and not pos_json.startswith('default')]
    #We want to make sure that the configs are processed in their numerical order
    JSON_FILE_PREFIX = "config_"
    json_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    

    #Enter all numbers which should be excluded:
    skip_experiments = [1,2]
    # read the api key from the json file
    with open('wandbkey.json') as f:
        data = json.load(f)
    wandb.login(key=data['Wandb_key'])
    for i in json_files:
        if i.startswith(JSON_FILE_PREFIX):
            experiment_key = int(os.path.splitext(i)[0][len(JSON_FILE_PREFIX):])
        if experiment_key in skip_experiments:
            continue
        
        try:
            # on purpose, we dont call in train_cfgs.py
            s=get_config(i,json_dir)
            net_obj=Train_cfgs(s,i)
            net_obj.config_wb()
            net_obj.default_data_prep()
            net_obj.data_loader()
            net_obj.model_set()
            net_obj.optimizer_set()
            net_obj.loss_set()
            net_obj.train()
            net_obj.quantize()

        except Exception as e:
            print("Failed run with", e)

        


