import torch
import time
import json

def torch_state(path):
    for i in range(10):
        try:
            #state = torch.load(path)
            state = torch.load(path, map_location=lambda storage, loc: storage)
            return state
        except Exception as e:
            print("Failed to load state",i,path, e)
            time.sleep(i)
            pass

    print("Failed to load state")
    return

def json_state(path):
    for i in range(10):
        try:
            with open(path) as f:
                state = json.load(f)
            return state
        except Exception as e:
            print("Failed to load json",i,path, e)
            time.sleep(i)
            pass

    print("Failed to load state")
    return None
