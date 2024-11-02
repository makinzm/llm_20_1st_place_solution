import os
import sys


# for running on kaggle
# https://www.kaggle.com/discussions/product-feedback/471939#2692856
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, "lib"))
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, "utils"))
    sys.path.insert(0, KAGGLE_AGENT_PATH)
else:
    sys.path.insert(0, "/kaggle/working/submission/lib")

# add current directory to path
# sys.path.insert(0, os.path.dirname(__file__))

try:
    from .utils.switchboard import switchboard
except:
    from utils.switchboard import switchboard


def agent_fn(obs, cfg):
    """The main hook, keep it named 'agent_fn'."""
    # return "no"
    print("#" * 120)
    print(f"turnType {obs.turnType}")
    try:
        response = switchboard(obs, cfg)
    except Exception as e:
        import traceback

        print(f"Error: {e}")

        traceback.print_exc()
        response = "no"
    print(f"response: {response}")
    print("#" * 120)

    if obs["turnType"] == "answer":
        if response not in ["yes", "no"]:
            response = "no"
    else:
        if response is None:
            response = "Error in response."

    return response