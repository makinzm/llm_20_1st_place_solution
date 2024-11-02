from pathlib import Path
import os
import re
import gc
from typing import Any

import torch

QUESTION_IS_ALPHA = "Is it Agent Alpha?"
# GUESS_IS_ALPHA = "Hurray! It's Agent Alpha!"


def is_local() -> bool:
    KAGGLE_NOTEBOOK_PATH = "/kaggle/working"
    KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
    return not os.path.exists(KAGGLE_AGENT_PATH) and not os.path.exists(
        KAGGLE_NOTEBOOK_PATH
    )
    # return not is_kaggle_agent() and not is_kaggle_notebook()


def is_kaggle_agent() -> bool:
    KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
    return os.path.exists(KAGGLE_AGENT_PATH)


def is_kaggle_notebook() -> bool:
    # KAGGLE_NOTEBOOK_PATH = "/kaggle/working"
    # return os.path.exists(KAGGLE_NOTEBOOK_PATH)
    return not is_local() and not is_kaggle_agent()

ObjectHavingQuestions = Any
Question = str

def get_last_question(obs: ObjectHavingQuestions) -> Question:
    return obs.questions[-1]


def get_keyword(obs: ObjectHavingQuestions, lower=True) -> str:
    keyword = obs["keyword"]
    if lower:
        keyword = keyword.lower()
    if keyword == "":
        raise ValueError("Keyword not found in observation")
    return keyword

def is_valid_answer(ans: str) -> bool:
    return ans in ["yes", "no"]


KAGGLE_NOTEBOOK_PATH = Path("/kaggle/working/submission/")
KAGGLE_AGENT_PATH = Path("/kaggle_simulations/agent/")

ROOT_PATH = Path(__file__).parent.parent
# if is_local():
#     ROOT_PATH = Path(__file__).parent.parent
# elif is_kaggle_agent():
#     ROOT_PATH = KAGGLE_AGENT_PATH
# elif is_kaggle_notebook():
#     ROOT_PATH = KAGGLE_NOTEBOOK_PATH


def clear_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    return
