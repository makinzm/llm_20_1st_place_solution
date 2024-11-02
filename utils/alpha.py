import random
import pickle
import re

from .util import ROOT_PATH
from .util import ObjectHavingQuestions, Cfg

VERSION = 100


def extract_testword(question):
    question_pattern = r'^Does the keyword \(in lowercase\) precede "([a-zA-Z\s]+)" in alphabetical order\?$'
    if not re.match(question_pattern, question):
        return None
    match = re.match(question_pattern, question)
    compare_word = match.group(1)
    return compare_word


def get_list_keywords_with_priority():
    path_list_keywords_with_priority = (
        ROOT_PATH / "data_submit" / "list_keywords_all_with_input_with_priority.pkl"
    )
    with open(path_list_keywords_with_priority, "rb") as f:
        list_keywords_with_priority = pickle.load(f)
    # list_keywords_with_priority = [
    #     (keyword, 1) for keyword, priority in list_keywords_with_priority
    # ]
    list_keywords_with_priority.sort(key=lambda x: x[0])
    return list_keywords_with_priority


def sieve_list_keywords_with_priority(list_keywords_with_priority, obs):
    questions = obs["questions"]
    answers = obs["answers"]
    guesses = obs["guesses"]
    for question, answer in zip(questions, answers):
        testword = extract_testword(question)
        if testword is None:
            continue
        # get index of testword
        idx_testword = -1
        for idx, (keyword, priority) in enumerate(list_keywords_with_priority):
            if keyword == testword:
                idx_testword = idx
                break
        if idx_testword == -1:
            continue
        if answer == "yes":
            list_keywords_with_priority = list_keywords_with_priority[:idx_testword]
        else:
            list_keywords_with_priority = list_keywords_with_priority[idx_testword:]
    # delete guesses from list_keywords_with_priority
    list_keywords_with_priority = [
        (keyword, priority)
        for keyword, priority in list_keywords_with_priority
        if keyword not in guesses
    ]
    return list_keywords_with_priority


def get_mid_testword(list_keywords_with_priority):
    if len(list_keywords_with_priority) == 0:
        return None
    # get priority sum
    priority_sum = sum([priority for keyword, priority in list_keywords_with_priority])
    print(
        f"priority_sum: {priority_sum}  number of keywords: {len(list_keywords_with_priority)}"
    )
    # get mid testword
    mid_priority = priority_sum / 2
    priority_sum = 0
    for keyword, priority in list_keywords_with_priority:
        priority_sum += priority
        if priority_sum >= mid_priority:
            print(f"selected priority: {priority}")
            return keyword
    return None


def agent_alpha(obs: ObjectHavingQuestions, _cfg: Cfg):
    list_keywords_with_priority = get_list_keywords_with_priority()
    list_keywords_with_priority = sieve_list_keywords_with_priority(
        list_keywords_with_priority, obs
    )
    testword = get_mid_testword(list_keywords_with_priority)
    if testword is None:
        return None
    if obs.turnType == "ask":
        testword = obs["guesses"][-1]
        response = f'Does the keyword (in lowercase) precede "{testword}" in alphabetical order?'
    elif obs.turnType == "guess":
        response = testword
    else:
        raise ValueError(f"Invalid turnType: {obs.turnType}")
    return response
