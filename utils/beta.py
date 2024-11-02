import time
import random

import numpy as np
import torch
import pickle

from .alpha import agent_alpha, VERSION
from .beta_answer import agent_beta_answer
from .util import ROOT_PATH, is_kaggle_agent, QUESTION_IS_ALPHA
from .util import ObjectHavingQuestions

# device = "cpu"
device = "cuda"
FLAG_FLOAT = True

RANDOM_GUESS_CUTOFF = 0.99
# PROB_WEIGHT = 0.95
PROB_WEIGHT = 0.9
USE_EQUAL_PROB = False


PROB_MATRIX = None


def get_prob_matrix():
    global PROB_MATRIX
    if PROB_MATRIX is not None:
        return PROB_MATRIX
    # prob_cutoff = 0.1
    # prob_cutoff = 0.0001
    path_prob_matrix = ROOT_PATH / "data_submit" / "data_submit.pt"
    data = torch.load(path_prob_matrix)
    if FLAG_FLOAT:
        data = data.float()
    data = data.to(device)
    # data[data < prob_cutoff] = prob_cutoff
    # data[data > 1 - prob_cutoff] = 1 - prob_cutoff
    data = 0.5 + PROB_WEIGHT * (data - 0.5)
    PROB_MATRIX = data
    return data


def get_list_questions():
    path_list_questions = ROOT_PATH / "data_submit" / "list_questions.pkl"
    with open(path_list_questions, "rb") as f:
        list_questions = pickle.load(f)
    return list_questions


def get_list_keywords():
    path_list_keywords = ROOT_PATH / "data_submit" / "list_keywords.pkl"
    with open(path_list_keywords, "rb") as f:
        list_keywords = pickle.load(f)
    return list_keywords


def get_probs_initial():
    list_keywords = get_list_keywords()
    Nkeywords = len(list_keywords)
    if USE_EQUAL_PROB:
        probs_initial = torch.ones(Nkeywords, dtype=torch.float32) / Nkeywords
    else:
        path_keyword_probs = ROOT_PATH / "data_submit" / "keyword_probs.pt"
        probs_initial = torch.load(path_keyword_probs)
    assert probs_initial.shape == (Nkeywords,)
    if FLAG_FLOAT:
        probs_initial = probs_initial.float()
    probs_initial = probs_initial.to(device)
    return probs_initial


def get_probs(obs, cfg, data, list_questions, list_keywords):
    questions = obs["questions"]
    answers = obs["answers"]
    guesses = obs["guesses"]
    probs = get_probs_initial()
    for question, answer in zip(questions, answers):
        if question == QUESTION_IS_ALPHA:
            continue
        if "Does the keyword (in lowercase) precede" in question:
            continue
        # TODO consider alpha agent
        idx_question = list_questions.index(question)
        if answer == "yes":
            probs *= data[idx_question]
        else:
            probs *= 1 - data[idx_question]
        probs /= probs.sum()
    for guess in guesses:
        # if guess == GUESS_IS_ALPHA:
        #     continue
        if guess not in list_keywords:
            continue
        idx_guess = list_keywords.index(guess)
        probs[idx_guess] = 0
    probs /= probs.sum()
    return probs


def get_idx_questions_asked(obs, list_questions):
    questions = obs["questions"]
    idx_questions_asked = []
    for question in questions:
        if question == QUESTION_IS_ALPHA:
            continue
        if "Does the keyword (in lowercase) precede" in question:
            continue
        if question in list_questions:
            idx_questions_asked.append(list_questions.index(question))
    return idx_questions_asked


def agent_beta(obs: ObjectHavingQuestions, cfg):
    """This agent function is a placeholder, roll out your own! Use LLM or any alternative."""

    if obs.turnType == "ask":
        n_question = len(obs["questions"]) + 1
        # rand double
        # if is_kaggle_agent():
        #     time_sleep = random.uniform(20, 40)
        #     time.sleep(time_sleep)
        data = get_prob_matrix()
        list_questions = get_list_questions()
        list_keywords = get_list_keywords()
        probs = get_probs(obs, cfg, data, list_questions, list_keywords)
        print(f"data.shape: {data.shape}")
        print(f"probs.shape: {probs.shape}")
        # entropy_bef_val = -(torch.special.xlogy(probs, probs)).sum(axis=0).item()
        # log 2
        entropy_bef_val = (
            -(torch.special.xlogy(probs, probs) / np.log(2)).sum(axis=0).item()
        )

        probs_new_yes = data * probs
        probs_new_no = (1 - data) * probs
        probs_new_yes_sum = probs_new_yes.sum(axis=1)
        probs_new_no_sum = probs_new_no.sum(axis=1)
        # print(f"probs_new_yes.shape: {probs_new_yes.shape}")
        # print(f"probs_new_yes_sum.shape: {probs_new_yes_sum.shape}")
        probs_new_yes /= probs_new_yes.sum(axis=1, keepdim=True)
        probs_new_no /= probs_new_no.sum(axis=1, keepdim=True)
        probs_new_yesno_sum = probs_new_yes_sum + probs_new_no_sum
        probs_new_yes_sum /= probs_new_yesno_sum
        probs_new_no_sum /= probs_new_yesno_sum
        # entropy_new_yes = -(torch.special.xlogy(probs_new_yes, probs_new_yes)).sum(
        #     axis=1
        # )
        # entropy_new_no = -(torch.special.xlogy(probs_new_no, probs_new_no)).sum(axis=1)
        # log 2
        entropy_new_yes = -(
            torch.special.xlogy(probs_new_yes, probs_new_yes) / np.log(2)
        ).sum(axis=1)
        entropy_new_no = -(
            torch.special.xlogy(probs_new_no, probs_new_no) / np.log(2)
        ).sum(axis=1)
        if 1:
            # use expected entropy
            entropy_new = (
                probs_new_yes_sum * entropy_new_yes + probs_new_no_sum * entropy_new_no
            )
        else:
            # use maximum entropy
            entropy_new = torch.maximum(entropy_new_yes, entropy_new_no)

        entropy_new[torch.isnan(entropy_new)] = 1e9
        # get question with minimum entropy, but has not been asked
        idx_questions_asked = get_idx_questions_asked(obs, list_questions)
        entropy_new[idx_questions_asked] = 1e9
        idx_question = entropy_new.argmin().item()
        question = list_questions[idx_question]
        print(f"question {n_question}: {question}")
        response = question

        entropy_new_val = entropy_new[idx_question].item()
        entropy_new_yes_val = entropy_new_yes[idx_question].item()
        entropy_new_no_val = entropy_new_no[idx_question].item()
        entropy_gain = entropy_bef_val - entropy_new_val

        print(f"entropy_bef_val: {entropy_bef_val:.4f}")
        print(f"entropy_new_val: {entropy_new_val:.4f}")
        print(f"entropy_new_yes_val: {entropy_new_yes_val:.4f}")
        print(f"entropy_new_no_val: {entropy_new_no_val:.4f}")
        print(f"entropy_gain: {entropy_gain:.4f}")

        return question

    elif obs.turnType == "guess":
        data = get_prob_matrix()
        list_questions = get_list_questions()
        list_keywords = get_list_keywords()
        probs = get_probs(obs, cfg, data, list_questions, list_keywords)
        # print(f"keyword: {keyword}, prob (confidence): {prob}")

        # print keywords with high probability
        n_print = 10
        idx_keywords = probs.argsort(descending=True)
        print(f"top {n_print} keywords with high probability.")
        for idx_keyword in idx_keywords[:10]:
            keyword = list_keywords[idx_keyword]
            prob = probs[idx_keyword].item()
            print(f"({prob:.4f}, {keyword})")

        if RANDOM_GUESS_CUTOFF == 1:
            idx_guess = probs.argmax().item()
        else:
            # introduce some randomness
            random.seed(int(time.time() * 1000))
            prob_max = probs.max().item()
            prob_cutoff = prob_max * RANDOM_GUESS_CUTOFF
            # get indexes with probability above cutoff
            idxs = torch.where(probs > prob_cutoff)[0]
            # get random index
            idx_guess = random.choice(idxs).item()
        # idx_guess = probs.argmin().item()  # TODO for testing
        keyword = list_keywords[idx_guess]
        response = keyword
        prob = probs[idx_guess].item()

        return keyword

    else:
        return agent_beta_answer(obs, cfg)

    return response
