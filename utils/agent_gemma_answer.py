import sys
import re
import time
import subprocess
import gc

import numpy as np
import transformers
import torch

from .util import get_last_question, get_keyword, is_valid_answer
from .model_gemma import model_gemma_load, model_to_cpu
from .model_all import delete_all_models


TEST = True


def format_question_from_deepmath(obs):
    question_raw = get_last_question(obs)
    keyword = get_keyword(obs)

    question_formatted = f'The keyword is "{keyword}". '
    question_formatted += question_raw
    # question_formatted = f'The keyword is "qatar". '
    # question_formatted += "Is the secret word a man-made structure?"
    question_formatted += " Answer to the question above with 'yes' or 'no'."

    return question_formatted


def get_answer_from_gemma(model, tokenizer, obs, cfg, question_formatted):
    messages = []
    messages.append(
        {
            "role": "user",
            "content": question_formatted,
        }
    )
    query_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # pipeline = transformers.pipeline(
    #     task="text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )
    # raw_output = pipeline(
    #     query_prompt,
    #     max_new_tokens=1024,
    #     do_sample=True,
    #     temperature=0.1,
    #     return_full_text=False,
    # )

    print(f"query_prompt: {query_prompt}")

    try:
        # get probability distribution of first token
        inputs = tokenizer(query_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # print(f"inputs: {inputs}")
        with torch.no_grad():
            output = model(**inputs)
        # print(f"output: {output}")
        logits = output.logits
        print(f"logits.shape: {logits.shape}")
        logits = logits[0, -1, :]
        print(f"logits.dtype: {logits.dtype}")
        print(f"logits.device: {logits.device}")
        print(f"type(logits): {type(logits)}")
        print(f"logits.shape: {logits.shape}")
        probs = torch.softmax(logits, dim=0)
        print(f"probs.shape: {probs.shape}")
        probs = probs.cpu().detach().numpy()
        print(f"probs.shape: {probs.shape}")
        probs_sum = probs.sum()
        print(f"probs.shape: {probs.shape}")
        time.sleep(2)
        "probability of yes and no"
        words = ["yes", "Yes", "no", "No"]
        prob_words = []
        for word in words:
            idx = tokenizer.convert_tokens_to_ids(word)
            prob = probs[idx] / probs_sum
            print(f"prob({word}): {prob}")
            print(f"prob({word}): {prob}", file=sys.stderr)
            prob_words.append(prob)
        prob_yes = prob_words[0] + prob_words[1]
        prob_no = prob_words[2] + prob_words[3]
        ret = "yes" if prob_yes > prob_no else "no"
        # # print 10 most probable words
        # print(f"probs.shape: {probs.shape}")
        # idxs = np.argsort(probs)[::-1][:10]
        # print(f"idxs: {idxs}")
        # for idx in idxs:
        #     print(f"idx: {idx}")
        #     word = tokenizer.convert_ids_to_tokens([idx])
        #     print(f"word: {word}")
        #     prob = probs[idx] / probs_sum
        #     print(f"prob({word}): {prob}")
        #     print(f"prob({word}): {prob}")
        # ret = "yes"
        print()
        # time.sleep(2)
        print("success in get_answer_from_gemma")
        print("success in get_answer_from_gemma", file=sys.stderr)
    except Exception as e:
        print("error in get_answer_from_gemma")
        print(e)
        ret = "yes"
    print("end get_answer_from_gemma", file=sys.stderr)
    print("end get_answer_from_gemma")
    return ret


def agent_gemma_answer(obs, cfg):
    # return "no"
    # return "yes"

    delete_all_models("gemma")
    model, tokenizer = model_gemma_load()
    assert model is not None
    assert tokenizer is not None

    question_formatted = format_question_from_deepmath(obs)

    ans = get_answer_from_gemma(model, tokenizer, obs, cfg, question_formatted)

    ans = ans.lower()
    if not is_valid_answer(ans):
        ans = "no"

    print(f"end agent_deepmath_answer. ans={ans}")

    # print(f"send model to cpu")
    # print(f"model device: {model.device}")
    # try:
    #     model = model_to_cpu()
    # except Exception as e:
    #     print(e)
    #     model = None

    # print device
    print(f"model device: {model.device}")

    # clear memory
    print(f"clear memory")
    gc.collect()
    torch.cuda.empty_cache()

    # sleep
    # time.sleep(2)

    return ans
