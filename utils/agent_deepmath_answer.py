import sys
import os
import re
import time
import subprocess
import gc

import transformers
import torch

from .util import get_last_question, get_keyword, is_valid_answer, is_local
from .model_deepmath import model_deepmath_load, model_to_cpu
from .model_gemma import model_gemma_load
from .model_all import delete_all_models

TEST = True


def post_process(output):
    if output is None:
        return None
    try:
        # strip
        output = re.sub(r"[^a-zA-Z]", "", output)
        output = output.lower()
        if output in ["yes", "true", "1"]:
            return "yes"
        elif output in ["no", "false", "0"]:
            return "no"
        else:
            return None
            # raise ValueError(f"Unexpected output: {output} in post_process")
    except:
        raise ValueError(f"Unexpected output: {output} in post_process")


def process_output(output, obs=None, n_try=None):
    result = output
    if not is_local():
        print("output")
        print(output)

    try:
        code_file = "code.py"
        if obs is not None:
            # make dir code
            os.makedirs("code", exist_ok=True)
            n_turn = len(obs["questions"])
            code_file = f"code/{n_turn:02}_{n_try}.py"
            print(code_file)

            os.makedirs("raw_output", exist_ok=True)
            with open(f"raw_output/{n_turn:02}_{n_try}.txt", "w") as fout:
                fout.write(output)

        code = output.split("```")[1][7:]
        with open(code_file, "w") as fout:
            fout.write(code)

        batcmd = "timeout 2 " + sys.executable + f" {code_file}"
        try:
            shell_output = subprocess.check_output(batcmd, shell=True).decode("utf8")
            print("SHELL OUTPUT", shell_output)
            code_output = post_process(shell_output)
            print("CODE RESULTS", code_output)
        except:
            code_output = None

    except Exception as e:
        print(e)
        print("ERROR PARSING")
        code_output = None

    try:
        result_output = re.findall(r"\\boxed\{(.*)\}", result)

        print("BOXED", result_output)
        if not len(result_output):
            result_output = None
        else:
            result_output = result_output[-1]

        print("BOXED", result_output)
        if not len(result_output):
            result_output = None

    except Exception as e:
        print(e)
        print("ERROR PARSING")
        result_output = None

    code_output = post_process(code_output)
    result_output = post_process(result_output)

    print(f"FINAL CODE OUTPUT: {code_output}")
    print(f"FINAL RESULT OUTPUT: {result_output}")

    return result_output, code_output


def format_question_from_deepmath(obs):
    question_raw = get_last_question(obs)
    keyword = get_keyword(obs).lower()

    # replace \"A\" to \"a\"
    for i in range(26):
        c_lower = chr(ord("a") + i)
        c_upper = chr(ord("A") + i)
        question_raw = question_raw.replace(f'"{c_upper}"', f'"{c_lower}"')
        question_raw = question_raw.replace(f"'{c_upper}'", f"'{c_lower}'")

    # TODO
    # question_raw = "Does the keyword start with one of the letters 'a', 'b', 'c', 'd', 'j', 'u' or 't'?"

    # question_formatted = f'The keyword is "{keyword}". '
    question_formatted = (
        f'The keyword is "{keyword}" (as a whole, so do not sort or split it). '
    )
    question_formatted += question_raw

    # # TODO
    # question_formatted += " Be case insensitive."

    # for deepmath
    question_formatted = question_formatted.replace(" sorting ", " lexicographical ")

    return question_formatted


def get_answer_from_deepmath(model, tokenizer, obs, cfg, question_formatted, n_try):
    keyword = get_keyword(obs).lower()

    tool_instruction = ""
    # with program
    tool_instruction += "\nPlease integrate step by step natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
    # tool_instruction += "\n use only default python libraries."

    messages = []
    messages.append({"role": "user", "content": question_formatted + tool_instruction})

    query_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # for faster computation
    output_head = ""
    output_head += "```python\n"
    output_head += "def solve():\n"
    output_head += f'    """{question_formatted}"""\n'
    output_head += f'    keyword = "{keyword}"\n'
    output_head += "    "
    query_prompt += output_head

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype="auto",
        device_map="auto",
    )
    raw_output = pipeline(
        query_prompt,
        max_new_tokens=1536,
        do_sample=True,
        # temperature=0.1,
        temperature=0.7,
        # temperature=0.9,
        return_full_text=False,
        stop_strings=["```output"],
        tokenizer=tokenizer,
    )
    raw_output = raw_output[0]["generated_text"]
    raw_output = output_head + raw_output

    # # generate using model.generate
    # input_ids = tokenizer(query_prompt, return_tensors="pt").input_ids
    # input_ids = input_ids.to(model.device)
    # max_length = 1024
    # temperature = 0.7
    # num_return_sequences = 1
    # do_sample = True
    # top_k = 50
    # top_p = 0.95
    # repetition_penalty = 1.0
    # bad_words_ids = None
    # pad_token_id = None
    # eos_token_id = None
    # length_penalty = 1.0
    # no_repeat_ngram_size = 0
    # num_beam_groups = 1
    # diversity_penalty = 0.0
    # use_cache = True
    # output_scores = None
    # output_attentions = None
    # output_hidden_states = None
    # output_attentions = None
    # output_hidden_states = None
    # output_scores = None
    # return_dict_in_generate = None
    # with torch.no_grad():
    #     output = model.generate(
    #         input_ids,
    #         max_length=max_length,
    #         temperature=temperature,
    #         num_return_sequences=num_return_sequences,
    #         do_sample=do_sample,
    #         top_k=top_k,
    #         top_p=top_p,
    #         # repetition_penalty=repetition_penalty,
    #         # bad_words_ids=bad_words_ids,
    #         # pad_token_id=pad_token_id,
    #         # eos_token_id=eos_token_id,
    #         # length_penalty=length_penalty,
    #         # no_repeat_ngram_size=no_repeat_ngram_size,
    #         # num_beam_groups=num_beam_groups,
    #         # diversity_penalty=diversity_penalty,
    #         # use_cache=use_cache,
    #         # output_scores=output_scores,
    #         # output_attentions=output_attentions,
    #         # output_hidden_states=output_hidden_states,
    #         # return_dict_in_generate=return_dict_in_generate,
    #     )
    #     raw_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    #     print("raw_output", raw_output, file=sys.stderr)

    # if TEST:
    #     print(raw_output)

    result_output, code_output = process_output(raw_output, obs, n_try)

    print("result_output", result_output)
    print("code_output", code_output)

    if not is_valid_answer(code_output):
        code_output = None
    # elif is_valid_answer(result_output):
    #     return result_output
    return code_output


def agent_deepmath_answer(obs, cfg):
    print("begin agent_deepmath_answer")
    delete_all_models("deepmath")
    model, tokenizer = model_deepmath_load()
    assert model is not None
    assert tokenizer is not None

    question_formatted = format_question_from_deepmath(obs)
    print(f"question_formatted: {question_formatted}")

    N_max = 5
    ans_dict = {}
    for n in range(N_max):
        try:
            ans = get_answer_from_deepmath(
                model, tokenizer, obs, cfg, question_formatted, n
            )
            if ans is not None:
                ans_dict[ans] = ans_dict.get(ans, 0) + 1
                if ans_dict[ans] > N_max // 2:
                    break
        except Exception as e:
            print(e)
    print(f"ans_dict: {ans_dict}")
    print(f"ans_dict: {ans_dict}", file=sys.stderr)
    if len(ans_dict) == 0:
        ans = "no"
    else:
        ans = max(ans_dict, key=ans_dict.get)

    # ans = ans.lower()
    # if not is_valid_answer(ans):
    #     ans = "no"

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
