# %%
import re
import sys

from .util import is_local, get_last_question, get_keyword
from .agent_deepmath_answer import agent_deepmath_answer
from .agent_gemma_answer import agent_gemma_answer

import re


def func0(keyword, question):
    keyword_pattern = r"^[a-zA-Z\s]+$"
    # question_pattern = r'^Does the keyword \(in lowercase\) precede "([a-zA-Z\s]+)" in alphabetical order\?$'
    # if not re.match(keyword_pattern, keyword) or not re.match(
    #     question_pattern, question
    # ):
    #     return None
    # match = re.match(question_pattern, question)
    match = re.search(
        r"keyword.*(?:come before|precede) \"([^\"]+)\" .+ order\?$", question
    )
    if match:
        compare_word = match.group(1)
        return keyword.lower() < compare_word.lower()
    else:
        return None


def func1(keyword, question):
    keyword_pattern = r"^[a-zA-Z\s]+$"
    question_pattern = r'^Does the keyword \(in lowercase\) come before "([a-zA-Z\s]+)" in alphabetical order\?$'
    if not re.match(keyword_pattern, keyword) or not re.match(
        question_pattern, question
    ):
        return None
    match = re.match(question_pattern, question)
    compare_word = match.group(1)
    return keyword.lower() < compare_word.lower()


def func2(keyword, question):
    keyword_pattern = r"^[a-zA-Z\s]+$"
    # question_pattern = r'^Does the keyword begins with the letter "([a-zA-Z])"\?$'
    question_pattern = [
        r'^Does the keyword begin with the letter "([a-zA-Z])"\?$',
        r'^Does the keyword start with the letter "([a-zA-Z])"\?$',
        r"^Does the keyword begin with the letter \'([a-zA-Z])\'$",
        r"^Does the keyword start with the letter \'([a-zA-Z])\'$",
        r"^Does the keyword begins with the letter \'([a-zA-Z])\'\?$",
        r"^Does the keyword begins with the letter \"([a-zA-Z])\"\?$",
    ]

    if not re.match(keyword_pattern, keyword) or not any(
        re.match(pattern, question) for pattern in question_pattern
    ):
        print("### NOT MATCH ###")
        return None
    search_letter = None
    for pattern in question_pattern:
        match = re.match(pattern, question)
        if match:
            search_letter = match.group(1)
            break
    if search_letter is None:
        return None
    return keyword.strip().lower().startswith(search_letter.lower())

    # if not re.match(keyword_pattern, keyword) or not re.match(
    #     question_pattern, question
    # ):
    #     return None

    # match = re.match(question_pattern, question)
    # search_letter = match.group(1)

    # return keyword.strip().lower().startswith(search_letter.lower())


def func3(keyword, question):
    keyword_pattern = r"^[a-zA-Z\s]+$"
    question_patterns = [
        r"^Does the keyword start with one of the letters \'([a-zA-Z]\'(?:, \'[a-zA-Z]\')*)(?: or \'[a-zA-Z]\')?\?$",
        r"^Does the keyword start with the letter \'([a-zA-Z])\'\?$",
    ]
    if not re.match(keyword_pattern, keyword) or not any(
        re.match(pattern, question) for pattern in question_patterns
    ):
        return None
    if re.match(question_patterns[0], question):
        letters = re.findall(r"'([a-zA-Z])'", question)
    else:
        match = re.match(question_patterns[1], question)
        letters = [match.group(1)]
    letters = [c.lower() for c in letters]
    return keyword.strip()[0].lower() in letters


def func4(keyword, question):
    keyword_pattern = r"^[a-zA-Z\s]+$"
    question_pattern = r"^Is the keyword one of the following\? ([a-zA-Z\s,]+)\?$"
    if not re.match(keyword_pattern, keyword) or not re.match(
        question_pattern, question
    ):
        return None
    match = re.match(question_pattern, question)
    options = [option.strip() for option in match.group(1).split(",")]
    return keyword.strip().lower() in [option.lower() for option in options]


def func5(keyword, question):
    keyword_pattern = r"^[a-zA-Z\s]+$"
    question_pattern = r"^Considering every letter in the name of the keyword, does the name of the keyword include the letter \'([A-Za-z])\'\?$"
    if not re.match(keyword_pattern, keyword) or not re.match(
        question_pattern, question
    ):
        return None
    match = re.match(question_pattern, question)
    search_letter = match.group(1)
    return search_letter.lower() in keyword.lower()


def func(keyword, question):
    solves = [func0, func1, func2, func3, func4, func5]
    for f in solves:
        result = f(keyword, question)
        if result is not None:
            return result
    return None


def ask_letter(question):
    # check if the text matches with "Does the keyword ... letter ...
    match = re.search(r"Does the keyword .* letter .*", question)
    # return True if the text matches
    # print(match)
    return bool(match)


def ask_chronological(question):
    words = [
        "alphabetical",
        "lexicographical",
        "sorting",
        "precede",
        "come before",
        "come after",
        "follow",
        "succeed",
        "letter",
    ]
    return any(word in question for word in words)


def use_program(obs, cfg):
    # return True  # TODO: for debug
    # return False  # TODO: for debug
    # return len(obs.questions) % 2 == 1

    question = get_last_question(obs)

    ret = False
    ret = ret or ask_letter(question)
    ret = ret or ask_chronological(question)

    return ret


def agent_beta_answer(obs, cfg):
    assert obs.turnType == "answer"
    turn = len(obs.questions)

    last_question = get_last_question(obs)
    keyword = get_keyword(obs)

    # rule-based
    ret = func(keyword, last_question)
    if ret is not None:
        return "yes" if ret else "no"

    if "Agent Alpha" in last_question and turn == 1:
        return "yes"
        # return "no"

    flag_program = use_program(obs, cfg)

    ret = None

    try:
        if flag_program:
            ret = agent_deepmath_answer(obs, cfg)
        else:
            ret = agent_gemma_answer(obs, cfg)

    except Exception as e:
        print(f"Error: {e}")
        ret = "no"

    if ret is None:
        ret = "no"

    # 標準エラー出力
    print(f"ret: {ret}", file=sys.stderr)

    return ret
