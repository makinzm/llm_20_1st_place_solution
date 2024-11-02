import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    set_seed,
)
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from accelerate import init_empty_weights

from .util import (
    is_local,
    is_kaggle_notebook,
    is_kaggle_agent,
    KAGGLE_AGENT_PATH,
    KAGGLE_NOTEBOOK_PATH,
    clear_cache,
)

set_seed(42)

MODEL_PATH = (
    "meta-llama/Meta-Llama-3-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    if is_local()
    else (
        KAGGLE_NOTEBOOK_PATH / "meta-llama-3-8b-instruct-quant-8bit"
        if is_kaggle_notebook()
        else KAGGLE_AGENT_PATH / "meta-llama-3-8b-instruct-quant-8bit"
    )
)

# QUANT = 4
QUANT = 8
# QUANT = None

tokenizer = None
model = None


def get_device():
    # get device of model
    device = model.device
    if "cuda" in str(device):
        device = "cuda"
    else:
        device = "cpu"
    return device


def model_to_gpu():
    global model
    if QUANT is not None:
        return model

    if model is None:
        return
    if get_device() == "cuda":
        return

    model.to("cuda")
    return model


def delete_model():
    global model
    global tokenizer
    model = None
    tokenizer = None
    clear_cache()
    return


def model_to_cpu():
    global model
    if QUANT is not None:
        return model

    if model is None:
        return
    if get_device() == "cpu":
        return

    model.to("cpu")
    return model


def model_gemma_load():
    global model
    global tokenizer

    if model is not None:
        model_to_gpu()
        return model, tokenizer

    print(f"is_local: {is_local()}")
    print(f"is_kaggle_notebook: {is_kaggle_notebook()}")
    print(f"is_kaggle_agent: {is_kaggle_agent()}")

    print("now loading model_gemma")

    print(f"MODEL_PATH: {MODEL_PATH}")
    # check if the model exists
    flag_model_exists = os.path.exists(MODEL_PATH)
    print(f"flag_model_exists: {flag_model_exists}")

    print("disable mem_efficient_sdp and flash_sdp")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    print("quantization config")
    quantization_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_has_fp16_weight=False,
    )

    print("AutoConfig.from_pretrained")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    config.gradient_checkpointing = True

    print("AutoTokenizer.from_pretrained")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    assert QUANT in [None, 4, 8]

    if QUANT == 4:
        assert False  # because weights are initially 8-bit
        print(f"AutoModelForCausalLM.from_pretrained, with quantization_config {QUANT}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            quantization_config=quantization_config_4bit,
            config=config,
        )
    elif QUANT == 8:
        print(f"AutoModelForCausalLM.from_pretrained, with quantization_config {QUANT}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_8bit=True,
            config=config,
            device_map="cuda:0",
            torch_dtype="auto",
        )
        # config = AutoConfig.from_pretrained(MODEL_PATH)
        # with init_empty_weights():
        #     model_empty = AutoModelForCausalLM.from_config(config)
        # bnb_quantization_config = BnbQuantizationConfig(
        #     load_in_8bit=True,
        # )
        # model = load_and_quantize_model(
        #     model_empty,
        #     weights_location=MODEL_PATH,
        #     bnb_quantization_config=bnb_quantization_config,
        #     device_map="auto",
        # )
    else:
        assert False  # because weights are initially 8-bit
        print("AutoModelForCausalLM.from_pretrained, without quantization_config")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            config=config,
        )

    print("successfully loaded model_gemma")

    return model, tokenizer
