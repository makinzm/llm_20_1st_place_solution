from .model_gemma import delete_model as delete_model_gemma
from .model_deepmath import delete_model as delete_model_deepmath


def delete_all_models(model_name=None):
    if model_name != "gemma":
        delete_model_gemma()
    if model_name != "deepmath":
        delete_model_deepmath()
    return
