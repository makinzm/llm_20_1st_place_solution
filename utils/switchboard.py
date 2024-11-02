from .alpha import agent_alpha, VERSION
from .beta import agent_beta
from .util import QUESTION_IS_ALPHA
from .util import ObjectHavingQuestions

FLAG_USE_ALPHA = True


def switchboard(obs: ObjectHavingQuestions, cfg):
    """The main switchboard, deciding which sub-agent to call."""

    if obs.turnType == "ask":
        # return (
        #     # "Does the keyword start with one of the letters 'S', 'A', 'K' or 'D'?"
        #     "Does the keyword start with the letter 'D'"
        # )
        response = None
        if FLAG_USE_ALPHA:
            if len(obs.questions) == 0:
                response = QUESTION_IS_ALPHA
            else:
                if obs.answers[0] == "yes":
                    response = agent_alpha(obs, cfg)
                else:
                    response = agent_beta(obs, cfg)
        if response is None:
            response = agent_beta(obs, cfg)

    elif obs.turnType == "guess":
        # return "guess"
        response = None
        if FLAG_USE_ALPHA and obs.answers[0] == "yes":
            response = agent_alpha(obs, cfg)
        if response is None:
            response = agent_beta(obs, cfg)

        # assert len(obs.questions) > 0 and len(obs.answers) > 0
        # if len(obs.answers) == 1:
        #     if obs.answers[-1] == "yes":
        #         response = f"bingo! version {VERSION}"
        #     else:
        #         response = f"too bad.."
        #         # Or we may call agent beta for help, straightaway
        # else:
        #     # We will play alpha if we got 'yes' to the first question.
        #     play_alpha = obs.answers[0] == "yes"
        #     if play_alpha:
        #         response = agent_alpha(obs, cfg)
        #     else:
        #         response = agent_beta(obs, cfg)

    elif obs.turnType == "answer":
        # TODO
        # return "no"

        print(f"last question: {obs.questions[-1]}")

        assert len(obs.questions) > 0
        # questioner_is_alpha = "agent alpha" in obs.questions[0].lower()
        questioner_is_alpha = False

        if questioner_is_alpha:
            response = agent_alpha(obs, cfg)

        else:
            # Well, the questioner isn't alpha, so.. use LLM?
            response = agent_beta(obs, cfg)

        # assert response in ["yes", "no"]
        if response not in ["yes", "no"]:
            print(f"ERROR in response: {response}")
            response = "no"

    else:
        # We don't expect to ever land here
        assert False, f"Unexpected turnType: {obs.turnType}"

    return response
