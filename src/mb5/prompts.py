"""Prompt templates and tokenisation of (prompt, answer) pairs.

Chat models get their own chat template (including any default system
prompt the template inserts: that is the shipped behaviour). Base models get
a plain "Question/Answer" frame.

The answer is tokenised *in context*: we tokenise prompt+answer jointly and
take the tail, so the token ids are the ones the model would actually emit.
If the joint tokenisation does not extend the prompt's ids cleanly we fall
back to separate tokenisation and flag it (boundary_ok=False).
"""
from __future__ import annotations

GEN_TEMPLATES = [
    "Name {singular}. Reply with only the name.",
    "Give one example of {plural}. Answer with the name only.",
    "Tell me {singular}. Just the name, nothing else.",
    "I need {singular}. Write only its name.",
    "Which is {singular}? Respond with the name alone.",
]

YESNO = 'Is "{name}" {singular}? Answer Yes or No.'

MC = (
    "Which of the following is {singular}?\n"
    "A. {a}\nB. {b}\nC. {c}\nD. {d}\n"
    "Answer with the letter only."
)

# Few-shot frames so base models answer in the right format. The examples are
# deliberately unrelated to every TailCat category.
BASE_YESNO_SHOTS = (
    'Question: Is "Paris" a city in France? Answer Yes or No.\nAnswer: Yes\n\n'
    'Question: Is "Lake Victoria" a mountain in Peru? Answer Yes or No.\nAnswer: No\n\n'
)
BASE_MC_SHOTS = (
    "Question: Which of the following is a planet?\nA. Sirius\nB. Mars\nC. Andromeda\nD. Halley\n"
    "Answer with the letter only.\nAnswer: B\n\n"
)
BASE_GEN_SHOTS = (
    "Question: Name a country in South America. Reply with only the name.\nAnswer: Peru\n\n"
)


def render(tok, user_text: str, chat: bool, shots: str = "") -> str:
    if chat:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
        )
    return f"{shots}Question: {user_text}\nAnswer:"


def answer_prefix(chat: bool) -> str:
    """Base models continue 'Answer:' with a space; chat models start fresh."""
    return "" if chat else " "


def encode_prompt(tok, prompt: str, chat: bool) -> list[int]:
    # Chat templates already contain any BOS token as text.
    return tok(prompt, add_special_tokens=not chat).input_ids


def encode_pair(tok, prompt: str, answer: str, chat: bool):
    p_ids = encode_prompt(tok, prompt, chat)
    joint = tok(prompt + answer, add_special_tokens=not chat).input_ids
    if len(joint) > len(p_ids) and joint[: len(p_ids)] == p_ids:
        return p_ids, joint[len(p_ids):], True
    a_ids = tok(answer, add_special_tokens=False).input_ids
    return p_ids, a_ids, False


def first_token_ids(tok, words: list[str]) -> list[int]:
    ids = []
    for w in words:
        enc = tok(w, add_special_tokens=False).input_ids
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def yes_no_tokens(tok, chat: bool):
    pre = "" if chat else " "
    yes = first_token_ids(tok, [pre + "Yes", pre + "yes", pre + "YES"])
    no = first_token_ids(tok, [pre + "No", pre + "no", pre + "NO"])
    return yes, no


def letter_tokens(tok, chat: bool):
    pre = "" if chat else " "
    return [first_token_ids(tok, [pre + L])[0] for L in "ABCD"]
