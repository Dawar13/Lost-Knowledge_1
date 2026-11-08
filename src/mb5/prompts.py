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

# Worked examples for the yes/no recognition probe, shown to BOTH base and chat
# models (G0 attempt 3, option A in notes/decisions.md). Attempts 1 and 2 gave the
# base model only the first and last example and the chat model none; the chat
# model then said "Yes" too readily to plausible sibling foils. Two Yes, two No;
# Kilimanjaro is a near miss (right type, neighbouring country). None of these is
# a TailCat member or foil.
YESNO_EXAMPLES = [
    ("Paris", "a city in France", "Yes"),
    ("Mount Kilimanjaro", "a mountain in Kenya", "No"),
    ("Toni Morrison", "a novelist from the United States", "Yes"),
    ("Lake Victoria", "a mountain in Peru", "No"),
]

# Two versions of the yes/no probe (notes/decisions.md, G0 attempts 2 and 3; option B):
#   apriori  the protocol fixed before any data (MAIN): chat models get the question alone,
#            base models get the Paris and Lake Victoria examples so they answer in format.
#   worked4  all four examples for both model types (G0 attempt 3; a sensitivity analysis:
#            it raised the chat model's Yes rate and lowered AUROC).
YESNO_PROBES = {"apriori": {"chat": [], "base": [YESNO_EXAMPLES[0], YESNO_EXAMPLES[3]]},
                "worked4": {"chat": YESNO_EXAMPLES, "base": YESNO_EXAMPLES}}
BASE_YESNO_SHOTS = "".join(f"Question: {YESNO.format(name=n, singular=s)}\nAnswer: {a}\n\n"
                           for n, s, a in YESNO_PROBES["apriori"]["base"])


def render_yesno(tok, name: str, singular: str, chat: bool, probe: str = "apriori") -> str:
    """Yes/no recognition prompt: examples as Question/Answer pairs for base
    models, as prior user/assistant turns for chat models."""
    q = YESNO.format(name=name, singular=singular)
    shots = YESNO_PROBES[probe]["chat" if chat else "base"]
    if not chat:
        pre = "".join(f"Question: {YESNO.format(name=n, singular=s)}\nAnswer: {a}\n\n" for n, s, a in shots)
        return f"{pre}Question: {q}\nAnswer:"
    msgs = []
    for n, s, a in shots:
        msgs += [{"role": "user", "content": YESNO.format(name=n, singular=s)}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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
