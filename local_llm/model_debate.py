"""
model_debate.py

Two local Ollama models (Gemma and Qwen) each answer a question, then engage in
a structured dialogue. On every turn a model reviews the current best proposal,
critiques it, and either AGREES with it or REVISES it with a better answer.

- If both models converge on a single answer, that agreed answer is printed.
- If they cannot agree within MAX_EXCHANGES turns, each model's own final answer
  and its reasoning for why its answer is better are printed.

Run:  python model_debate.py
"""

import re
import requests

# --- Configuration ---
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Names must match what `ollama list` shows.
MODEL_A = "gemma4:cloud"
MODEL_B = "nemotron-3-super:cloud"

# Display names used throughout the debate output and prompts.
NAME_A = "Gemma"
NAME_B = "Nemotron"

MAX_EXCHANGES = 10  # Total review turns before we give up on consensus.


# --- Ollama helper ---
def ask_model(model, prompt):
    """
    Sends a prompt to an Ollama model and returns the generated text.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_GENERATE_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    text = data.get("response")
    if text is None:
        raise RuntimeError(
            f"Ollama returned no response for model '{model}'. "
            f"Response: {data}. "
            f"Make sure the model is pulled: `ollama pull {model}`"
        )
    return text.strip()


# --- Response parsing ---
def parse_turn(raw):
    """
    Parses a structured debate turn into (decision, critique, proposed_answer).

    Expected format from the model:
        DECISION: AGREE | REVISE
        CRITIQUE: <reasoning>
        PROPOSED_ANSWER: <answer>

    Falls back gracefully if the model doesn't follow the format exactly.
    """
    decision_match = re.search(r"DECISION:\s*(AGREE|REVISE)", raw, re.IGNORECASE)
    critique_match = re.search(
        r"CRITIQUE:\s*(.*?)(?=\nPROPOSED_ANSWER:|\Z)", raw, re.IGNORECASE | re.DOTALL
    )
    answer_match = re.search(
        r"PROPOSED_ANSWER:\s*(.*)", raw, re.IGNORECASE | re.DOTALL
    )

    decision = decision_match.group(1).upper() if decision_match else "REVISE"
    critique = critique_match.group(1).strip() if critique_match else raw.strip()
    proposed = answer_match.group(1).strip() if answer_match else ""

    return decision, critique, proposed


# --- Prompt builders ---
def initial_prompt(question):
    return f"""You are an expert assistant. Answer the following question as clearly
and accurately as possible.

QUESTION:
{question}

Provide only your answer."""


def review_prompt(reviewer_name, question, current_proposal, author_name):
    return f"""You are {reviewer_name}, participating in a debate with another AI to
agree on the single best answer to a question.

QUESTION:
{question}

The current best proposed answer (written by {author_name}) is:
\"\"\"
{current_proposal}
\"\"\"

Critically evaluate this proposed answer. If it is already the best possible
answer and you fully agree with it, AGREE. Otherwise, REVISE it into a better,
more accurate, and clearer answer.

Respond EXACTLY in this format:
DECISION: AGREE or REVISE
CRITIQUE: <your honest reasoning about the current proposal>
PROPOSED_ANSWER: <if REVISE, your improved answer; if AGREE, restate the answer you agree with>"""


def defense_prompt(name, question, own_answer, other_answer):
    return f"""You are {name}. You and another AI could not agree on the best answer
to the following question.

QUESTION:
{question}

YOUR ANSWER:
\"\"\"
{own_answer}
\"\"\"

THE OTHER AI'S ANSWER:
\"\"\"
{other_answer}
\"\"\"

Explain concisely why YOUR answer is better than the other AI's answer."""


# --- Debate loop ---
def run_debate(question):
    print(f"❓ Question: {question}\n")

    # 1. Each model produces an initial answer.
    print(f"🟢 {NAME_A} is drafting an initial answer...")
    answer_a = ask_model(MODEL_A, initial_prompt(question))
    print(f"{NAME_A}:\n{answer_a}\n")

    print(f"🔵 {NAME_B} is drafting an initial answer...")
    answer_b = ask_model(MODEL_B, initial_prompt(question))
    print(f"{NAME_B}:\n{answer_b}\n")

    # The "current proposal" starts as A's answer; B reviews first.
    current_proposal = answer_a
    author_name = NAME_A

    # Track each model's latest proposed answer for the no-consensus fallback.
    latest = {NAME_A: answer_a, NAME_B: answer_b}

    # Reviewers alternate, starting with B (since A authored the proposal).
    reviewers = [
        (NAME_B, MODEL_B),
        (NAME_A, MODEL_A),
    ]

    for exchange in range(1, MAX_EXCHANGES + 1):
        reviewer_name, reviewer_model = reviewers[(exchange - 1) % 2]

        # A model shouldn't review its own proposal; swap if needed.
        if reviewer_name == author_name:
            reviewer_name, reviewer_model = reviewers[exchange % 2]

        print(f"--- Exchange {exchange}: {reviewer_name} reviews {author_name}'s proposal ---")
        raw = ask_model(
            reviewer_model,
            review_prompt(reviewer_name, question, current_proposal, author_name),
        )
        decision, critique, proposed = parse_turn(raw)

        print(f"{reviewer_name} DECISION: {decision}")
        print(f"{reviewer_name} CRITIQUE: {critique}\n")

        if decision == "AGREE":
            print("=" * 60)
            print(f"✅ CONSENSUS REACHED after {exchange} exchange(s)!")
            print("=" * 60)
            print(f"\nAgreed best answer:\n{current_proposal}\n")
            return

        # REVISE: the reviewer becomes the author of the new proposal.
        if proposed:
            current_proposal = proposed
            latest[reviewer_name] = proposed
        author_name = reviewer_name

    # 2. No consensus after MAX_EXCHANGES: each model defends its own answer.
    print("=" * 60)
    print(f"❌ NO CONSENSUS after {MAX_EXCHANGES} exchanges.")
    print("=" * 60)

    print(f"\n🟢 {NAME_A}'s final answer:")
    print(latest[NAME_A])
    print(f"\n🟢 {NAME_A}'s reasoning for why its answer is better:")
    print(ask_model(MODEL_A, defense_prompt(NAME_A, question, latest[NAME_A], latest[NAME_B])))

    print(f"\n🔵 {NAME_B}'s final answer:")
    print(latest[NAME_B])
    print(f"\n🔵 {NAME_B}'s reasoning for why its answer is better:")
    print(ask_model(MODEL_B, defense_prompt(NAME_B, question, latest[NAME_B], latest[NAME_A])))


# --- Main Execution ---
if __name__ == "__main__":
    #question = "What is the most important quality for a software engineer to have, and why?"
    question = "Make the most optimised route to visit Stonehendge, Oxford, Cambridge, and London in a single day, starting from London. Provide the route in order of visit and the estimated time for each leg of the journey."
    run_debate(question)
