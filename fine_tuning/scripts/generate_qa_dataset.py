import json
import random
import re
import time
from pathlib import Path

import ollama
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "fine_tuning" / "dataset"

MODEL = "gemma2:2b"
SEED = 42

TRAIN_LIMIT = 20
VALIDATION_LIMIT = 5
TEST_LIMIT = 5

random.seed(SEED)

QUESTION_TYPES = [
    "factual",
    "definition",
    "provision",
]


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_json(text):
    text = text.strip()

    # IMPORTANT: preserve whitespace inside question/answer strings.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Recover JSON if the model adds surrounding text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def clean_legal_text(text):
    # Repair common PDF extraction spacing problems.
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[a-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)

    # Repair common lowercase word joins created by PDF extraction.
    replacements = {
        'bymediation': 'by mediation',
        'ofreceive': 'of receive',
        'outof': 'out of',
        'anyexpenditure': 'any expenditure',
        'GoodsandServices': 'Goods and Services',
        'shallbe': 'shall be',
        'underthe': 'under the',
        'withthe': 'with the',
        'fromthe': 'from the',
        'intothe': 'into the',
        'ofthe': 'of the',
        'inthe': 'in the',
        'tothe': 'to the',
        'onthe': 'on the',
        'forthe': 'for the',
        'asthe': 'as the',
        'bythe': 'by the',
        'orany': 'or any',
        'canbe': 'can be',
        'maybe': 'may be',
        'shallhave': 'shall have',
        'shallnot': 'shall not',
        'doesnot': 'does not',
        'isnot': 'is not',
        'arethe': 'are the',
        'isthe': 'is the',
        'ofthe': 'of the',
        'andthe': 'and the',
        'withinthe': 'within the',
        'accordingto': 'according to',
        'punishmentfor': 'punishment for',
        'whatisthe': 'what is the',
        'whatarethe': 'what are the',
    }

    for old, new in replacements.items():
        text = re.sub(r'(?i)' + re.escape(old), new, text)

    # Normalize whitespace.
    text = re.sub(r'\s+', ' ', text)

    # Remove obvious page-footnote noise.
    text = re.sub(
        r'\s*_{5,}\s*',
        ' ',
        text
    )

    return text.strip()

def looks_like_bad_answer(answer):
    compact = answer.replace(' ', '')

    # Reject severe word collapsing.
    if len(answer.split()) < 4 and len(answer) > 80:
        return True

    # Reject outputs that appear to have lost normal word spacing.
    collapsed_markers = [
        'Theanswer', 'Themaximum', 'TheDistrict',
        'TheGovernor', 'Whatisthe', 'Underwhat',
        'Whatarethe', 'Whathappens', 'Accordingto',
        'inthe', 'ofthe', 'tothe', 'fromthe',
        'givingwritten', 'NationalCommission', 'StateCommission',
        'DistrictCommission', 'underthe', 'ofreceive', 'ofgoods',
        'orany', 'intothe', 'bythe', 'withthe', 'forthe',
        'onthe', 'fromthe', 'asthe', 'shallbe', 'canbe'
    ]

    if any(marker.lower() in answer.lower() for marker in collapsed_markers):
        return True

    # Reject obvious PDF-reference garbage.
    bad_patterns = [
        'Ins.by',
        'Subs.by',
        'w.e.f.',
        'Act,1969',
        'Act,1971',
        'FourthSch.',
        'EighthSch.',
        '________________',
    ]

    for pattern in bad_patterns:
        if pattern.lower() in answer.lower():
            return True

    # Reject answers that admit the passage does not contain the information.
    unsupported_markers = [
        'not explicitly stated',
        'not stated in the passage',
        'cannot be determined from the passage',
        'not provided in the passage',
        'information is not available',
        'not mentioned in the passage'
    ]

    if any(marker.lower() in answer.lower() for marker in unsupported_markers):
        return True

    # Reject suspiciously long generated answers.
    if len(answer) > 1800:
        return True

    # Reject vague or incomplete answers that do not provide the requested detail.
    incomplete_markers = [
        'such qualifications as may be prescribed',
        'such manner as may be prescribed',
        'as may be prescribed',
        'as prescribed',
        'provided in the relevant provision',
        'provided in the passage',
        'mentioned in the passage'
    ]

    answer_lower = answer.lower().strip()
    if any(marker in answer_lower for marker in incomplete_markers):
        if len(answer.split()) < 30:
            return True

    return False


def verify_and_normalize_qa(context, question, answer, evidence):
    prompt = f"""
You are validating one generated Indian legal QA example.

Use ONLY the supplied legal passage as evidence.

Your job is NOT to rewrite a valid answer unnecessarily. Accept the QA pair when the question can be answered from the passage and the answer is factually supported by the passage.

REJECT ONLY when there is a concrete problem such as:
- the answer contains a factual claim absent from the passage
- the answer invents an Article, section, clause, number, date, amount, penalty, authority, procedure, or legal consequence
- the question asks for information that the passage does not provide
- the answer completes missing text from a truncated passage
- the answer contradicts the passage
- the question and answer are unrelated to the passage

Allowed:
- obvious PDF word-spacing corrections
- normal paraphrasing that preserves the meaning
- removing unnecessary wording
- normal English spacing and punctuation

Do NOT reject merely because the answer is paraphrased instead of copied word-for-word.

Return ONLY valid JSON.

If valid:
{{"accept": true, "question": "...", "answer": "..."}}

If invalid:
{{"accept": false, "question": "", "answer": ""}}

LEGAL PASSAGE:
{context}

QUESTION:
{question}

ANSWER:
{answer}

EVIDENCE:
{evidence}

IMPORTANT:
- The evidence must directly support the answer.
- The answer must directly answer the exact question.
- Reject if the question asks for information not contained in the evidence.
- Reject if the answer contains information not supported by the evidence.
- Reject vague answers such as "provided in the passage" when the question asks for a specific fact.
"""

    try:
        r = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 300}
        )

        x = extract_json(r["message"]["content"])

        if not isinstance(x, dict):
            print("VERIFIER REJECT: invalid JSON/object")
            print("RAW:", r["message"]["content"])
            return None

        print("VERIFIER RESULT:", x)

        if x.get("accept") is not True:
            print("VERIFIER REJECT: model returned accept != true")
            return None

        q = clean_legal_text(str(x.get("question", "")))
        a = clean_legal_text(str(x.get("answer", "")))

        if not q or not a:
            return None

        if q.upper() == "REJECT" or a.upper() == "REJECT":
            return None

        if len(q) < 15 or len(a) < 20:
            return None

        # Deterministic evidence check: legal references in the answer
        # must also appear in the exact evidence supplied to the verifier.
        legal_refs = re.findall(r'\b(?:Article|Part|Chapter|section|clause)\s+[A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)?', a, re.I)
        evidence_lower = evidence.lower()
        for ref in legal_refs:
            if ref.lower() not in evidence_lower:
                print(f'VERIFIER REJECT: answer reference not supported by evidence: {ref}')
                return None

        if re.search(r'\bArticle\s+\d+', q, re.I):
            if not re.search(r'\bArticle\s+\d+', context, re.I):
                return None

        if re.search(r'\bsection\s+\d+', q, re.I):
            if not re.search(r'\bsection\s+\d+', context, re.I):
                return None

        if looks_like_bad_answer(a):
            return None

        return q, a

    except Exception as e:
        print("VERIFIER ERROR:", type(e).__name__, str(e))
        return None

def generate_example(passage):
    question_type = random.choice(QUESTION_TYPES)

    source = passage["source"]
    page = passage["page"]
    context = clean_legal_text(passage["context"])

    prompt = f"""
You are creating a high-quality supervised fine-tuning dataset for an Indian legal question-answering system.

Your task is to create EXACTLY ONE question-answer pair from ONLY the legal passage provided below.

STRICT EVIDENCE RULES:
1. Use ONLY information explicitly present in the supplied passage.
2. Do NOT use general legal knowledge, memory, or information from outside the passage.
3. Do NOT invent or infer an Article number, section number, schedule number, authority, date, penalty, amount, deadline, procedure, or legal consequence that is not explicitly stated in the passage.
4. The answer must be directly supported by one or more statements in the passage.
5. Do not combine unrelated provisions or entries merely because they occur in the same passage.
6. Do not interpret a numbered entry as an Article or Schedule unless the passage explicitly identifies it that way.
7. Do not use amendment notes, footnotes, page numbers, citation metadata, or OCR/PDF artifacts as evidence.
8. If the passage does not contain enough clear information to create a precise question and answer, return REJECT.
9. Preserve the exact legal identifier used in the passage. If the passage says section, use section; if it says Article, use Article; if it gives a numbered entry in a Schedule, do not call it an Article.
10. Never invent or change an Article number, section number, clause number, Schedule number, or entry number.
11. Never ask a question about an identifier unless that identifier is explicitly visible in the supplied passage.
12. If the passage contains only a fragment of a provision, ask only about facts explicitly contained in that fragment.

QUESTION REQUIREMENTS:
- Create one precise question about a specific fact, definition, provision, requirement, authority, procedure, condition, amount, period, or legal rule explicitly stated in the passage.
- The question must be answerable completely from the passage.
- IMPORTANT: The answer must directly answer the exact question asked. Do not create a question asking for a punishment, penalty, amount, date, period, qualification, authority, or procedure unless that exact information is explicitly stated in the passage.
- If the passage says that something is prohibited, protected, required, allowed, or not allowed, do not turn that statement into a question asking for a punishment unless the punishment is explicitly stated.
- Example of an invalid pair: passage says 'No person shall be punished for the same offence more than once', but question asks 'What is the punishment?'. Reject this type of question.
- Never use a phrase from the passage as an answer merely because it contains words related to the question. The answer must actually provide the requested information.
- Do not ask broad questions such as "What are the main points discussed in the passage?"
- Do not ask questions about information that is only implied.
- Do not ask about a provision that is not actually included in the supplied passage.
- If the passage contains only part of a provision, ask only about the part that is explicitly available.

ANSWER REQUIREMENTS:
- Answer the question directly and precisely.
- Use only facts explicitly supported by the passage.
- Preserve the legal meaning of the passage.
- Do not add explanations from outside knowledge.
- Do not copy amendment notes or footnotes unless they are necessary to answer the question.
- Use normal English spacing between every word.
- Do not concatenate words.
- Keep the answer concise but complete.

QUESTION TYPE:
{question_type}

OUTPUT FORMAT:
Return ONLY valid JSON and nothing else.

If the passage is insufficient, return exactly:
{{
  "question": "REJECT",
  "answer": "REJECT",
  "evidence": "REJECT"
}}

Otherwise return exactly:
{{
  "question": "...",
  "answer": "...",
  "evidence": "..."
}}

EVIDENCE REQUIREMENT:
- The evidence must be copied directly from the supplied legal passage.
- The evidence must contain the specific statement that supports the answer.
- Do not invent, paraphrase, or complete missing text in the evidence.
- If no exact supporting evidence exists, return REJECT.

Legal source:
{source}

Page:
{page}

Legal passage:
{context}
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Generate legal QA strictly from supplied evidence."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1,
                "num_predict": 350
            }
        )

        result = extract_json(response["message"]["content"])

        if not result:
            return None

        question = clean_legal_text(str(result.get("question", "")))
        answer = clean_legal_text(str(result.get("answer", "")))
        evidence = clean_legal_text(str(result.get("evidence", "")))

        if not evidence or evidence.upper() == "REJECT":
            print("REJECT: missing evidence")
            return None

        context_norm = re.sub(r"\s+", " ", context).strip().lower()
        evidence_norm = re.sub(r"\s+", " ", evidence).strip().lower()

        if evidence_norm not in context_norm:
            print("REJECT: evidence not found in passage")
            return None

        # Explicitly reject passages that the model could not answer reliably.
        if question.strip().upper() == "REJECT" or answer.strip().upper() == "REJECT":
            print("REJECT: model returned REJECT")
            return None

        generic_questions = {
            "answer the legal question using only the provided legal context.",
            "answer the legal question using only the provided context.",
            "answer the question using only the provided legal context.",
            "answer the question using only the provided context."
        }
        q_lower = question.strip().lower()
        if q_lower in generic_questions:
            print("REJECT: generic question")
            return None

        question_markers = ["what", "which", "who", "when", "where", "why", "how", "under what", "whether"]
        if not any(q_lower.startswith(marker + " ") or q_lower.startswith(marker + "?") for marker in question_markers):
            print("REJECT: question is not a real legal question")
            return None

        if len(question) < 15 or len(answer) < 20:
            print("REJECT: question or answer too short")
            return None

        # Do not apply answer-specific artifact checks to questions.

        # Reject questions that introduce unsupported legal identifiers.
        if re.search(r'\bArticle\s+\d+', question, flags=re.IGNORECASE):
            if not re.search(r'\bArticle\s+\d+', context, flags=re.IGNORECASE):
                return None

        if re.search(r'\bsection\s+\d+', question, flags=re.IGNORECASE):
            if not re.search(r'\bsection\s+\d+', context, flags=re.IGNORECASE):
                return None

        if looks_like_bad_answer(answer):
            print("Rejected low-quality answer")
            return None

        verified = verify_and_normalize_qa(context, question, answer, evidence)
        if not verified:
            print("Rejected by final legal evidence check")
            return None

        question, answer = verified

        return {
            "instruction": "Answer the legal question using only the provided legal context.",
            "input": f"Legal context:\n{context}\n\nQuestion:\n{question}",
            "output": answer,
            "evidence": evidence,
            "source": source,
            "page": page,
            "chunk_id": passage["chunk_id"],
            "question_type": question_type
        }

    except Exception as e:
        print(f"\nGeneration error: {e}")
        return None


def generate_split(input_file, output_file, limit):
    passages = load_jsonl(DATASET_DIR / input_file)

    # Shuffle passages deterministically.
    random.shuffle(passages)

    # CPU-friendly: inspect only a small number of passages.
    # This avoids spending hours scanning the entire corpus.
    max_attempts = min(len(passages), max(limit * 5, 25))
    candidates = passages[:max_attempts]

    generated = []
    attempted = 0

    print(f"\\nGenerating up to {limit} valid examples from {input_file}")
    print(f"Available passages: {len(passages)}")
    print(f"Maximum attempts: {max_attempts}")

    for passage in tqdm(candidates, desc=input_file):
        if len(generated) >= limit:
            break

        attempted += 1
        example = generate_example(passage)

        if example:
            generated.append(example)
            print(f"Accepted: {len(generated)}/{limit}")

        time.sleep(0.05)

    save_jsonl(DATASET_DIR / output_file, generated)

    print(f"Attempted passages : {attempted}")
    print(f"Created valid examples: {len(generated)}: {output_file}")

    return generated

def main():
    print("=" * 60)
    print("LEGAL INSTRUCTION DATASET GENERATOR")
    print("=" * 60)
    print(f"Model: {MODEL}")

    train = generate_split(
        "train_passages.jsonl",
        "train.jsonl",
        TRAIN_LIMIT
    )

    validation = generate_split(
        "validation_passages.jsonl",
        "validation.jsonl",
        VALIDATION_LIMIT
    )

    test = generate_split(
        "test_passages.jsonl",
        "test.jsonl",
        TEST_LIMIT
    )

    metadata = {
        "model_used_for_generation": MODEL,
        "seed": SEED,
        "actual_examples": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test)
        },
        "question_types": QUESTION_TYPES
    }

    with open(
        DATASET_DIR / "qa_generation_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Training examples      : {len(train)}")
    print(f"Validation examples   : {len(validation)}")
    print(f"Test examples         : {len(test)}")


if __name__ == "__main__":
    main()
