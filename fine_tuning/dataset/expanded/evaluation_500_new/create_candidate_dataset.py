import csv
from pathlib import Path

OUT = Path(__file__).parent / "legal_evaluation_500_candidates.csv"

if OUT.exists():
    raise SystemExit(
        f"STOP: {OUT.name} already exists. Nothing was overwritten."
    )

constitution_topics = [
    "equality before the law", "prohibition of discrimination",
    "equality of opportunity in public employment", "abolition of untouchability",
    "abolition of titles", "freedom of speech and expression",
    "protection of life and personal liberty", "protection against arbitrary arrest",
    "right to education", "constitutional remedies and writs",
    "freedom of religion", "cultural and educational rights",
    "protection of minority educational institutions", "fundamental duties",
    "Directive Principles of State Policy", "uniform civil code",
    "organisation of village panchayats", "right to work and public assistance",
    "protection of the environment", "separation of judiciary and executive",
    "composition of Parliament", "powers of the President",
    "appointment of the Prime Minister", "powers of the Governor",
    "State legislative assemblies", "Supreme Court jurisdiction",
    "High Court jurisdiction", "judicial review",
    "constitutional amendment procedure", "Finance Commission",
    "Election Commission", "Comptroller and Auditor General",
    "Union and State legislative powers", "Union List",
    "State List", "Concurrent List", "constitutional emergency provisions",
    "President's Rule", "financial emergency", "official language provisions",
]

consumer_topics = [
    "the meaning of a consumer", "consumer rights",
    "consumer dispute redressal", "defects in goods",
    "deficiency in services", "unfair trade practices",
    "restrictive trade practices", "product liability",
    "unfair contracts", "misleading advertisements",
    "the Central Consumer Protection Authority", "consumer mediation",
    "District Consumer Commission", "State Consumer Commission",
    "National Consumer Commission", "consumer complaints",
    "filing a consumer complaint electronically", "complaints against e-commerce sellers",
    "product liability of manufacturers", "product liability of sellers",
    "product liability of service providers", "consumer protection in online shopping",
    "rights of consumers in transactions", "compensation for consumer harm",
    "replacement of defective goods", "refunds for defective goods",
    "removal of defects in goods", "discontinuation of unfair practices",
    "consumer commissions and their jurisdiction", "appeals in consumer disputes",
    "mediation cells in consumer disputes", "misleading product claims",
    "endorsements in advertisements", "unfair pricing practices",
    "consumer protection rules for e-commerce", "responsibilities of product sellers",
    "consumer dispute remedies", "complaints involving services",
    "consumer protection in digital transactions", "the objectives of the 2019 Act",
]

ood_topics = [
    "binary search trees", "database normalization", "Python list slicing",
    "operating system scheduling", "computer network routing",
    "machine learning overfitting", "gradient descent",
    "neural network activation functions", "cloud computing architecture",
    "version control with Git", "HTML form validation",
    "SQL joins", "computer monitor brightness", "keyboard shortcuts",
    "photosynthesis", "ocean tides", "planetary motion",
    "water cycle", "electric vehicle batteries", "weather forecasting",
]

constitution_templates = [
    "What does {topic} mean under the Constitution of India?",
    "Which constitutional provisions address {topic}?",
    "Explain the constitutional framework concerning {topic}.",
    "How does the Constitution of India deal with {topic}?",
    "What should a student know about {topic} in the Constitution?",
]

consumer_templates = [
    "What does the Consumer Protection Act, 2019 say about {topic}?",
    "Which provisions of the 2019 Act address {topic}?",
    "Explain {topic} under the Consumer Protection Act, 2019.",
    "How is {topic} handled under the 2019 consumer law?",
    "What should a consumer know about {topic} under the 2019 Act?",
]

ood_templates = [
    "What is the basic concept of {topic}?",
    "Can you explain how {topic} works?",
    "What are the main principles behind {topic}?",
    "Why is {topic} used or important?",
    "What should a beginner know about {topic}?",
]

rows = []
counter = 1

def add_group(topics, templates, label, source):
    global counter
    for topic in topics:
        for template in templates:
            rows.append({
                "ID": f"NEW500-{counter:03d}",
                "Question": template.format(topic=topic),
                "Expected": label,
                "Expected_Source": source,
                "Predicted": "",
                "Correct": "",
                "Expected_Answer_Reference": "",
                "review_status": "pending_manual_review",
                "candidate_origin": "synthetic_template_candidate",
                "split": "candidate_not_yet_approved",
            })
            counter += 1

add_group(
    constitution_topics, constitution_templates,
    "constitution", "constitution_of_india.pdf"
)
add_group(
    consumer_topics, consumer_templates,
    "consumer_protection", "consumer_protection_act_2019.pdf"
)
add_group(
    ood_topics, ood_templates,
    "out_of_domain", ""
)

assert len(rows) == 500, f"Expected 500 rows, got {len(rows)}"
assert len({r["Question"] for r in rows}) == 500, "Duplicate questions detected"

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("x", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Created:", OUT)
print("Total questions:", len(rows))
for label in ("constitution", "consumer_protection", "out_of_domain"):
    print(f"{label}: {sum(r['Expected'] == label for r in rows)}")
print("Review status: all pending_manual_review")
print("Note: these are synthetic candidates, not a verified benchmark.")
