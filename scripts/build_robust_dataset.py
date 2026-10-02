import json
import random
import os

# Semantic query variations per class
constitution_queries = [
    "What protection does Article 21 grant regarding personal liberty?",
    "Can a citizen file a writ petition under Article 226 in the High Court?",
    "How does the Basic Structure doctrine restrict constitutional amendments?",
    "What are the ground rules for declaring a national emergency under Article 352?",
    "Explain equality before law guaranteed under Article 14.",
    "What remedies exist for enforcement of Fundamental Rights under Article 32?",
    "What is the interplay between Directive Principles and Fundamental Rights?",
    "Under what circumstances can the President's rule be imposed under Article 356?",
    "What are the discretionary powers of a State Governor?",
    "How does Article 19 safeguard freedom of speech and expression?",
    "What constitutes a violation of freedom of religion under Article 25?",
    "How is the Preamble interpreted in assessing constitutional validity?",
    "What is the procedure for amending the Constitution under Article 368?",
    "Can fundamental rights be suspended during a state emergency?",
    "What legal protections exist against double jeopardy under Article 20?"
]

consumer_queries = [
    "How do I file a formal complaint against an e-commerce platform for defective items?",
    "What is the pecuniary jurisdiction of the District Consumer Disputes Redressal Commission?",
    "Can I claim product liability damages from a seller under CPA 2019?",
    "What action can be taken against misleading advertisements and false claims?",
    "What constitutes unfair trade practice in online retail sales?",
    "How many days do I have to request a refund for a faulty product?",
    "What are consumer rights under Section 2(9) of the Consumer Protection Act?",
    "How to appeal an order passed by the State Consumer Disputes Redressal Commission?",
    "Is a manufacturer liable for manufacturing defects causing personal injury?",
    "Can a service provider be sued for deficiency of service in medical care?",
    "What is the role of the Central Consumer Protection Authority (CCPA)?",
    "How can a consumer seek replacement or compensation for damaged goods?",
    "Does CPA 2019 cover unfair contracts imposed by builders?",
    "What are the penalties for non-compliance with Consumer Forum orders?",
    "Can a class action suit be filed on behalf of multiple affected consumers?"
]

ood_queries = [
    "What is the deadline for filing income tax returns for salaried individuals?",
    "How do I apply for EPF withdrawal online after leaving a job?",
    "What are the current GST rates for software development services?",
    "What is the penalty for driving without a valid driving license under the Motor Vehicles Act?",
    "How to report an online financial fraud complaint with the cyber crime police?",
    "What documents are required to register a Private Limited Company?",
    "How is gratuity calculated for employees completing five years of service?",
    "What are the compliance requirements under Section 138 of Negotiable Instruments Act for check bounce?",
    "How to file a patent application for a software invention in India?",
    "What are the tenant rights regarding rent increase under local rent control acts?"
]

# Modifiers to inject realistic phrasing variation
prefixes = [
    "In India, ", "Please clarify: ", "According to legal provisions, ", 
    "Is it true that ", "Under Indian law, ", "Can someone explain if ",
    "What are the exact guidelines when ", "How does the law handle "
]

suffixes = [
    " in current judicial practice?", " as per recent legal precedent?",
    " according to statutory provisions?", " when resolving a dispute?",
    " in a formal court setting?", " under applicable regulations?"
]

def generate_variations(base_list, domain, expected_source, count):
    records = []
    for i in range(count):
        base_text = random.choice(base_list)
        
        # Apply light structural variations
        p = random.choice([""] + prefixes)
        s = random.choice([""] + suffixes)
        
        query = f"{p}{base_text[0].lower() + base_text[1:] if p else base_text}"
        if s and not query.endswith("?"):
            query = f"{query.rstrip('.')} {s}"
            
        records.append({
            "id": f"{domain}_{i+1}",
            "query": query,
            "domain": domain,
            "expected_source": expected_source
        })
    return records

os.makedirs("data", exist_ok=True)

data = []
data.extend(generate_variations(constitution_queries, "constitution", "Constitution of India", 250))
data.extend(generate_variations(consumer_queries, "consumer_protection", "Consumer Protection Act, 2019", 250))
data.extend(generate_variations(ood_queries, "out_of_domain", "Out-of-Domain", 100))

random.shuffle(data)

with open("data/robust_dataset_600.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Dataset created with {len(data)} items at data/robust_dataset_600.json")
