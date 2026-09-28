import ollama
import json
import re

# Real questions drawn from actual article/section topics — diverse, not repeats of the small set
CONSTITUTION_QUESTIONS = [
    "What does Article 21 of the Constitution protect?",
    "Explain the writ jurisdiction under Article 32.",
    "How does Article 246 distribute legislative powers?",
    "What are the Directive Principles of State Policy?",
    "What is the significance of the Preamble?",
    "What is Article 14 about equality before law?",
    "What powers does the President have under the Constitution?",
    "What are Fundamental Rights under Part III?",
    "Explain the concept of judicial review under the Constitution.",
    "What fundamental duties does a citizen have?",
    "How is the Union List different from the State List?",
    "What is Article 19 and what freedoms does it protect?",
    "How is the Supreme Court's jurisdiction defined?",
    "What is the procedure to amend the Constitution?",
    "What is the difference between Article 32 and Article 226?",
    "Explain the federal structure created by the Constitution.",
    "What is the role of the Governor under the Constitution?",
    "What is Article 15 about prohibition of discrimination?",
    "What is Article 17 about abolition of untouchability?",
    "What is the significance of Article 44 (Uniform Civil Code)?",
    "What is the composition of the Council of Ministers?",
    "What is the role of the Election Commission under the Constitution?",
    "What is Article 356 (President's Rule) about?",
    "What are the qualifications to become a Member of Parliament?",
    "What is the procedure for impeachment of the President?",
    "What is Article 51A about fundamental duties?",
    "How are High Court judges appointed?",
    "What is the significance of the 42nd Amendment?",
    "What is Article 370 and its historical significance?",
    "What is the concept of basic structure doctrine?",
    "What is Article 300A about the right to property?",
    "How does the Constitution define citizenship?",
    "What is the role of the Attorney General of India?",
    "What is Article 124 about the Supreme Court?",
    "What are the emergency provisions under the Constitution?",
    "What is the significance of Schedule 7?",
    "What is Article 39 about principles of policy?",
    "How does the Constitution protect minority rights?",
    "What is the process of forming a new state?",
    "What is the significance of the 73rd Amendment (Panchayati Raj)?",
]

CONSUMER_QUESTIONS = [
    "What are the six consumer rights under the Consumer Protection Act?",
    "What is the pecuniary jurisdiction of the District Commission?",
    "What is Product Liability under the 2019 Act?",
    "What is an unfair trade practice under consumer law?",
    "What is a misleading advertisement?",
    "What are the powers of the State Commission?",
    "What remedies are available to a consumer?",
    "What is meant by deficiency in service?",
    "What is the role of the Central Consumer Protection Authority?",
    "What is the jurisdiction of the National Commission?",
    "How does the Act define a defect in goods?",
    "What is the procedure for filing a consumer complaint?",
    "What penalties exist for false advertising?",
    "Who can file a complaint under the Consumer Protection Act?",
    "What is the composition of the State Council?",
    "What is e-commerce liability under the Act?",
    "What is a product liability action?",
    "What is mediation under the Consumer Protection Act?",
    "What is the time limit for filing a consumer complaint?",
    "What is the definition of 'consumer' under the Act?",
    "What is the definition of 'service' under the Act?",
    "What powers does the District Commission have to grant relief?",
    "What is the appeal process from District to State Commission?",
    "What is the appeal process from State to National Commission?",
    "What is a class action complaint under the Act?",
    "What is the penalty for non-compliance with a Commission order?",
    "What is the role of consumer protection councils?",
    "What is meant by 'spurious goods' under the Act?",
    "What is the liability of an e-commerce entity for a defective product?",
    "What is the concept of 'unfair contract' under the 2019 Act?",
    "What disclosures must e-commerce platforms make to consumers?",
    "What is the penalty for misleading advertisement by an endorser?",
    "What is the composition of the National Commission?",
    "What is a 'complainant' under the Consumer Protection Act?",
    "What is the significance of the Consumer Protection (E-Commerce) Rules?",
    "What is the process for mediation settlement under the Act?",
    "What is the role of the Director General under CCPA?",
    "What is 'restrictive trade practice' under consumer law?",
    "What is the punishment for manufacturing defective products causing injury?",
    "What is a 'design defect' under Product Liability?",
]

OUT_OF_DOMAIN_QUESTIONS = [
    "What is the statutory penalty for driving without a license?",
    "How do I file an income tax return?",
    "What is the punishment for theft under the IPC?",
    "What are the visa requirements for traveling to the US?",
    "How do I register a trademark in India?",
    "What is the process for divorce under the Hindu Marriage Act?",
    "What is the weather like today?",
    "How do I apply for a passport?",
    "What is the GST rate on electronics?",
    "Explain the rules of cricket.",
    "What is the best programming language to learn?",
    "How do I cook biryani?",
    "What is the penalty for cheque bounce under the Negotiable Instruments Act?",
    "What are the eligibility criteria for a home loan?",
    "How is company registration done under the Companies Act?",
    "What is the punishment for murder under the IPC?",
    "What is the procedure for bail under CrPC?",
    "What are the grounds for anticipatory bail?",
    "What is the Right to Information Act about?",
    "What does the Motor Vehicles Act say about drunk driving?",
    "What is the minimum wage law in India?",
    "What are the labor laws for maternity leave?",
    "What is the Arbitration and Conciliation Act?",
    "What is cyberbullying law in India?",
    "What is the Environmental Protection Act about?",
    "What are the rules under the Juvenile Justice Act?",
    "What is the Prevention of Corruption Act?",
    "How does copyright law protect authors?",
    "What is the Insolvency and Bankruptcy Code?",
    "What is the Real Estate Regulation Act (RERA)?",
    "What rights do I have if arrested by police?",
    "Can my landlord evict me without notice?",
    "What is the legal age for marriage in India?",
    "What is the punishment for cybercrime?",
    "Is triple talaq illegal in India?",
    "What is the SEBI Act about?",
    "What is the Foreign Exchange Management Act (FEMA)?",
    "What is the procedure for filing an FIR?",
    "What is the Right to Education Act about?",
    "What is the National Food Security Act?",
]

print(f"Total questions: {len(CONSTITUTION_QUESTIONS) + len(CONSUMER_QUESTIONS) + len(OUT_OF_DOMAIN_QUESTIONS)}")
print(f"  Constitution: {len(CONSTITUTION_QUESTIONS)}")
print(f"  Consumer Protection: {len(CONSUMER_QUESTIONS)}")
print(f"  Out of domain: {len(OUT_OF_DOMAIN_QUESTIONS)}")

all_data = []
for q in CONSTITUTION_QUESTIONS:
    all_data.append({"question": q, "label": "constitution", "expected": 1})
for q in CONSUMER_QUESTIONS:
    all_data.append({"question": q, "label": "consumer_protection", "expected": 1})
for q in OUT_OF_DOMAIN_QUESTIONS:
    all_data.append({"question": q, "label": "out_of_domain", "expected": 0})

with open("large_guardrail_test_set.json", "w") as f:
    json.dump(all_data, f, indent=2)

print(f"\nSaved {len(all_data)} questions to large_guardrail_test_set.json")
