import json, random, os
constitution_topics = ['Fundamental Rights Article 14 to 21', 'Writs Article 32 and 226', 'Directive Principles of State Policy', 'Amendments Article 368', 'Emergency Provisions Article 352', 'President and Governor Powers', 'Preamble and Basic Structure']
consumer_topics = ['Defective products CPA 2019', 'Unfair trade practices e-commerce', 'District State National Commissions', 'Consumer Rights Section 2(9)', 'Product Liability claims', 'Misleading advertisements penalties', 'Refunds and replacements']
ood_topics = ['Employees Provident Fund EPF rules', 'GST tax filing deadlines', 'Income Tax return computation', 'Traffic violation fines Motor Vehicles Act', 'Cybercrime complaints IT Act', 'Company registration Companies Act']
def generate_samples(topics, domain, count):
    samples = []
    templates = ['What are the provisions regarding {topic}?', 'Explain the legal rules governing {topic}.', 'Can a citizen file a claim based on {topic}?', 'What does the law specify about {topic}?']
    for i in range(count):
        topic = random.choice(topics)
        template = random.choice(templates)
        samples.append({'id': f'{domain}_{i+1}', 'query': template.format(topic=topic), 'domain': domain, 'expected_source': 'Constitution' if domain == 'constitution' else ('Consumer Protection Act, 2019' if domain == 'consumer_protection' else 'Out-of-Domain')})
    return samples
os.makedirs('data', exist_ok=True)
data = []
data.extend(generate_samples(constitution_topics, 'constitution', 400))
data.extend(generate_samples(consumer_topics, 'consumer_protection', 400))
data.extend(generate_samples(ood_topics, 'out_of_domain', 200))
random.shuffle(data)
with open('data/dataset_1000.json', 'w') as f: json.dump(data, f, indent=2)
print(f'Successfully generated dataset with {len(data)} items at data/dataset_1000.json')
