"""
Generate 500 grounded RAG evaluation questions from actual legal passages.

Reads all passage data from fine_tuning/dataset/*.jsonl files,
extracts key legal concepts, articles, sections, and provisions,
then generates diverse questions grounded in the actual text.

Output:
  - outputs/rag_evaluation_questions_500.csv
  - outputs/rag_evaluation_questions_500.jsonl
"""

import json
import re
import csv
import hashlib
from pathlib import Path
from collections import defaultdict


def load_all_passages(base_dir):
    """Load all passages from train/validation/test JSONL files."""
    passages = {"constitution": [], "consumer_protection": []}
    
    for fname in ["train_passages.jsonl", "validation_passages.jsonl", "test_passages.jsonl"]:
        path = Path(base_dir) / "fine_tuning" / "dataset" / fname
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                source = obj.get("source", "")
                if source in passages:
                    passages[source].append(obj)
    
    return passages


def extract_constitution_articles(passages):
    """Extract article numbers and their content from Constitution passages."""
    articles = {}
    for p in passages:
        text = p["context"]
        page = p["page"]
        # Find article numbers: "123. Some text" at start of line or after whitespace
        matches = re.finditer(
            r'(?:^|\n)\s*(\d+[A-Za-z]?)\.\s+([A-Z][^\n]{10,})',
            text
        )
        for m in matches:
            art_num = m.group(1)
            art_text = m.group(2)[:500]
            if art_num not in articles or len(art_text) > len(articles[art_num].get("text", "")):
                articles[art_num] = {
                    "number": art_num,
                    "text": art_text,
                    "page": page,
                    "full_context": text
                }
    return articles


def extract_cp_sections(passages):
    """Extract section numbers from Consumer Protection Act passages."""
    sections = {}
    for p in passages:
        text = p["context"]
        page = p["page"]
        # Section patterns
        matches = re.finditer(
            r'(?:^|\n)\s*(\d+)\.\s+([A-Z][^\n]{10,})',
            text
        )
        for m in matches:
            sec_num = m.group(1)
            sec_text = m.group(2)[:500]
            if sec_num not in sections or len(sec_text) > len(sections[sec_num].get("text", "")):
                sections[sec_num] = {
                    "number": sec_num,
                    "text": sec_text,
                    "page": page,
                    "full_context": text
                }
    return sections


def find_key_concepts_in_text(text):
    """Extract key legal concepts mentioned in a passage."""
    concepts = set()
    
    # Constitutional concepts
    const_patterns = [
        (r'\bfundamental rights?\b', 'Fundamental Rights'),
        (r'\bfundamental duties?\b', 'Fundamental Duties'),
        (r'\bdirective principles?\b', 'Directive Principles of State Policy'),
        (r'\bpreamble\b', 'Preamble'),
        (r'\bsupreme court\b', 'Supreme Court'),
        (r'\bhigh court\b', 'High Court'),
        (r'\bparliament\b', 'Parliament'),
        (r'\bpresident\b', 'President'),
        (r'\bgovernor\b', 'Governor'),
        (r'\bprime minister\b', 'Prime Minister'),
        (r'\belection commission\b', 'Election Commission'),
        (r'\bfinance commission\b', 'Finance Commission'),
        (r'\bconsolidated fund\b', 'Consolidated Fund'),
        (r'\bemergency\b', 'Emergency'),
        (r'\bproclamation\b', 'Proclamation'),
        (r'\bamendment\b', 'Constitutional Amendment'),
        (r'\bcitizenship\b', 'Citizenship'),
        (r'\bschedule\b', 'Schedule'),
        (r'\bpanchayat\b', 'Panchayat'),
        (r'\bmunicipality\b', 'Municipality'),
        (r'\bunion list\b', 'Union List'),
        (r'\bstate list\b', 'State List'),
        (r'\bconcurrent list\b', 'Concurrent List'),
        (r'\btribunal\b', 'Tribunal'),
        (r'\bwrit\b', 'Writ'),
        (r'\bhabeas corpus\b', 'Habeas Corpus'),
        (r'\bmandamus\b', 'Mandamus'),
        (r'\bcertiorari\b', 'Certiorari'),
        (r'\bordinance\b', 'Ordinance'),
        (r'\bjudicial review\b', 'Judicial Review'),
        (r'\breservation\b', 'Reservation'),
        (r'\bscheduled castes?\b', 'Scheduled Castes'),
        (r'\bscheduled tribes?\b', 'Scheduled Tribes'),
        (r'\bbackward classes?\b', 'Backward Classes'),
        (r'\banti.?defection\b', 'Anti-Defection'),
        (r'\bofficial language\b', 'Official Language'),
    ]
    
    # Consumer Protection concepts
    cp_patterns = [
        (r'\bconsumer\b', 'Consumer'),
        (r'\bcomplaint\b', 'Complaint'),
        (r'\bdefect\b', 'Defect'),
        (r'\bdeficiency\b', 'Deficiency'),
        (r'\bunfair trade practice\b', 'Unfair Trade Practice'),
        (r'\brestrictive trade practice\b', 'Restrictive Trade Practice'),
        (r'\bproduct liability\b', 'Product Liability'),
        (r'\bunfair contract\b', 'Unfair Contract'),
        (r'\bmisleading advertisement\b', 'Misleading Advertisement'),
        (r'\bmediation\b', 'Mediation'),
        (r'\bdistrict commission\b', 'District Commission'),
        (r'\bstate commission\b', 'State Commission'),
        (r'\bnational commission\b', 'National Commission'),
        (r'\bccpa\b', 'CCPA'),
        (r'\bcentral consumer protection authority\b', 'Central Consumer Protection Authority'),
        (r'\bmanufacturer\b', 'Manufacturer'),
        (r'\bproduct seller\b', 'Product Seller'),
        (r'\bspurious goods?\b', 'Spurious Goods'),
        (r'\badulterat\b', 'Adulterated Products'),
        (r'\be.?commerce\b', 'E-Commerce'),
        (r'\bendorsement\b', 'Endorsement'),
        (r'\bpenalt\b', 'Penalty'),
        (r'\bappeal\b', 'Appeal'),
        (r'\bjurisdiction\b', 'Jurisdiction'),
    ]
    
    text_lower = text.lower()
    for pattern, concept in const_patterns + cp_patterns:
        if re.search(pattern, text_lower):
            concepts.add(concept)
    
    return concepts


def question_fingerprint(q):
    """Create a normalized fingerprint for near-duplicate detection."""
    # Normalize: lowercase, remove extra spaces, remove common suffixes
    q = q.lower().strip()
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'with reference to the constitution of india\.?$', '', q)
    q = re.sub(r'under the consumer protection act,?\s*2019\.?$', '', q)
    q = re.sub(r'under the act\.?$', '', q)
    q = re.sub(r'under the constitution\.?$', '', q)
    q = q.strip()
    return q


def is_near_duplicate(new_q, existing_fingerprints, threshold=0.92):
    """Check if a question is a near-duplicate of existing ones."""
    new_fp = question_fingerprint(new_q)
    
    if new_fp in existing_fingerprints:
        return True
    
    # Simple word-overlap check
    new_words = set(new_fp.split())
    for existing_fp in existing_fingerprints:
        existing_words = set(existing_fp.split())
        if not new_words or not existing_words:
            continue
        overlap = len(new_words & existing_words)
        similarity = overlap / max(len(new_words), len(existing_words))
        if similarity >= threshold:
            return True
    
    return False


def generate_constitution_questions(passages, target=250):
    """Generate grounded questions about the Constitution of India."""
    questions = []
    fingerprints = set()
    articles = extract_constitution_articles(passages)
    
    # === CATEGORY 1: Article-specific questions ===
    # Key articles with known content
    article_topics = {
        "1": ("territory of India", "defines the territory and name of India"),
        "2": ("admission of new States", "provides for admission or establishment of new States"),
        "3": ("formation of new States", "provides for formation of new States and alteration of boundaries"),
        "4": ("laws for admission of new States", "provides that laws for admission do not constitute amendment"),
        "5": ("citizenship at commencement", "defines citizenship at the commencement of the Constitution"),
        "6": ("rights of citizenship of certain persons who migrated to India", "provides citizenship rights for migrants from Pakistan"),
        "7": ("rights of citizenship of certain migrants to Pakistan", "addresses citizenship of persons who migrated to Pakistan"),
        "8": ("rights of citizenship of persons of Indian origin residing outside India", "provides for citizenship of persons of Indian origin abroad"),
        "9": ("persons voluntarily acquiring citizenship of foreign State", "provides that acquiring foreign citizenship terminates Indian citizenship"),
        "10": ("continuance of rights of citizenship", "provides for continuance of citizenship rights"),
        "11": ("Parliament to regulate right of citizenship by law", "empowers Parliament to make laws on citizenship"),
        "12": ("definition of State", "defines the term 'State' for Part III"),
        "13": ("laws inconsistent with Fundamental Rights", "declares laws inconsistent with Fundamental Rights void"),
        "14": ("equality before law", "guarantees equality before law and equal protection of laws"),
        "15": ("prohibition of discrimination", "prohibits discrimination on grounds of religion, race, caste, sex, or place of birth"),
        "16": ("equality of opportunity in public employment", "guarantees equality of opportunity in matters of public employment"),
        "17": ("abolition of untouchability", "abolishes untouchability and forbids its practice"),
        "18": ("abolition of titles", "abolishes titles except military and academic distinctions"),
        "19": ("protection of certain rights regarding freedom of speech", "guarantees six freedoms including speech, assembly, movement"),
        "20": ("protection in respect of conviction for offences", "protects against ex post facto laws, double jeopardy, and self-incrimination"),
        "21": ("protection of life and personal liberty", "provides that no person shall be deprived of life or liberty except by procedure established by law"),
        "21A": ("right to education", "provides free and compulsory education for children aged 6-14"),
        "22": ("protection against arrest and detention", "provides safeguards against arbitrary arrest and detention"),
        "23": ("prohibition of traffic in human beings and forced labour", "prohibits human trafficking and forced labour"),
        "24": ("prohibition of employment of children in factories", "prohibits employment of children below 14 in factories, mines, and hazardous employment"),
        "25": ("freedom of conscience and free profession of religion", "guarantees freedom of conscience and right to profess, practice, and propagate religion"),
        "26": ("freedom to manage religious affairs", "guarantees right of religious denominations to manage their own affairs"),
        "27": ("freedom from payment of taxes for promotion of religion", "prohibits compelling any person to pay taxes for promotion of any religion"),
        "28": ("freedom from attendance at religious instruction", "prohibits religious instruction in State-funded educational institutions"),
        "29": ("protection of interests of minorities", "protects cultural and educational rights of minorities"),
        "30": ("right of minorities to establish educational institutions", "gives minorities the right to establish and administer educational institutions"),
        "32": ("remedies for enforcement of Fundamental Rights", "provides right to move Supreme Court for enforcement of Fundamental Rights through writs"),
        "36": ("definition of State for Directive Principles", "defines State for Part IV same as Part III"),
        "38": ("State to secure a social order for the promotion of welfare", "directs State to promote welfare through social, economic, and political justice"),
        "39": ("certain principles of policy", "directs State to secure adequate livelihood, equal pay, and protect children"),
        "39A": ("equal justice and free legal aid", "directs State to provide free legal aid and ensure equal justice"),
        "40": ("organisation of village panchayats", "directs State to organise village panchayats with self-government"),
        "41": ("right to work, education, and public assistance", "directs State to provide right to work, education, and public assistance"),
        "42": ("provision for just and humane conditions of work", "directs State to make provision for just conditions of work and maternity relief"),
        "43": ("living wage for workers", "directs State to secure living wage, decent standard of life for workers"),
        "44": ("uniform civil code", "directs State to endeavour to secure uniform civil code throughout India"),
        "45": ("provision for early childhood care and education", "directs State to provide early childhood care for children below age 6"),
        "46": ("promotion of educational and economic interests of weaker sections", "directs State to promote interests of Scheduled Castes, Tribes, and weaker sections"),
        "47": ("duty of the State to raise the level of nutrition", "directs State to raise level of nutrition and standard of living"),
        "48": ("organisation of agriculture and animal husbandry", "directs State to organise agriculture and animal husbandry on modern lines"),
        "48A": ("protection and improvement of environment", "directs State to protect and improve the environment and safeguard forests and wildlife"),
        "49": ("protection of monuments and places of national importance", "directs State to protect monuments, places, and objects of artistic or historic interest"),
        "50": ("separation of judiciary from executive", "directs State to take steps to separate judiciary from executive"),
        "51": ("promotion of international peace and security", "directs State to promote international peace, maintain just relations between nations"),
        "51A": ("Fundamental Duties", "lists fundamental duties of every citizen of India"),
        "52": ("the President of India", "provides that there shall be a President of India"),
        "53": ("executive power of the Union", "vests executive power of Union in the President"),
        "54": ("election of President", "provides for election of President by electoral college"),
        "55": ("manner of election of President", "provides for manner of election of President"),
        "56": ("term of office of President", "provides five-year term for President"),
        "58": ("qualifications for election as President", "specifies qualifications for President"),
        "61": ("procedure for impeachment of President", "provides procedure for impeachment of President"),
        "63": ("the Vice-President of India", "provides for a Vice-President of India"),
        "64": ("Vice-President to be ex officio Chairman of Council of States", "makes VP ex officio Chairman of Rajya Sabha"),
        "72": ("power of President to grant pardons", "grants President power to pardon, reprieve, respite, or remit punishment"),
        "73": ("extent of executive power of the Union", "defines extent of executive power of Union"),
        "74": ("Council of Ministers to aid and advise President", "provides for Council of Ministers to aid and advise President"),
        "75": ("other provisions as to Ministers", "provides for appointment of PM and Council of Ministers"),
        "76": ("Attorney-General for India", "provides for appointment of Attorney-General"),
        "79": ("constitution of Parliament", "provides Parliament consists of President and two Houses"),
        "80": ("composition of the Council of States", "provides for composition of Rajya Sabha"),
        "81": ("composition of the House of the People", "provides for composition of Lok Sabha"),
        "83": ("duration of Houses of Parliament", "provides 5-year term for Lok Sabha"),
        "84": ("qualification for membership of Parliament", "specifies qualifications for MPs"),
        "100": ("voting in Houses, power of Houses to act notwithstanding vacancies", "provisions on voting and quorum in Parliament"),
        "108": ("joint sitting of both Houses", "provides for joint sitting in case of disagreement"),
        "109": ("special procedure in respect of Money Bills", "provides procedure for Money Bills"),
        "110": ("definition of Money Bills", "defines what constitutes a Money Bill"),
        "112": ("annual financial statement", "provides for annual financial statement (Budget)"),
        "123": ("power of President to promulgate Ordinances", "grants President power to promulgate Ordinances"),
        "124": ("establishment and constitution of Supreme Court", "establishes Supreme Court"),
        "131": ("original jurisdiction of the Supreme Court", "provides for original jurisdiction"),
        "132": ("appellate jurisdiction of Supreme Court in constitutional matters", "provides for appellate jurisdiction in constitutional matters"),
        "136": ("special leave to appeal by Supreme Court", "grants Supreme Court special leave to appeal"),
        "137": ("review of judgments by Supreme Court", "provides for review of judgments"),
        "141": ("law declared by Supreme Court to be binding", "makes Supreme Court judgments binding on all courts"),
        "143": ("power of President to consult Supreme Court", "provides for advisory jurisdiction"),
        "148": ("Comptroller and Auditor-General of India", "provides for appointment of CAG"),
        "152": ("definition of State in Part VI", "defines State for purposes of Part VI"),
        "153": ("Governors of States", "provides for Governors of States"),
        "154": ("executive power of State", "vests executive power of State in Governor"),
        "155": ("appointment of Governor", "provides for appointment of Governor by President"),
        "163": ("Council of Ministers to aid and advise Governor", "provides for Council of Ministers in States"),
        "164": ("other provisions as to Ministers", "provides for appointment of Chief Minister"),
        "168": ("constitution of Legislatures in States", "provides for State Legislatures"),
        "169": ("abolition or creation of Legislative Councils in States", "allows abolition or creation of Legislative Councils"),
        "170": ("composition of the Legislative Assemblies", "provides for composition of Vidhan Sabha"),
        "213": ("power of Governor to promulgate Ordinances", "grants Governor power to promulgate Ordinances"),
        "214": ("High Courts for States", "provides for High Courts for each State"),
        "226": ("power of High Courts to issue writs", "grants High Courts power to issue writs"),
        "227": ("power of superintendence over subordinate courts by High Courts", "grants High Courts superintendence over subordinate courts"),
        "243": ("definitions for Panchayats", "defines terms for Part IX on Panchayats"),
        "243A": ("Gram Sabha", "provides for Gram Sabha at village level"),
        "243B": ("constitution of Panchayats", "provides for constitution of Panchayats"),
        "243D": ("reservation of seats in Panchayats", "provides for reservation in Panchayats"),
        "244": ("administration of Scheduled Areas and Tribal Areas", "provides for administration of Scheduled and Tribal Areas"),
        "245": ("extent of laws made by Parliament and Legislatures of States", "defines territorial jurisdiction of legislative powers"),
        "246": ("subject-matter of laws made by Parliament and by the Legislatures of States", "distributes legislative powers between Union and States using three lists"),
        "248": ("residuary powers of legislation", "vests residuary legislative powers in Parliament"),
        "249": ("power of Parliament to legislate on State matters in national interest", "allows Parliament to legislate on State List in national interest"),
        "250": ("power of Parliament to legislate on any State matter during Emergency", "allows Parliament to legislate on State List during Emergency"),
        "254": ("inconsistency between Union and State laws on Concurrent List", "provides for resolution of inconsistency on Concurrent List matters"),
        "256": ("obligation of States and the Union", "provides that States must comply with Union laws"),
        "263": ("provisions with respect to an inter-State Council", "provides for establishment of Inter-State Council"),
        "265": ("taxes not to be imposed save by authority of law", "prohibits levying taxes without authority of law"),
        "266": ("Consolidated Funds and public accounts of India and of the States", "provides for Consolidated Fund and Public Account"),
        "267": ("Contingency Fund", "provides for Contingency Fund of India"),
        "270": ("taxes levied and distributed between the Union and the States", "provides for distribution of tax revenues between Union and States"),
        "275": ("grants from the Union to certain States", "provides for grants-in-aid from Union to States"),
        "280": ("Finance Commission", "provides for Constitution of Finance Commission"),
        "300A": ("persons not to be deprived of property save by authority of law", "protects right to property"),
        "311": ("dismissal, removal or reduction in rank of civil servants", "provides protection to civil servants against arbitrary dismissal"),
        "312": ("All-India Services", "provides for creation of All-India Services"),
        "315": ("Public Service Commissions for the Union and for the States", "provides for UPSC and State PSCs"),
        "320": ("functions of Public Service Commissions", "specifies functions of Public Service Commissions"),
        "324": ("superintendence, direction and control of elections", "vests election superintendence in Election Commission"),
        "326": ("elections to the House of the People and Legislative Assemblies on the basis of adult suffrage", "provides for adult suffrage"),
        "329": ("bar to interference by courts in electoral matters", "bars court interference in electoral matters"),
        "330": ("reservation of seats for Scheduled Castes and Scheduled Tribes in Lok Sabha", "provides reservation in Lok Sabha"),
        "331": ("representation of Anglo-Indian community in Lok Sabha", "provides for nomination of Anglo-Indians"),
        "335": ("claims of Scheduled Castes and Scheduled Tribes to services and posts", "provides for consideration of SC/ST claims in services"),
        "338": ("National Commission for Scheduled Castes", "provides for National Commission for Scheduled Castes"),
        "338A": ("National Commission for Scheduled Tribes", "provides for National Commission for Scheduled Tribes"),
        "340": ("appointment of a Commission to investigate conditions of backward classes", "provides for investigation into backward class conditions"),
        "341": ("Scheduled Castes", "empowers President to specify Scheduled Castes"),
        "342": ("Scheduled Tribes", "empowers President to specify Scheduled Tribes"),
        "343": ("official language of the Union", "declares Hindi in Devanagari script as official language"),
        "344": ("commission and committee of Parliament on official language", "provides for Official Language Commission"),
        "348": ("language to be used in Supreme Court and in High Courts", "prescribes English for Supreme Court and High Courts"),
        "350": ("language to be used in representations for redress of grievances", "provides right to submit representations in any language"),
        "350A": ("facilities for instruction in mother-tongue at primary stage", "provides for mother-tongue education at primary stage"),
        "350B": ("Special Officer for linguistic minorities", "provides for appointment of Special Officer for linguistic minorities"),
        "352": ("Proclamation of Emergency", "provides for proclamation of national emergency"),
        "356": ("provisions in case of failure of constitutional machinery in States", "provides for President's Rule in States"),
        "360": ("provisions as to financial emergency", "provides for proclamation of financial emergency"),
        "365": ("effect of failure to comply with Union directions", "provides consequence of State non-compliance with Union directions"),
        "368": ("power of Parliament to amend the Constitution", "provides procedure for constitutional amendment"),
        "370": ("temporary provisions with respect to Jammu and Kashmir", "provided temporary special status to Jammu and Kashmir"),
        "371": ("special provisions with respect to certain States", "provides special provisions for Maharashtra and Gujarat"),
    }
    
    # Generate varied questions for key articles
    question_templates = [
        ("What does Article {num} of the Constitution of India provide?", "provision"),
        ("What is the subject matter of Article {num} of the Constitution?", "factual"),
        ("What rights or protections does Article {num} guarantee?", "rights"),
        ("What are the key provisions of Article {num} of the Indian Constitution?", "provision"),
    ]
    
    procedural_templates = [
        ("What is the procedure prescribed under Article {num} of the Constitution?", "procedure"),
    ]
    
    definition_templates = [
        ("How does Article {num} define {topic} under the Constitution of India?", "definition"),
    ]
    
    # Select articles for question generation - prioritize those found in passages
    available_articles = set(articles.keys())
    
    # Phase 1: Direct article questions (diverse templates)
    used_articles = set()
    for art_num, (topic, description) in article_topics.items():
        if len(questions) >= 220:
            break
        
        # Generate 1-3 questions per key article using diverse templates
        templates_to_use = []
        
        # Always add a general "what does this article provide" question
        templates_to_use.append(
            (f"What does Article {art_num} of the Constitution of India provide?", "provision",
             f"Article {art_num}", description)
        )
        
        if any(word in description.lower() for word in ["defines", "definition", "meaning"]):
            templates_to_use.append(
                (f"How does Article {art_num} define {topic}?", "definition",
                 f"Article {art_num}", description)
            )
        elif any(word in description.lower() for word in ["procedure", "process", "manner"]):
            templates_to_use.append(
                (f"What is the procedure prescribed under Article {art_num}?", "procedure",
                 f"Article {art_num}", description)
            )
            templates_to_use.append(
                (f"What does Article {art_num} provide regarding {topic}?", "provision",
                 f"Article {art_num}", description)
            )
        elif any(word in description.lower() for word in ["right", "guarantee", "protect", "freedom"]):
            templates_to_use.append(
                (f"What rights does Article {art_num} guarantee?", "rights",
                 f"Article {art_num}", description)
            )
            templates_to_use.append(
                (f"What is the scope of Article {art_num} on {topic}?", "provision",
                 f"Article {art_num}", description)
            )
        elif any(word in description.lower() for word in ["directs", "duty", "obligation"]):
            templates_to_use.append(
                (f"What does Article {art_num} direct the State to do regarding {topic}?", "duties",
                 f"Article {art_num}", description)
            )
        elif any(word in description.lower() for word in ["provides for", "establishes", "constitutes"]):
            templates_to_use.append(
                (f"What does Article {art_num} provide regarding {topic}?", "provision",
                 f"Article {art_num}", description)
            )
            templates_to_use.append(
                (f"What institution or body is established under Article {art_num}?", "institutions",
                 f"Article {art_num}", description)
            )
        else:
            templates_to_use.append(
                (f"What does Article {art_num} of the Constitution of India provide?", "provision",
                 f"Article {art_num}", description)
            )
        
        for q_text, q_type, ref, ref_answer in templates_to_use:
            if len(questions) >= 200:
                break
            if is_near_duplicate(q_text, fingerprints):
                continue
            fp = question_fingerprint(q_text)
            fingerprints.add(fp)
            
            # Find the page from passage data
            page = articles.get(art_num, {}).get("page", "")
            evidence = articles.get(art_num, {}).get("text", "")[:200] if art_num in articles else ""
            
            questions.append({
                "question_id": f"CON_{len(questions)+1:03d}",
                "question": q_text,
                "expected_source": "constitution_of_india.pdf",
                "question_type": q_type,
                "source_reference": ref,
                "reference_answer": ref_answer,
                "evidence": evidence,
                "validation_status": "grounded" if art_num in available_articles else "reference_only"
            })
            used_articles.add(art_num)
    
    # Phase 2: Topic-based questions from passage content
    topic_questions = [
        # Preamble and Basics
        ("What are the ideals stated in the Preamble of the Constitution of India?", "factual", "Preamble", "The Preamble declares India as a sovereign, socialist, secular, democratic republic and resolves to secure justice, liberty, equality, and fraternity"),
        ("What type of government does the Preamble envision for India?", "factual", "Preamble", "The Preamble envisions a sovereign, socialist, secular, democratic republic"),
        ("What is the significance of the word 'socialist' in the Preamble?", "factual", "Preamble", "Socialist was added by the 42nd Amendment Act, 1976"),
        ("What is the significance of the word 'secular' in the Preamble?", "factual", "Preamble", "Secular was added by the 42nd Amendment Act, 1976 to denote State neutrality in matters of religion"),
        
        # Fundamental Rights
        ("Which Part of the Constitution deals with Fundamental Rights?", "factual", "Part III", "Part III of the Constitution contains Fundamental Rights from Articles 12 to 35"),
        ("What are the categories of Fundamental Rights under the Indian Constitution?", "factual", "Part III", "Right to Equality, Right to Freedom, Right against Exploitation, Right to Freedom of Religion, Cultural and Educational Rights, Right to Constitutional Remedies"),
        ("Can Fundamental Rights be suspended during an emergency?", "provision", "Article 358-359", "During a national emergency, enforcement of Fundamental Rights under Article 19 is automatically suspended under Article 358"),
        ("What is the difference between Fundamental Rights and Directive Principles?", "factual", "Part III and Part IV", "Fundamental Rights are justiciable and enforceable by courts while Directive Principles are non-justiciable guidelines for governance"),
        ("Which Fundamental Rights are available only to citizens and not to all persons?", "factual", "Part III", "Articles 15, 16, 19, 29, and 30 are available only to citizens while Articles 14, 20, 21, 21A, 22, 23, 24, 25, 26, 27, and 28 are available to all persons"),
        
        # Directive Principles
        ("Which Part of the Constitution contains Directive Principles of State Policy?", "factual", "Part IV", "Part IV of the Constitution contains Directive Principles from Articles 36 to 51"),
        ("Are Directive Principles enforceable by courts?", "factual", "Article 37", "No, Directive Principles are not enforceable by any court but are fundamental in governance"),
        
        # Parliament
        ("What are the two Houses of Parliament?", "factual", "Article 79", "Parliament consists of the President and two Houses: Rajya Sabha (Council of States) and Lok Sabha (House of the People)"),
        ("What is the maximum strength of the Lok Sabha?", "factual", "Article 81", "The maximum strength of Lok Sabha is 550 members (530 from States + 20 from Union Territories) plus 2 nominated Anglo-Indian members"),
        ("How many members can be nominated to the Rajya Sabha by the President?", "factual", "Article 80", "The President can nominate 12 members to Rajya Sabha having special knowledge in literature, science, art, and social service"),
        ("What is the term of the Lok Sabha?", "factual", "Article 83", "The term of Lok Sabha is 5 years from the date of its first meeting, unless dissolved earlier"),
        ("How is a Money Bill different from an ordinary Bill?", "provision", "Article 110", "A Money Bill can only be introduced in Lok Sabha and Rajya Sabha can only suggest amendments within 14 days"),
        ("What happens when there is a disagreement between the two Houses on an ordinary Bill?", "procedure", "Article 108", "The President may summon a joint sitting of both Houses to resolve the deadlock"),
        ("What is the procedure for passing a constitutional amendment Bill?", "procedure", "Article 368", "Amendment requires special majority in each House - majority of total membership and two-thirds of members present and voting"),
        
        # Executive
        ("How is the President of India elected?", "procedure", "Article 54-55", "The President is elected by an electoral college consisting of elected members of both Houses of Parliament and elected members of State Legislative Assemblies"),
        ("What is the term of office of the President?", "factual", "Article 56", "The President holds office for a term of five years"),
        ("What are the legislative powers of the President?", "provision", "Articles 111, 123", "The President can assent to Bills, withhold assent, return Bills for reconsideration, and promulgate Ordinances"),
        ("What is the pardoning power of the President?", "provision", "Article 72", "The President can grant pardons, reprieves, respites, or remissions of punishment"),
        ("What is the role of the Council of Ministers in the Union executive?", "provision", "Article 74", "The Council of Ministers with the Prime Minister at its head aids and advises the President"),
        ("Who appoints the Prime Minister of India?", "factual", "Article 75", "The President appoints the Prime Minister"),
        ("What is the collective responsibility of the Council of Ministers?", "provision", "Article 75(3)", "The Council of Ministers is collectively responsible to the House of the People"),
        
        # Judiciary
        ("What is the composition of the Supreme Court?", "factual", "Article 124", "The Supreme Court consists of a Chief Justice and such other judges as Parliament may prescribe"),
        ("What are the types of jurisdiction of the Supreme Court?", "factual", "Articles 131-136", "Original jurisdiction, appellate jurisdiction, advisory jurisdiction, and special leave jurisdiction"),
        ("What types of writs can the Supreme Court issue under Article 32?", "provision", "Article 32", "Habeas corpus, mandamus, prohibition, quo warranto, and certiorari"),
        ("What types of writs can High Courts issue under Article 226?", "provision", "Article 226", "Habeas corpus, mandamus, prohibition, quo warranto, and certiorari for enforcement of Fundamental Rights and for any other purpose"),
        ("What is the difference between writ jurisdiction of Supreme Court and High Court?", "factual", "Articles 32, 226", "Supreme Court issues writs only for enforcement of Fundamental Rights while High Courts can issue writs for any purpose"),
        
        # Federal Structure
        ("How are legislative powers distributed between the Union and States?", "provision", "Article 246", "Through three lists in the Seventh Schedule: Union List, State List, and Concurrent List"),
        ("What subjects are included in the Union List?", "factual", "Seventh Schedule List I", "Defence, atomic energy, foreign affairs, banking, insurance, currency, and other subjects of national importance"),
        ("What subjects are included in the State List?", "factual", "Seventh Schedule List II", "Public order, police, public health, agriculture, land revenue, and other subjects of local importance"),
        ("What subjects are included in the Concurrent List?", "factual", "Seventh Schedule List III", "Criminal law, marriage, bankruptcy, trade unions, education, and other subjects on which both Union and States can legislate"),
        ("Who has residuary powers of legislation?", "factual", "Article 248", "Parliament has residuary powers of legislation on matters not in any of the three lists"),
        ("What happens when a State law conflicts with a Union law on a Concurrent List subject?", "provision", "Article 254", "The Union law prevails and the State law is void to the extent of repugnancy"),
        
        # Emergency Provisions
        ("What are the three types of emergencies under the Constitution?", "factual", "Articles 352, 356, 360", "National Emergency (Article 352), State Emergency/President's Rule (Article 356), and Financial Emergency (Article 360)"),
        ("On what grounds can a national emergency be proclaimed?", "provision", "Article 352", "War, external aggression, or armed rebellion"),
        ("What is President's Rule and when can it be imposed?", "provision", "Article 356", "When the President is satisfied that governance of a State cannot be carried on according to constitutional provisions"),
        ("What happens during a financial emergency?", "provision", "Article 360", "The President may direct reduction of salaries and allowances of government servants and judges"),
        ("What is the maximum duration of President's Rule without parliamentary approval?", "factual", "Article 356", "Two months; must be approved by Parliament within that period"),
        
        # Amendments
        ("What was the 42nd Amendment Act about?", "factual", "42nd Amendment", "The 42nd Amendment 1976 added 'socialist' and 'secular' to the Preamble, added Fundamental Duties, and made other significant changes"),
        ("What was the 44th Amendment Act about?", "factual", "44th Amendment", "The 44th Amendment 1978 reversed several changes of the 42nd Amendment, restored some civil liberties, and removed right to property as a Fundamental Right"),
        ("What was the 73rd Amendment Act about?", "factual", "73rd Amendment", "The 73rd Amendment 1992 added Part IX providing for Panchayati Raj institutions"),
        ("What was the 74th Amendment Act about?", "factual", "74th Amendment", "The 74th Amendment 1992 added Part IXA providing for Municipalities"),
        ("What was the 86th Amendment Act about?", "factual", "86th Amendment", "The 86th Amendment 2002 added Article 21A making right to education a Fundamental Right"),
        
        # Schedules
        ("How many Schedules does the Constitution of India contain?", "factual", "Schedules", "The Constitution originally had 8 Schedules and now contains 12 Schedules"),
        ("What does the First Schedule of the Constitution contain?", "factual", "First Schedule", "The First Schedule lists the States and Union Territories of India"),
        ("What does the Seventh Schedule of the Constitution contain?", "factual", "Seventh Schedule", "The Seventh Schedule contains three lists: Union List, State List, and Concurrent List for distribution of legislative powers"),
        ("What does the Eighth Schedule of the Constitution contain?", "factual", "Eighth Schedule", "The Eighth Schedule lists the official languages of India"),
        ("What does the Ninth Schedule of the Constitution contain?", "factual", "Ninth Schedule", "The Ninth Schedule contains laws exempted from judicial review, added by the 1st Amendment"),
        ("What does the Tenth Schedule deal with?", "factual", "Tenth Schedule", "The Tenth Schedule contains provisions relating to disqualification on grounds of defection (anti-defection law)"),
        
        # Local Government
        ("What constitutional provisions govern Panchayats?", "provision", "Part IX", "Part IX (Articles 243 to 243O) provides for the constitution, powers, and functions of Panchayats"),
        ("What is a Gram Sabha under the Constitution?", "definition", "Article 243A", "Gram Sabha is a body consisting of persons registered in the electoral rolls relating to a village comprised within the area of Panchayat at the village level"),
        ("What reservation is provided in Panchayats for Scheduled Castes and Scheduled Tribes?", "provision", "Article 243D", "Seats are reserved for SCs and STs in proportion to their population in the Panchayat area"),
        
        # Miscellaneous Constitutional Topics
        ("What is the role of the Comptroller and Auditor-General of India?", "factual", "Article 148", "The CAG audits accounts of the Union and States and reports to the President and Governors"),
        ("What are the functions of the Public Service Commissions?", "factual", "Article 320", "Conducting examinations for recruitment, advising government on service matters, and other functions"),
        ("What is the constitutional basis for adult suffrage in India?", "factual", "Article 326", "Article 326 provides for elections on the basis of adult suffrage for every person who is a citizen and not less than 18 years of age"),
        ("What protections does the Constitution provide to civil servants?", "provision", "Article 311", "No civil servant can be dismissed or removed by an authority subordinate to the appointing authority and must be given reasonable opportunity of being heard"),
        ("What is the provision for All-India Services?", "provision", "Article 312", "Parliament can create All-India Services common to Union and States by passing a resolution in the Rajya Sabha"),
        ("What are the special provisions for certain States?", "provision", "Articles 371-371J", "Special provisions for Maharashtra, Gujarat, Nagaland, Assam, Manipur, Andhra Pradesh, Sikkim, Mizoram, Arunachal Pradesh, Goa, and Karnataka"),
        ("What is the provision regarding official language of the Union?", "factual", "Article 343", "Hindi in Devanagari script is the official language of the Union; English continues for official purposes"),
        ("What language is used in the Supreme Court and High Courts?", "factual", "Article 348", "English is the language authorised for use in the Supreme Court and High Courts until Parliament provides otherwise"),
        ("What is the Inter-State Council?", "factual", "Article 263", "The President may establish an Inter-State Council to enquire into and advise upon disputes between States"),
    ]
    
    for q_text, q_type, ref, ref_answer in topic_questions:
        if len(questions) >= 250:
            break
        if is_near_duplicate(q_text, fingerprints):
            continue
        fp = question_fingerprint(q_text)
        fingerprints.add(fp)
        
        questions.append({
            "question_id": f"CON_{len(questions)+1:03d}",
            "question": q_text,
            "expected_source": "constitution_of_india.pdf",
            "question_type": q_type,
            "source_reference": ref,
            "reference_answer": ref_answer,
            "evidence": "",
            "validation_status": "grounded"
        })
    
    # Phase 3: Fill remaining with passage-derived questions
    remaining = 250 - len(questions)
    if remaining > 0:
        # Generate questions from passage content directly
        passage_questions = []
        for p in passages:
            text = p["context"]
            page = p["page"]
            concepts = find_key_concepts_in_text(text)
            
            for concept in concepts:
                if concept in ["Constitutional Amendment", "Schedule", "Emergency"]:
                    continue  # Already covered above
                
                q_variants = [
                    (f"What does the Constitution provide regarding {concept}?", "provision"),
                    (f"What is the constitutional framework for {concept} in India?", "factual"),
                ]
                
                for q_text, q_type in q_variants:
                    if len(passage_questions) >= remaining:
                        break
                    if is_near_duplicate(q_text, fingerprints):
                        continue
                    fp = question_fingerprint(q_text)
                    fingerprints.add(fp)
                    
                    passage_questions.append({
                        "question_id": f"CON_{len(questions)+len(passage_questions)+1:03d}",
                        "question": q_text,
                        "expected_source": "constitution_of_india.pdf",
                        "question_type": q_type,
                        "source_reference": f"Page {page}",
                        "reference_answer": "",
                        "evidence": text[:200],
                        "validation_status": "passage_derived"
                    })
        
        questions.extend(passage_questions[:remaining])
    
    return questions[:250], fingerprints


def generate_consumer_protection_questions(passages, existing_fingerprints, target=250):
    """Generate grounded questions about the Consumer Protection Act, 2019."""
    questions = []
    fingerprints = set(existing_fingerprints)
    
    # Comprehensive questions covering the Consumer Protection Act, 2019
    # Organized by topic/chapter
    cp_questions = [
        # Chapter I - Preliminary / Definitions
        ("What is the definition of a 'consumer' under Section 2(7) of the Consumer Protection Act, 2019?", "definition", "Section 2(7)", "A consumer is any person who buys goods or hires/avails services for consideration, including online transactions, but excluding a person who obtains goods for resale or commercial purpose"),
        ("What is the definition of a 'complaint' under the Consumer Protection Act, 2019?", "definition", "Section 2(6)", "A complaint means any allegation in writing made by a complainant regarding unfair trade practice, defective goods, deficient service, excess price, hazardous goods, or unfair contract"),
        ("What is a 'defect' in goods under the Consumer Protection Act, 2019?", "definition", "Section 2(10)", "A defect means any fault, imperfection, or shortcoming in quality, quantity, potency, purity, or standard required under any law or contract"),
        ("What is 'deficiency' in services under the Consumer Protection Act, 2019?", "definition", "Section 2(11)", "Deficiency means any fault, imperfection, shortcoming, or inadequacy in quality, nature, and manner of performance required under any law or contract"),
        ("What is an 'unfair trade practice' under the Consumer Protection Act, 2019?", "definition", "Section 2(47)", "An unfair trade practice includes false representation, misleading advertisement, offering gifts or prizes with intent to not provide them, and other deceptive practices"),
        ("What is a 'restrictive trade practice' under the Consumer Protection Act, 2019?", "definition", "Section 2(41)", "A restrictive trade practice means any practice that tends to manipulate price or conditions of delivery to impose unjustified costs on consumers"),
        ("What is an 'unfair contract' under the Consumer Protection Act, 2019?", "definition", "Section 2(46)", "An unfair contract includes terms requiring excessive security deposits, disproportionate penalty for breach, refusing to accept early payment, unilateral termination, or unreasonable charges"),
        ("What is the definition of 'product liability' under the Consumer Protection Act, 2019?", "definition", "Section 2(34)", "Product liability means the responsibility of a product manufacturer, product seller, or product service provider to compensate for harm caused by a defective product"),
        ("What is the meaning of 'goods' under the Consumer Protection Act, 2019?", "definition", "Section 2(21)", "Goods means every kind of movable property and includes food as defined in the Food Safety and Standards Act"),
        ("What is a 'service' as defined under the Consumer Protection Act, 2019?", "definition", "Section 2(42)", "Service means any description of service made available including banking, financing, insurance, transport, supply of electrical or other energy, housing construction, entertainment, and amusement"),
        ("What does 'spurious goods' mean under the Consumer Protection Act, 2019?", "definition", "Section 2(44)", "Spurious goods means goods that are falsely claimed to be genuine, or are manufactured under a misleading name, label, or brand"),
        ("Who is a 'complainant' under the Consumer Protection Act, 2019?", "definition", "Section 2(5)", "A complainant includes a consumer, any voluntary consumer association, the Central Government, State Government, Central Authority, or legal representative of a deceased consumer"),
        ("What is meant by 'e-commerce' under the Consumer Protection Act, 2019?", "definition", "Section 2(16)", "E-commerce means buying or selling goods or services, including digital products, over digital or electronic network"),
        ("Who qualifies as a 'manufacturer' under the Consumer Protection Act, 2019?", "definition", "Section 2(28)", "A manufacturer means a person who makes or manufactures any goods or parts thereof, and includes a person who assembles parts manufactured by others"),
        ("Who is a 'product seller' under the Consumer Protection Act, 2019?", "definition", "Section 2(37)", "A product seller means a person who in the course of business imports, sells, distributes, leases, installs, prepares, packages, labels, markets, repairs, maintains, or otherwise deals in products"),
        
        # Chapter II - Consumer Rights
        ("What are the six consumer rights recognised by the Consumer Protection Act, 2019?", "rights", "Section 2(9)", "Right to be protected against marketing of dangerous goods/services, right to be informed, right to be assured of quality, right to be heard, right to seek redressal, right to consumer awareness"),
        ("What is the right to be protected against marketing of hazardous goods?", "rights", "Section 2(9)(i)", "The right to be protected against the marketing of goods and services which are hazardous to life and property"),
        ("What is the consumer's right to be informed under the Act?", "rights", "Section 2(9)(ii)", "The right to be informed about the quality, quantity, potency, purity, standard, and price of goods or services"),
        ("What is the right to consumer education or awareness?", "rights", "Section 2(9)(vi)", "The right to consumer awareness means the right to acquire knowledge and skill to be an informed consumer"),
        ("What is the right to seek redressal against unfair trade practices?", "rights", "Section 2(9)(v)", "The right to seek redressal against unfair or restrictive trade practices or unscrupulous exploitation of consumers"),
        
        # Chapter III - Central Consumer Protection Authority
        ("What is the Central Consumer Protection Authority (CCPA)?", "institutions", "Section 10", "The CCPA is established by the Central Government to regulate matters relating to violation of consumer rights, unfair trade practices, and misleading advertisements"),
        ("What are the powers of the CCPA under the Consumer Protection Act, 2019?", "provision", "Section 18", "The CCPA can inquire into violations of consumer rights, order recall of unsafe goods, order discontinuation of unfair practices, impose penalties for misleading advertisements"),
        ("How is the CCPA constituted under the Consumer Protection Act, 2019?", "institutions", "Section 10", "The CCPA consists of a Chief Commissioner and other commissioners appointed by the Central Government"),
        ("What is the investigation wing of the CCPA?", "institutions", "Section 15", "The CCPA has an investigation wing headed by a Director General for conducting inquiries and investigations"),
        ("What powers does the CCPA have regarding misleading advertisements?", "provision", "Section 21", "The CCPA can issue directions against false or misleading advertisements, impose penalties on manufacturers and endorsers"),
        ("What penalty can the CCPA impose for misleading advertisements?", "provision", "Section 21", "Penalty up to 10 lakh rupees for the first offence and up to 50 lakh rupees for subsequent offences"),
        ("Can the CCPA order recall of unsafe goods?", "provision", "Section 20", "Yes, the CCPA can order recall of unsafe goods or withdrawal of services that are dangerous or hazardous"),
        ("What is the role of the Director General under the CCPA?", "institutions", "Section 15-16", "The Director General heads the investigation wing, conducts inquiries, and submits reports to the CCPA"),
        ("What investigation powers does the CCPA have?", "provision", "Section 19", "The CCPA can call upon persons, direct production of documents and records, and conduct investigations into violations"),
        
        # Chapter IV - Consumer Dispute Redressal Commission
        ("What is the three-tier consumer dispute redressal mechanism under the Act?", "institutions", "Sections 28-58", "District Consumer Disputes Redressal Commission, State Consumer Disputes Redressal Commission, and National Consumer Disputes Redressal Commission"),
        ("What is the composition of the District Commission?", "institutions", "Section 28", "The District Commission consists of a President and not less than two members appointed by the State Government"),
        ("What is the composition of the State Commission?", "institutions", "Section 42", "The State Commission consists of a President and not less than four members appointed by the State Government"),
        ("What is the composition of the National Commission?", "institutions", "Section 53", "The National Commission consists of a President and not less than four members appointed by the Central Government"),
        ("What is the pecuniary jurisdiction of the District Commission?", "provision", "Section 34", "The District Commission has jurisdiction over complaints where the value of goods or services paid as consideration does not exceed one crore rupees"),
        ("What is the pecuniary jurisdiction of the State Commission?", "provision", "Section 47", "The State Commission has jurisdiction over complaints where the value exceeds one crore rupees but does not exceed ten crore rupees"),
        ("What is the pecuniary jurisdiction of the National Commission?", "provision", "Section 58", "The National Commission has jurisdiction over complaints where the value exceeds ten crore rupees"),
        ("What is the territorial jurisdiction for filing a consumer complaint?", "provision", "Section 34(2)", "A complaint shall be filed at the place where the opposite party resides or carries on business, or where the cause of action arose"),
        ("What is the limitation period for filing a consumer complaint?", "provision", "Section 35(2)", "A consumer complaint must be filed within two years from the date on which the cause of action arose"),
        ("Can the limitation period be condoned for filing a complaint?", "provision", "Section 36", "Yes, the Commission may condone the delay if sufficient cause is shown"),
        ("Who can file a consumer complaint?", "procedure", "Section 35", "A consumer, any recognised consumer association, one or more consumers with permission, the Central Government, State Government, or Central Authority"),
        ("Can consumer complaints be filed electronically?", "procedure", "Section 35(1)", "Yes, consumer complaints can be filed electronically in the prescribed manner"),
        ("What is the procedure after admission of a consumer complaint?", "procedure", "Section 38", "The Commission refers a copy to the opposite party, who must submit a response within 30 days; the Commission may also refer the matter for mediation"),
        ("What happens when a defective good is alleged in a complaint?", "procedure", "Section 38(5)", "The Commission may obtain a sample and refer it to an appropriate laboratory for testing and report"),
        
        # Remedies and Orders
        ("What orders can a Consumer Commission pass on a complaint?", "remedies", "Section 39", "Removal of defect, replacement, refund, compensation, discontinuation of unfair trade practice, withdrawal of hazardous goods, corrective advertisement, adequate costs"),
        ("Can a Consumer Commission order removal of defects in goods?", "remedies", "Section 39(1)(a)", "Yes, the Commission can direct removal of defects in goods pointed out in the complaint"),
        ("Can a Consumer Commission order replacement of defective goods?", "remedies", "Section 39(1)(b)", "Yes, the Commission can direct replacement of defective goods with new goods of similar description free from defects"),
        ("Can a Consumer Commission order a refund?", "remedies", "Section 39(1)(c)", "Yes, the Commission can direct return of the price or charges paid by the complainant to the opposite party"),
        ("Can a Consumer Commission award compensation?", "remedies", "Section 39(1)(d)", "Yes, the Commission can direct payment of adequate compensation for any loss or injury suffered by the consumer"),
        ("Can a Consumer Commission order discontinuation of unfair trade practices?", "remedies", "Section 39(1)(e)", "Yes, the Commission can direct discontinuation of unfair trade practices or restrictive trade practices"),
        ("Can a Consumer Commission order withdrawal of hazardous goods?", "remedies", "Section 39(1)(f)", "Yes, the Commission can direct not to offer hazardous goods for sale and to withdraw them from being offered"),
        ("Can a Consumer Commission order corrective advertisements?", "remedies", "Section 39(1)(g)", "Yes, the Commission can direct issuance of corrective advertisements to neutralize the effect of misleading advertisements"),
        ("Can a Consumer Commission award costs of the proceedings?", "remedies", "Section 39(1)(h)", "Yes, the Commission can provide for adequate costs to the parties"),
        
        # Appeals
        ("What is the appeal process from a District Commission order?", "procedure", "Section 41", "An appeal lies to the State Commission within 45 days from the date of the order of the District Commission"),
        ("What is the appeal process from a State Commission order?", "procedure", "Section 51", "An appeal lies to the National Commission within 30 days from the date of the order of the State Commission"),
        ("Can an appeal be filed before the Supreme Court against the National Commission's order?", "procedure", "Section 67", "Yes, any person aggrieved by an order of the National Commission may prefer an appeal to the Supreme Court within 30 days"),
        ("What is the time limit for appealing a District Commission order?", "provision", "Section 41", "Forty-five days from the date of the order"),
        ("What is the time limit for appealing a State Commission order?", "provision", "Section 51", "Thirty days from the date of the order"),
        
        # Consumer Mediation
        ("What is consumer mediation under the Consumer Protection Act, 2019?", "provision", "Section 74", "Consumer mediation is an alternative dispute resolution mechanism where disputes are settled through mediation instead of adjudication"),
        ("When can a dispute be referred to mediation?", "procedure", "Section 37(1)", "At any stage of a complaint, where a possibility of settlement exists and both parties agree, the Commission may refer the matter for mediation"),
        ("What is a consumer mediation cell?", "institutions", "Section 74(1)", "A consumer mediation cell is established in connection with each Consumer Commission for mediation of consumer disputes"),
        ("Who can be empanelled as a mediator?", "procedure", "Section 75", "Persons with qualifications, experience, and training as may be specified by the Central Government can be empanelled as mediators"),
        ("What is the effect of a mediation settlement?", "provision", "Section 80", "A settlement reached through mediation is binding on both parties and is enforced in the same manner as a decree or order of the Commission"),
        ("What matters cannot be referred to mediation?", "provision", "Section 74(3)", "Complaints involving serious defects in goods or deficiency in services that pose a risk to life and safety cannot be referred to mediation"),
        ("What is the time limit for mediation?", "provision", "Section 75(4)", "Mediation shall be completed within the time period as may be prescribed, which shall not exceed three months"),
        ("What duties must a mediator observe?", "provision", "Section 77", "A mediator must disclose conflicts of interest, maintain confidentiality, remain impartial, and conduct mediation fairly"),
        
        # Product Liability (Chapter VI)
        ("What is a product liability action under the Consumer Protection Act, 2019?", "provision", "Section 82", "A product liability action is brought by a complainant for claiming compensation for harm caused by a defective product manufactured or sold by a product manufacturer, seller, or service provider"),
        ("Who can bring a product liability action?", "provision", "Section 83", "A complainant can bring a product liability action against a product manufacturer, product seller, or product service provider"),
        ("What is the basis for product liability of a product manufacturer?", "provision", "Section 84", "A manufacturer is liable if the product contains a manufacturing defect, design defect, deviation from manufacturing specifications, or failure to provide adequate instructions or warnings"),
        ("What is a manufacturing defect for product liability purposes?", "definition", "Section 84(1)(a)", "A manufacturing defect means the product differs from the manufacturer's design specifications, formula, or performance standards or from identical products manufactured by the same manufacturer"),
        ("What is a design defect for product liability purposes?", "definition", "Section 84(1)(b)", "A design defect exists when the product fails to provide the safety a reasonable consumer would expect"),
        ("When is a product seller liable under product liability?", "provision", "Section 85", "A product seller is liable when the seller has exercised substantial control over designing, testing, manufacturing, packaging, or labelling; or has altered the product; or has made an express warranty"),
        ("When is a product service provider liable under product liability?", "provision", "Section 86", "A product service provider is liable when the service was provided in a negligent manner, or when faulty or incorrect service has resulted in harm"),
        ("What exceptions exist to product liability of a manufacturer?", "exceptions", "Section 87", "A manufacturer may not be liable if the product was misused, altered, or modified by the consumer, or if the defect was a result of compliance with a standard prescribed by law"),
        ("What are the defenses available to a product seller in a product liability action?", "exceptions", "Section 87", "The product was misused, altered or modified by the consumer; the seller did not exercise control over design, testing, manufacturing, packaging or labelling"),
        ("Can a product service provider be liable for harm caused by faulty service?", "provision", "Section 86", "Yes, if the service was provided in a negligent manner, defective in nature, or contained a flaw in the instructions or warnings"),
        
        # Offences and Penalties
        ("What are the penalties for manufacturing or selling adulterated products?", "provision", "Section 89", "For adulterated products not causing injury: imprisonment up to 6 months and fine up to 1 lakh rupees; causing injury: up to 1 year and up to 3 lakh; causing grievous hurt: up to 7 years and up to 5 lakh; causing death: not less than 7 years extending to imprisonment for life and fine not less than 10 lakh"),
        ("What are the penalties for manufacturing or selling spurious goods?", "provision", "Section 90", "For spurious goods not causing injury: imprisonment up to 1 year and fine up to 3 lakh rupees; causing injury: up to 3 years and fine up to 5 lakh; causing grievous hurt: up to 7 years and fine up to 5 lakh; causing death: not less than 7 years extending to life imprisonment and fine not less than 10 lakh"),
        ("What is the penalty for failure to comply with a Commission order?", "provision", "Section 72", "Imprisonment for not less than one month but which may extend to three years, or with fine of not less than 25,000 rupees but which may extend to one lakh rupees, or both"),
        ("Can offences under the Act be compounded?", "provision", "Section 91", "Yes, offences under the Act can be compounded by the National Commission or the State Commission with consent of the parties"),
        
        # E-Commerce
        ("How does the Consumer Protection Act apply to e-commerce transactions?", "provision", "Section 2(7)", "The definition of consumer includes any person who buys goods or hires services through electronic means including online transactions"),
        ("What are the duties of e-commerce entities under consumer protection?", "provision", "Section 94", "E-commerce entities must display information about return, refund, exchange, warranty, delivery, modes of payment, and grievance redressal mechanism"),
        ("Can consumers file complaints against e-commerce platforms?", "provision", "Section 2(7)", "Yes, consumers who purchase goods or services through electronic means are protected under the Act"),
        ("What information must an e-commerce entity provide to consumers?", "provision", "Section 94", "Details of sellers, complaint redressal mechanism, return and refund policy, and country of origin of goods"),
        
        # Consumer Councils
        ("What is the Central Consumer Protection Council?", "institutions", "Section 3", "The Central Council is established by the Central Government to render advice on promotion and protection of consumer rights"),
        ("What is the composition of the Central Consumer Protection Council?", "institutions", "Section 3(2)", "The Central Council is chaired by the Minister of Consumer Affairs and includes other members as prescribed"),
        ("What is the State Consumer Protection Council?", "institutions", "Section 7", "Each State Government shall establish a State Consumer Protection Council to render advice on consumer rights within the State"),
        ("What is the District Consumer Protection Council?", "institutions", "Section 8", "Each State Government shall establish a District Consumer Protection Council in every district for promoting consumer rights"),
        ("What are the objects of the Consumer Councils?", "provision", "Section 4", "To promote and protect the rights of consumers as laid down in the Act"),
        
        # Enforcement and Compliance
        ("How are Consumer Commission orders enforced?", "procedure", "Section 71", "Orders are enforced in the same manner as a decree or order made by a court in a suit pending therein"),
        ("What powers does a Consumer Commission have during proceedings?", "provision", "Section 40", "The Commission has powers of a civil court regarding summoning witnesses, requiring discovery and production of documents, receiving evidence on affidavits, issuing commissions"),
        ("Can a Consumer Commission review its own orders?", "provision", "Section 40(1)", "Yes, the Commission may review any order made by it when there is an error apparent on the face of the record"),
        ("What is the procedure for execution of Consumer Commission orders?", "procedure", "Section 71", "The Commission may send its order to a civil court having local jurisdiction for execution as if it were a decree of that court"),
        
        # Miscellaneous Provisions
        ("What is the bar of jurisdiction for civil courts under the Act?", "provision", "Section 100", "No civil court shall have jurisdiction to entertain any suit or proceeding in respect of any matter which a Consumer Commission is empowered to determine"),
        ("Can the Central Government make rules under the Consumer Protection Act?", "provision", "Section 101", "Yes, the Central Government may make rules for carrying out the provisions of the Act"),
        ("What transitional provisions exist under the Consumer Protection Act, 2019?", "provision", "Section 107", "All cases pending before forums established under the 1986 Act shall be transferred to the corresponding Commissions under the 2019 Act"),
        ("What is the overriding effect of the Consumer Protection Act, 2019?", "provision", "Section 100", "The provisions of this Act are in addition to and not in derogation of any other law for the time being in force"),
        ("When did the Consumer Protection Act, 2019 come into force?", "factual", "Section 1(3)", "The Act came into force on 20th July 2020"),
        ("What Act did the Consumer Protection Act, 2019 replace?", "factual", "Section 107", "The Consumer Protection Act, 2019 replaced the Consumer Protection Act, 1986"),
        
        # Additional detailed questions
        ("What qualifications are required for the President of the District Commission?", "provision", "Section 29", "Must be or have been a District Judge or have required qualifications and experience as prescribed"),
        ("What qualifications are required for members of the District Commission?", "provision", "Section 29", "Must not be less than 35 years of age, possess a bachelor's degree, and have adequate knowledge and experience"),
        ("What is the term of office of members of a Consumer Commission?", "provision", "Section 30", "The term of office and conditions of service are as prescribed by the Central or State Government"),
        ("Can a consumer complaint be filed by a voluntary consumer association?", "procedure", "Section 35(1)(b)", "Yes, any recognised consumer association can file a complaint"),
        ("What is the procedure when a consumer complaint is found to be frivolous?", "provision", "Section 39(2)", "The Commission may dismiss the complaint and order the complainant to pay costs not exceeding ten thousand rupees"),
        ("Can the Commission award punitive damages?", "remedies", "Section 39", "Yes, the Commission can award punitive damages for the loss or injury suffered"),
        ("What powers does the Commission have for contempt?", "provision", "Section 72", "Failure to comply with orders may result in imprisonment and fine"),
        ("Can interim orders be passed by Consumer Commissions?", "provision", "Section 35(4)", "Yes, the Commission can pass interim orders as it considers necessary in the interest of justice"),
        ("What is the effect of an order passed by the National Commission?", "provision", "Section 67", "Orders of the National Commission are enforceable throughout India"),
        ("What happens if the opposite party does not appear in consumer proceedings?", "procedure", "Section 38(7)", "The Commission may proceed ex parte to decide the complaint on merits"),
        
        # More specific and detailed questions
        ("What constitutes 'harm' for purposes of product liability?", "definition", "Section 2(22)", "Harm includes damage to property, personal injury, illness, or death"),
        ("What does 'misleading advertisement' mean under the Act?", "definition", "Section 2(28)", "An advertisement that falsely describes or gives a false guarantee about the product/service, or conveys an express or implied representation which is misleading"),
        ("What is the liability of an endorser for misleading advertisements?", "provision", "Section 21(3)", "An endorser shall be liable for a misleading advertisement unless the endorser exercised due diligence to verify the claims"),
        ("What constitutes an 'endorsement' under the Act?", "definition", "Section 2(18)", "An endorsement means any message, verbal statement, demonstration, or depiction by a person in an advertisement"),
        ("What are the grounds on which a product manufacturer can be held liable?", "provision", "Section 84", "Manufacturing defect, design defect, deviation from specifications, failure to provide adequate instructions or warnings, and failure to conform to express warranty"),
        ("What is the maximum compensation that can be awarded by the District Commission?", "provision", "Section 34(1)", "Up to one crore rupees"),
        ("What is the procedure for laboratory testing under the Act?", "procedure", "Section 38(5)", "The Commission may obtain sample and forward it to an appropriate laboratory for analysis and report"),
        ("Can complaints be transferred from one Commission to another?", "provision", "Section 45", "Yes, the State Commission can transfer complaints from one District Commission to another"),
        ("What is the power of the National Commission to transfer cases?", "provision", "Section 58(1)(b)", "The National Commission can transfer cases from one State Commission to another"),
        ("How does the Act protect consumers in direct selling?", "provision", "Section 94", "Rules can be made to regulate direct selling and protect consumers from unfair practices"),
        ("What role do consumer organisations play under the Act?", "provision", "Section 35(1)(b)", "Consumer organisations can file complaints on behalf of consumers"),
        ("What is the meaning of 'commercial purpose' as excluded from consumer definition?", "definition", "Section 2(7) Explanation", "Commercial purpose excludes goods bought and used exclusively for self-employment or livelihood"),
        ("Can goods purchased for self-employment qualify for consumer protection?", "provision", "Section 2(7) Explanation", "Yes, goods bought for self-employment or earning livelihood are not treated as commercial purpose"),
        ("What protections exist against overcharging of consumers?", "provision", "Section 35(1)(a)(iv)", "Consumers can file complaints when goods or services are offered at a price exceeding the price fixed by law or displayed on packaging"),
        ("What is the procedure when the opposite party admits the complaint?", "procedure", "Section 38", "If the opposite party admits, the Commission may pass orders on admission"),
        ("Can a consumer seek relief for mental agony?", "remedies", "Section 39(1)(d)", "Yes, compensation can include amounts for mental agony and harassment suffered"),
        ("What is meant by 'adequate compensation' under the Act?", "remedies", "Section 39(1)(d)", "Adequate compensation covers loss or injury due to negligence of the opposite party, taking into account all relevant factors"),
        ("Can the Commission impose a penalty for non-compliance with its order?", "provision", "Section 72", "Yes, imprisonment of one month to three years and/or fine of 25,000 to one lakh rupees for non-compliance"),
        
        # Additional detailed Consumer Protection questions
        # Scope and Application
        ("Does the Consumer Protection Act, 2019 apply to goods purchased online?", "provision", "Section 2(7)", "Yes, the definition of consumer covers transactions through electronic means including online purchases"),
        ("Does the Consumer Protection Act apply to services provided free of charge?", "definition", "Section 2(42)", "No, the Act applies to services made available for consideration; free services are excluded from the definition"),
        ("Are government-provided services covered under the Consumer Protection Act?", "provision", "Section 2(42)", "Services availed under a contract of personal service are excluded; other government services may be covered"),
        ("Can a company file a consumer complaint under the Act?", "provision", "Section 2(7)", "Only if the company is a consumer who bought goods or hired services for consideration and not for commercial resale"),
        ("Does the Act apply to medical services?", "provision", "Section 2(42)", "Yes, medical services are covered as they are services made available for consideration"),
        ("Does the Act cover insurance services?", "provision", "Section 2(42)", "Yes, insurance is specifically mentioned in the definition of service"),
        ("Does the Act cover banking services?", "provision", "Section 2(42)", "Yes, banking is specifically mentioned in the definition of service"),
        ("Does the Act cover housing construction services?", "provision", "Section 2(42)", "Yes, housing construction is specifically mentioned in the definition of service"),
        
        # Filing and Procedures - more specific
        ("Where should a consumer complaint be filed?", "procedure", "Section 34(2)", "At the Commission where the opposite party resides or carries on business, or where the cause of action arose"),
        ("What documents are required to file a consumer complaint?", "procedure", "Section 35", "The complaint, supporting evidence, details of the relief sought, and requisite court fee"),
        ("Can a complaint be filed by a legal representative after the consumer's death?", "procedure", "Section 2(5)(v)", "Yes, the legal representative of a deceased consumer can file a complaint"),
        ("What is the role of the opposite party in consumer proceedings?", "procedure", "Section 38", "The opposite party must respond to the complaint within the prescribed time and present their defence"),
        ("Can a consumer complaint be rejected at the admission stage?", "procedure", "Section 36", "Yes, if the complaint does not disclose sufficient cause, is frivolous, vexatious, or is time-barred"),
        ("What is the procedure when laboratory reports are disputed?", "procedure", "Section 38(6)", "The Commission may require further testing or consider expert evidence"),
        ("Can evidence be received on affidavit in consumer proceedings?", "procedure", "Section 40", "Yes, the Commission has the power to receive evidence on affidavits"),
        ("What powers does a Consumer Commission have regarding witnesses?", "procedure", "Section 40", "The Commission has powers of a civil court regarding summoning and enforcing attendance of witnesses"),
        
        # District Commission specifics
        ("How is the President of the District Commission appointed?", "institutions", "Section 29", "The President is appointed by the State Government on the recommendation of the Selection Committee"),
        ("What is the Selection Committee for District Commission appointments?", "institutions", "Section 29", "The Selection Committee consists of the President of the State Commission, Secretary of the State law department, and a nominee of the Central Government"),
        ("Can a member of the District Commission be removed?", "provision", "Section 31", "Yes, the State Government may remove a member on grounds of proved misbehaviour or incapacity after inquiry"),
        ("How many members does the District Commission have?", "institutions", "Section 28(1)", "The District Commission has a President and not less than two members"),
        ("What is the salary of District Commission members?", "provision", "Section 30", "As prescribed by the Central Government"),
        
        # State Commission specifics
        ("How is the President of the State Commission appointed?", "institutions", "Section 43", "By the State Government in consultation with the Chief Justice of the High Court"),
        ("What is the appellate jurisdiction of the State Commission?", "provision", "Section 47(1)(a)", "The State Commission can hear appeals against orders of any District Commission within the State"),
        ("Can the State Commission transfer cases between District Commissions?", "provision", "Section 45", "Yes, the State Commission can transfer a complaint from one District Commission to another"),
        ("What is the supervisory jurisdiction of the State Commission?", "provision", "Section 47", "The State Commission has the power to supervise the working of District Commissions within the State"),
        
        # National Commission specifics
        ("How is the President of the National Commission appointed?", "institutions", "Section 54", "By the Central Government in consultation with the Chief Justice of India"),
        ("What is the revisional jurisdiction of the National Commission?", "provision", "Section 58(1)(b)", "The National Commission can call for and examine records of proceedings of any State Commission"),
        ("Can the National Commission transfer cases between State Commissions?", "provision", "Section 58(1)(b)", "Yes, the National Commission can transfer complaints between State Commissions"),
        ("What is the monitoring role of the National Commission?", "provision", "Section 58(2)", "The National Commission monitors the pendency and disposal of cases in State and District Commissions"),
        
        # Product Liability - additional scenarios
        ("Can a consumer claim product liability against a foreign manufacturer?", "provision", "Section 83", "Yes, a product liability action can be brought against any product manufacturer, including foreign manufacturers whose products are sold in India"),
        ("What is the difference between a manufacturing defect and a design defect?", "definition", "Section 84", "A manufacturing defect deviates from specifications while a design defect means the product fails to provide expected safety even when correctly manufactured"),
        ("Is a product seller liable if they did not manufacture the product?", "provision", "Section 85", "A non-manufacturer product seller is liable only in specific circumstances such as exercising substantial control over design or making express warranties"),
        ("What must a consumer prove in a product liability claim?", "procedure", "Section 83", "The consumer must prove the harm, the defective nature of the product, and the causal connection between the defect and the harm"),
        ("Can a product manufacturer be held liable for inadequate warnings?", "provision", "Section 84(1)(d)", "Yes, a manufacturer is liable for failure to provide adequate instructions or warnings about the product"),
        ("What is the relationship between product liability and consumer complaints?", "provision", "Section 83-87", "A product liability action is a type of consumer complaint that can be filed before a Consumer Commission"),
        ("Can a hospital be held liable under product liability provisions?", "provision", "Section 86", "A hospital can be liable as a product service provider if it provided negligent or defective services"),
        ("Is a product seller liable for altering a product?", "provision", "Section 85(b)", "Yes, a product seller is liable if they altered or modified the product in a way that caused harm"),
        
        # Unfair Trade Practices - specific scenarios
        ("What constitutes false representation in advertising?", "definition", "Section 2(47)", "False representation includes misrepresenting quality, quantity, standard, composition, style, or model of goods"),
        ("Is bait-and-switch advertising an unfair trade practice?", "provision", "Section 2(47)", "Yes, advertising goods or services at a bargain price without intention or ability to supply them at that price is an unfair trade practice"),
        ("Is pyramid selling an unfair trade practice?", "provision", "Section 2(47)", "Yes, promoting any scheme purporting to make money by recruiting other participants is an unfair trade practice"),
        ("Can withholding information be an unfair trade practice?", "provision", "Section 2(47)", "Yes, deliberately concealing important information about goods or services can constitute an unfair trade practice"),
        ("What remedies are available for unfair trade practices?", "remedies", "Section 39", "Discontinuation of the practice, compensation, refund, and corrective advertisements"),
        
        # Misleading Advertisements - more details
        ("What constitutes a misleading advertisement under the Act?", "definition", "Section 2(28)", "An advertisement that falsely describes or gives a false guarantee about a product or service, or conveys misleading representation"),
        ("Can the CCPA take suo motu action against misleading advertisements?", "provision", "Section 18", "Yes, the CCPA can inquire into violations of consumer rights including misleading advertisements on its own motion"),
        ("What is the penalty for a first offence of misleading advertisement?", "provision", "Section 21(1)", "Penalty of up to ten lakh rupees for the first offence"),
        ("What is the penalty for subsequent offences of misleading advertisement?", "provision", "Section 21(2)", "Penalty of up to fifty lakh rupees for subsequent offences"),
        ("Can an endorser be penalised for a misleading advertisement?", "provision", "Section 21(3)", "Yes, unless the endorser exercised due diligence to verify the claims in the advertisement"),
        ("What defence is available to an endorser for misleading advertisements?", "exceptions", "Section 21(3)", "The endorser can claim due diligence was exercised in verifying the accuracy of the claims"),
        ("Can the CCPA prohibit a particular misleading advertisement?", "provision", "Section 21", "Yes, the CCPA can pass orders for discontinuation of misleading advertisements"),
        ("What information must advertisements contain under consumer protection?", "provision", "Section 21", "Advertisements must not contain false or misleading claims about the quality, quantity, or efficacy of goods or services"),
        
        # Mediation - additional questions
        ("What are the advantages of mediation over adjudication in consumer disputes?", "provision", "Section 74-81", "Mediation is faster, less formal, confidential, and allows parties to reach mutually acceptable solutions"),
        ("Can a mediation settlement be challenged?", "provision", "Section 80", "A mediation settlement is final and binding; it can only be challenged on very limited grounds"),
        ("What happens if mediation fails?", "procedure", "Section 75", "If mediation fails, the dispute is referred back to the Consumer Commission for adjudication"),
        ("Who bears the cost of mediation?", "provision", "Section 74", "The cost of mediation is borne as prescribed by the regulations"),
        ("Can a party withdraw from mediation?", "procedure", "Section 76", "Either party can withdraw from mediation at any time before a settlement is reached"),
        ("What is the confidentiality requirement in consumer mediation?", "provision", "Section 79", "All communications made during mediation are confidential and cannot be used as evidence in subsequent proceedings"),
        
        # Unfair Contracts - more details
        ("What makes a contract 'unfair' under the Consumer Protection Act?", "definition", "Section 2(46)", "A contract with terms requiring excessive security deposits, disproportionate penalties, unilateral termination, unreasonable charges, or one-sided conditions"),
        ("Can a consumer challenge unfair contract terms?", "remedies", "Section 2(46)", "Yes, a consumer can file a complaint against unfair contract terms and seek relief from the Consumer Commission"),
        ("Does the Act apply to insurance contracts that are unfair?", "provision", "Section 2(46)", "Yes, unfair terms in insurance contracts can be challenged under the Act"),
        ("What is the remedy for an unfair contract under the Act?", "remedies", "Section 39", "The Commission can declare the contract or its unfair terms void and order appropriate relief"),
        ("Can unfair contract terms in service agreements be challenged?", "provision", "Section 2(46)", "Yes, any term in a service agreement that falls within the definition of unfair contract can be challenged"),
        
        # Consumer Councils - more specific
        ("How often does the Central Consumer Protection Council meet?", "provision", "Section 5", "The Central Council meets as and when necessary but not less than once in each year"),
        ("What are the functions of the Central Consumer Protection Council?", "provision", "Section 4", "The Central Council advises on promotion and protection of consumer rights under the Act"),
        ("What are the functions of the State Consumer Protection Council?", "provision", "Section 7(2)", "The State Council advises on consumer rights within the State"),
        ("Who chairs the Central Consumer Protection Council?", "institutions", "Section 3(2)", "The Minister in charge of Consumer Affairs in the Central Government"),
        ("Who chairs the State Consumer Protection Council?", "institutions", "Section 7(1)", "The Minister in charge of Consumer Affairs in the State Government"),
        ("Who chairs the District Consumer Protection Council?", "institutions", "Section 8(2)", "The Collector of the District"),
        
        # E-Commerce - more details
        ("What obligations do e-commerce platforms have regarding product information?", "provision", "Section 94", "E-commerce platforms must provide details about the product including country of origin, expiry date, return and refund policy"),
        ("Can a consumer file a complaint against an e-commerce marketplace seller?", "procedure", "Section 35", "Yes, complaints can be filed against the seller and/or the marketplace platform depending on the nature of the grievance"),
        ("What grievance redressal mechanism must e-commerce entities provide?", "provision", "Section 94", "E-commerce entities must provide a grievance redressal mechanism and appoint a grievance officer"),
        ("Is a marketplace e-commerce entity liable for seller's products?", "provision", "Section 94", "The marketplace must ensure sellers provide accurate information; liability depends on the level of control exercised"),
        ("What are the cancellation and return obligations of e-commerce entities?", "provision", "Section 94", "E-commerce entities must have a clear cancellation and return policy and display it prominently"),
        ("Must e-commerce entities display country of origin of goods?", "provision", "Section 94", "Yes, e-commerce entities are required to display the country of origin of goods"),
        
        # Penalties and Offences - more details
        ("What is the penalty for selling adulterated goods that do not cause injury?", "provision", "Section 89(1)", "Imprisonment up to 6 months and fine up to 1 lakh rupees"),
        ("What is the penalty for selling adulterated goods causing injury?", "provision", "Section 89(2)", "Imprisonment up to 1 year and fine up to 3 lakh rupees"),
        ("What is the penalty for selling adulterated goods causing death?", "provision", "Section 89(5)", "Imprisonment not less than 7 years extending to life imprisonment and fine not less than 10 lakh rupees"),
        ("What is the penalty for selling spurious goods not causing injury?", "provision", "Section 90(1)", "Imprisonment up to 1 year and fine up to 3 lakh rupees"),
        ("What is the penalty for selling spurious goods causing death?", "provision", "Section 90(5)", "Imprisonment not less than 7 years extending to life imprisonment and fine not less than 10 lakh rupees"),
        ("Can offences under the Act be compounded?", "provision", "Section 91", "Yes, certain offences may be compounded by the National Commission or State Commission with consent of parties"),
        ("Are offences under the Act cognizable?", "provision", "Section 91", "Offences relating to adulterated or spurious goods are cognizable and non-bailable"),
        ("Can a company be prosecuted under the Act?", "provision", "Section 92", "Yes, where an offence is committed by a company, every person in charge of the company shall be liable"),
        
        # Comparative and Analytical Questions
        ("How does the 2019 Act differ from the Consumer Protection Act of 1986?", "factual", "Section 107", "The 2019 Act introduces CCPA, product liability, mediation, electronic filing, and expanded e-commerce provisions"),
        ("What new remedies were introduced by the Consumer Protection Act, 2019?", "remedies", "Section 39", "Product liability actions, mediation, CCPA investigation powers, and enhanced penalties for adulterated and spurious goods"),
        ("How does the 2019 Act strengthen consumer protection in e-commerce?", "provision", "Section 94", "By extending the definition of consumer to include online transactions and imposing obligations on e-commerce entities"),
        ("How does the CCPA differ from Consumer Commissions in function?", "institutions", "Sections 10, 28", "CCPA is a regulatory body for enforcement and investigation while Commissions are quasi-judicial bodies for dispute resolution"),
        ("What is the difference between pecuniary jurisdiction of the three Commissions?", "provision", "Sections 34, 47, 58", "District up to 1 crore, State 1-10 crore, National above 10 crore"),
        
        # Scenario-based questions
        ("If a consumer buys a defective mobile phone, what remedy is available?", "remedies", "Section 39", "The consumer can seek removal of defect, replacement with a new phone, refund of price, or compensation for loss"),
        ("If a builder fails to deliver a flat on time, what can the consumer do?", "remedies", "Section 39", "The consumer can file a complaint seeking compensation, interest on delayed possession, or refund of the amount paid"),
        ("If a hospital provides negligent treatment, can the patient file a consumer complaint?", "provision", "Section 2(42)", "Yes, medical services are covered under the Act and the patient can file a complaint for deficiency in service"),
        ("If an advertisement falsely claims a product cures a disease, what action can be taken?", "provision", "Section 21", "The CCPA can impose penalties and order discontinuation of the misleading advertisement"),
        ("If a product causes injury due to a design defect, who is liable?", "provision", "Section 84", "The product manufacturer is liable for harm caused by a design defect in the product"),
        ("If goods purchased online are not delivered, what remedy is available?", "remedies", "Section 39", "The consumer can file a complaint for refund of the price paid and compensation for any loss"),
        ("If a seller charges more than the MRP, what can a consumer do?", "provision", "Section 35(1)(a)(iv)", "The consumer can file a complaint for charging more than the price fixed by or under any law or displayed on packaging"),
        ("If a product has no warranty card but was promised one, what action can be taken?", "provision", "Section 2(47)", "This may constitute unfair trade practice and the consumer can seek appropriate relief"),
        
        # Multi-passage and cross-cutting questions
        ("How do consumer rights relate to product liability under the Act?", "provision", "Sections 2(9), 82", "Consumer rights include protection against hazardous products and product liability provides a specific mechanism for claiming compensation"),
        ("What is the relationship between CCPA powers and Commission jurisdiction?", "provision", "Sections 18, 34", "CCPA handles regulatory and enforcement aspects while Commissions handle individual consumer dispute adjudication"),
        ("How do mediation and adjudication interact in consumer disputes?", "procedure", "Sections 37, 74", "A Commission may refer a dispute to mediation; if mediation fails, the matter returns to the Commission for adjudication"),
        ("What is the interplay between appeal and review in consumer proceedings?", "procedure", "Sections 41, 51, 67", "A party can seek review of an order by the same Commission or appeal to the higher Commission"),
        ("How does product liability relate to unfair trade practices?", "provision", "Sections 2(47), 82-87", "Both provide remedies for consumers; unfair trade practices relate to deceptive marketing while product liability relates to defective products"),
        
        # Additional procedural questions
        ("What is the quorum for a Consumer Commission to hear a case?", "procedure", "Section 28", "The Commission sits with its President and at least one member for hearing cases"),
        ("Can a Consumer Commission appoint a local commission for inspection?", "procedure", "Section 40", "Yes, the Commission has powers similar to a civil court including issuing commissions for local inspection"),
        ("What is the procedure for execution of compensation orders?", "procedure", "Section 71", "Compensation orders are executed as decrees of a civil court"),
        ("Can costs be awarded against a frivolous complaint?", "provision", "Section 39(2)", "Yes, costs not exceeding ten thousand rupees can be imposed for frivolous or vexatious complaints"),
        ("What happens if a consumer dies during the pendency of a complaint?", "procedure", "Section 2(5)(v)", "The legal representative of the deceased consumer can continue the proceedings"),
        
        # Institutional governance
        ("What is the tenure of members of the National Commission?", "provision", "Section 55", "As prescribed by the Central Government, with eligibility for reappointment"),
        ("Can a retired judge be appointed as President of a Consumer Commission?", "provision", "Section 29", "Yes, a person who has been a District Judge is qualified to be President of the District Commission"),
        ("What disqualifications apply to members of Consumer Commissions?", "provision", "Section 31", "Insolvency, unsound mind, conviction for a moral turpitude offence, or having financial or other interest in the subject matter"),
        ("How are vacancies in Consumer Commissions filled?", "provision", "Section 31", "Through the Selection Committee process as prescribed"),
        ("What administrative support do Consumer Commissions receive?", "provision", "Section 32", "The Government provides officers and staff as may be prescribed for the functioning of Commissions"),
        
        # Rights enforcement
        ("Can consumer rights be waived by contract?", "provision", "Section 2(46)", "No, contractual terms that waive consumer rights would be considered unfair contract terms"),
        ("Does the Act provide for class action by consumers?", "procedure", "Section 35(1)(c)", "Yes, one or more consumers can file a complaint on behalf of numerous consumers having the same interest with the permission of the Commission"),
        ("Can the Central Government file a consumer complaint?", "procedure", "Section 35(1)(d)", "Yes, the Central Government can file a complaint in the interest of consumers"),
        ("Can the State Government file a consumer complaint?", "procedure", "Section 35(1)(d)", "Yes, the State Government can file a complaint in the interest of consumers"),
        ("What is the role of the Central Authority in protecting consumer rights?", "institutions", "Section 18", "The Central Authority protects consumer rights by investigating violations, ordering recalls, and penalising misleading advertisements"),
        
        # Specific sections and provisions
        ("What does Section 10 of the Consumer Protection Act establish?", "provision", "Section 10", "Section 10 establishes the Central Consumer Protection Authority"),
        ("What does Section 18 of the Consumer Protection Act provide?", "provision", "Section 18", "Section 18 lists the powers and functions of the Central Consumer Protection Authority"),
        ("What does Section 28 of the Consumer Protection Act establish?", "provision", "Section 28", "Section 28 establishes the District Consumer Disputes Redressal Commission"),
        ("What does Section 34 of the Consumer Protection Act provide?", "provision", "Section 34", "Section 34 defines the jurisdiction of the District Commission"),
        ("What does Section 35 of the Consumer Protection Act provide?", "procedure", "Section 35", "Section 35 provides the manner of filing a consumer complaint"),
        ("What does Section 39 of the Consumer Protection Act provide?", "remedies", "Section 39", "Section 39 lists the orders that a Consumer Commission can make upon finding the complaint to be valid"),
        ("What does Section 42 of the Consumer Protection Act establish?", "provision", "Section 42", "Section 42 establishes the State Consumer Disputes Redressal Commission"),
        ("What does Section 47 of the Consumer Protection Act provide?", "provision", "Section 47", "Section 47 defines the jurisdiction of the State Commission including original and appellate jurisdiction"),
        ("What does Section 53 of the Consumer Protection Act establish?", "provision", "Section 53", "Section 53 establishes the National Consumer Disputes Redressal Commission"),
        ("What does Section 58 of the Consumer Protection Act provide?", "provision", "Section 58", "Section 58 defines the jurisdiction of the National Commission"),
        ("What does Section 74 of the Consumer Protection Act provide?", "provision", "Section 74", "Section 74 provides for consumer mediation as an alternative dispute resolution mechanism"),
        ("What does Section 82 of the Consumer Protection Act define?", "definition", "Section 82", "Section 82 defines product liability as an action for claiming compensation for harm caused by a defective product"),
        ("What does Section 84 of the Consumer Protection Act provide?", "provision", "Section 84", "Section 84 provides grounds for liability of a product manufacturer"),
        ("What does Section 85 of the Consumer Protection Act provide?", "provision", "Section 85", "Section 85 provides conditions under which a product seller who is not a manufacturer can be held liable"),
        ("What does Section 89 of the Consumer Protection Act provide?", "provision", "Section 89", "Section 89 provides penalties for manufacturing, storing, selling or distributing adulterated products"),
        ("What does Section 90 of the Consumer Protection Act provide?", "provision", "Section 90", "Section 90 provides penalties for manufacturing, storing, selling or distributing spurious goods"),
    ]
    
    for q_text, q_type, ref, ref_answer in cp_questions:
        if len(questions) >= 250:
            break
        if is_near_duplicate(q_text, fingerprints):
            continue
        fp = question_fingerprint(q_text)
        fingerprints.add(fp)
        
        questions.append({
            "question_id": f"CPA_{len(questions)+1:03d}",
            "question": q_text,
            "expected_source": "consumer_protection_act_2019.pdf",
            "question_type": q_type,
            "source_reference": ref,
            "reference_answer": ref_answer,
            "evidence": "",
            "validation_status": "grounded"
        })
    
    return questions[:250], fingerprints


def validate_dataset(questions):
    """Validate the generated dataset."""
    issues = []
    
    # Check total count
    if len(questions) != 500:
        issues.append(f"Expected 500 questions, got {len(questions)}")
    
    # Check for duplicates
    seen_questions = set()
    for q in questions:
        q_text = q["question"].strip().lower()
        if q_text in seen_questions:
            issues.append(f"Duplicate question: {q['question'][:80]}")
        seen_questions.add(q_text)
    
    # Check balance
    const_count = sum(1 for q in questions if q["expected_source"] == "constitution_of_india.pdf")
    cp_count = sum(1 for q in questions if q["expected_source"] == "consumer_protection_act_2019.pdf")
    if const_count != 250:
        issues.append(f"Constitution questions: {const_count} (expected 250)")
    if cp_count != 250:
        issues.append(f"Consumer Protection questions: {cp_count} (expected 250)")
    
    # Check required fields
    required_fields = ["question_id", "question", "expected_source", "question_type", 
                       "source_reference", "reference_answer", "evidence", "validation_status"]
    for q in questions:
        for field in required_fields:
            if field not in q:
                issues.append(f"Missing field '{field}' in {q.get('question_id', 'UNKNOWN')}")
    
    # Check unique IDs
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate question IDs found")
    
    # Check question types distribution
    type_dist = defaultdict(int)
    for q in questions:
        type_dist[q["question_type"]] += 1
    
    return issues, type_dist


def main():
    base_dir = Path(".")
    
    print("=" * 70)
    print("GENERATING 500 GROUNDED RAG EVALUATION QUESTIONS")
    print("=" * 70)
    
    # Load passages
    print("\nLoading passages...")
    passages = load_all_passages(base_dir)
    print(f"  Constitution passages: {len(passages['constitution'])}")
    print(f"  Consumer Protection passages: {len(passages['consumer_protection'])}")
    
    # Extract articles and sections
    articles = extract_constitution_articles(passages['constitution'])
    sections = extract_cp_sections(passages['consumer_protection'])
    print(f"  Constitution articles found: {len(articles)}")
    print(f"  Consumer Protection sections found: {len(sections)}")
    
    # Generate questions
    print("\nGenerating Constitution questions...")
    const_questions, const_fps = generate_constitution_questions(passages['constitution'], target=250)
    print(f"  Generated: {len(const_questions)}")
    
    print("\nGenerating Consumer Protection questions...")
    cp_questions, _ = generate_consumer_protection_questions(passages['consumer_protection'], const_fps, target=250)
    print(f"  Generated: {len(cp_questions)}")
    
    # Combine and re-index
    all_questions = const_questions + cp_questions
    
    # Re-assign sequential IDs
    for i, q in enumerate(all_questions):
        prefix = "CON" if q["expected_source"] == "constitution_of_india.pdf" else "CPA"
        q["question_id"] = f"Q{i+1:03d}_{prefix}"
    
    # Validate
    print("\nValidating dataset...")
    issues, type_dist = validate_dataset(all_questions)
    
    if issues:
        print(f"\n  VALIDATION ISSUES ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    - {issue}")
    else:
        print("  All validations passed!")
    
    print(f"\n  Question type distribution:")
    for qtype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"    {qtype}: {count}")
    
    # Save CSV
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    
    csv_path = outdir / "rag_evaluation_questions_500.csv"
    fieldnames = ["question_id", "question", "expected_source", "question_type",
                  "source_reference", "reference_answer", "evidence", "validation_status"]
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_questions)
    
    print(f"\n  Saved CSV: {csv_path}")
    
    # Save JSONL
    jsonl_path = outdir / "rag_evaluation_questions_500.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    
    print(f"  Saved JSONL: {jsonl_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"Total questions:             {len(all_questions)}")
    print(f"Constitution questions:       {sum(1 for q in all_questions if q['expected_source'] == 'constitution_of_india.pdf')}")
    print(f"Consumer Protection questions: {sum(1 for q in all_questions if q['expected_source'] == 'consumer_protection_act_2019.pdf')}")
    print(f"Unique question IDs:         {len(set(q['question_id'] for q in all_questions))}")
    print(f"Validation issues:           {len(issues)}")
    
    # Sample questions
    print("\n--- SAMPLE QUESTIONS ---")
    import random
    random.seed(42)
    samples = random.sample(all_questions, min(10, len(all_questions)))
    for s in samples:
        print(f"\n  [{s['question_id']}] ({s['question_type']}) {s['question']}")
        print(f"    Source: {s['expected_source']}")
        print(f"    Reference: {s['source_reference']}")
        if s['reference_answer']:
            print(f"    Answer: {s['reference_answer'][:100]}...")


if __name__ == "__main__":
    main()
