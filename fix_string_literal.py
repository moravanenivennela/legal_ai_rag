with open("rag_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

broken = 'sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 15]'
fixed = 'sentences = [s.strip() for s in answer.replace(chr(10), " ").split(".") if len(s.strip()) > 15]'

if broken in content:
    content = content.replace(broken, fixed)
    with open("rag_engine.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed.")
else:
    print("Pattern not found via exact match, trying line-based fix...")
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if 'sentences = [s.strip() for s in answer.replace("' in line and i + 1 < len(lines):
            # Merge this line with the next one (the literal newline split)
            merged = line + lines[i + 1]
            lines[i] = merged
            del lines[i + 1]
            content = "\n".join(lines)
            with open("rag_engine.py", "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Merged lines {i+1} and {i+2}, saved.")
            break
    else:
        print("Could not locate the broken line automatically.")
