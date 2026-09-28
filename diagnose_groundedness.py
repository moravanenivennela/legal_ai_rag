from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

print("Label mapping (id2label):", model.model.config.id2label)
print()

# Test 1: an obvious, direct entailment case
context = "The Consumer Protection Act, 2019 recognizes six consumer rights including the right to safety and the right to information."
hypothesis_true = "The Act recognizes the right to safety and the right to information."
hypothesis_false = "The Act says nothing about consumer rights."

scores_true = model.predict([(context, hypothesis_true)])
scores_false = model.predict([(context, hypothesis_false)])

print("Should be HIGH entailment:", scores_true)
print("Should be LOW entailment:", scores_false)
