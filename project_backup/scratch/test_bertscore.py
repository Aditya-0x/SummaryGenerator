import evaluate
import torch

bertscore = evaluate.load("bertscore")
predictions = ["hello world"]
references = ["hello world"]
results = bertscore.compute(predictions=predictions, references=references, lang="en")
print(results)
