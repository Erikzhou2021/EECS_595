import json

file_path = "BBQ_Gender_identity_small_sample.jsonl"  # Replace with your actual file path

data = []
answer_dict = {0: 'ans0', 1: 'ans1', 2: 'ans2'}
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)  # Parse each line as a JSON object
        context = entry["context"]  # Get 'context' field
        question = entry["question"]  # Get 'question' field
        true_label = entry["label"]  # Get true 'label' field
        
        input = context + " " + question
        output = entry[answer_dict[true_label]] # Get correct answer
        data.append((input, output))
        
for x in range(5):
    print(data[x])