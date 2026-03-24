import argparse
import ast
import json
from pathlib import Path
import random
from tqdm import tqdm
from ollama import chat


def parse_args():
    parser = argparse.ArgumentParser(description='Run learning trials with deterministic randomness.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomization (default: 42).')
    return parser.parse_args()



args = parse_args()
random.seed(args.seed)
def load_shuffled_animals(seed, results_dir='results'):
    file_path = Path(results_dir) / f'shuffled_animals_{seed}.txt'
    if not file_path.exists():
        raise FileNotFoundError(f'Shuffled animals file not found: {file_path}')

    file_contents = file_path.read_text(encoding='utf-8').strip()
    animals = ast.literal_eval(file_contents)
    if not isinstance(animals, list) or not all(isinstance(animal, str) for animal in animals):
        raise ValueError(f'Invalid shuffled animals data in {file_path}')

    return animals


def load_messages(seed, results_dir='results'):
    file_path = Path(results_dir) / f'messages_{seed}.json'
    if not file_path.exists():
        raise FileNotFoundError(f'Messages file not found: {file_path}')

    return json.loads(file_path.read_text(encoding='utf-8'))


args = parse_args()
animals = load_shuffled_animals(args.seed)
messages = load_messages(args.seed)
print(animals)
print(messages)

trials = []
correct = 0
for i in range(len(animals)):
    for j in range(i + 1, len(animals)):
        if j != i + 1:
            trials.append((animals[i], animals[j]))

for data in tqdm(trials, desc="Trials"):
    order = random.choice([0, 1])
    animal1, animal2 = data if order == 0 else (data[1], data[0])
    temp_message = messages.copy()
    temp_message.append({'role': 'user', 'content': f'{animal1} {animal2}'})

    response = chat(
        model='kimi-k2.5:cloud',
        messages=temp_message,
    )

    answer = response.message.content.strip().lower()
    if answer == data[0].lower():
        correct += 1
    else:
        print(f'Incorrect: {data[0]} vs {data[1]} - Model answered: {answer}')

print(f'Correct: {correct} out of {len(trials)}')