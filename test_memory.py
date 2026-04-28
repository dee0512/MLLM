import argparse
import ast
import csv
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
    file_path = Path(results_dir) / f'1{seed}.json'
    if not file_path.exists():
        raise FileNotFoundError(f'Messages file not found: {file_path}')

    return json.loads(file_path.read_text(encoding='utf-8'))


args = parse_args()
animals = load_shuffled_animals(args.seed)
memory = "".join(animals)
mem_length = len(memory)
thinking_enabled = True

messages = messages = [{'role': 'user', 
             'content': 'You are a learning agent that is trying to learn the relationship between pairs of words. ' +
             'You do not have any memory. You context will be erased after every interation. ' +
             f'However, you do have a very short term memory that is a string of {mem_length} alpha-numeric characters. '+
             f'At the start the memory will be {"0"*mem_length}. You will participate in a number of trials. '+
             'In each trial you will be presented with exactly two words and need to select either the first word or second. '+
             f'You will then receive an indication of whether you were correct or not, after which you will respond with a string of {mem_length} '+
             'letters of alpha-numeric characters that will serve as your memory for future inputs.The goal is to learn the relationship'+
             ' between the words as quickly as possible. Remember, during the trial respond with only the first word or the second word '+
             'and no other text. Try to use the memory to keep track of any information that might be useful for learning the relationship between the words. '+
             'Keep in mind that parts of old memory might be worth keeping if it contains useful information, but you can also overwrite it with new information '+
             'if you think that would be more useful for future trials.'}]
print(animals)
print(messages)
print(memory)
trials = []
correct = 0
trial_rows = []
for i in range(len(animals)):
    for j in range(i + 1, len(animals)):
        if j != i + 1:
            trials.append((animals[i], animals[j]))


for order in [0, 1]:
    for data in tqdm(trials, desc="Trials"):
        animal1, animal2 = data if order == 0 else (data[1], data[0])
        temp_message = messages.copy()
        temp_message.append({'role': 'user', 'content': f'{animal1} {animal2}, Memory: {memory}'})

        response = chat(
            model='kimi-k2.5:cloud',
            messages=temp_message,
            options={"think": thinking_enabled}
        )

        answer = response.message.content.strip().lower()
        expected = data[0].lower()
        is_correct = answer == expected
        thinking_output = getattr(response.message, 'thinking', '') if thinking_enabled else ''

        row = {
            'order': order,
            'animal_1': animal1,
            'animal_2': animal2,
            'expected_answer': data[0],
            'model_answer': response.message.content.strip(),
            'result': 'Correct' if is_correct else 'Incorrect',
            'memory': memory,
        }
        if thinking_enabled:
            row['thinking'] = thinking_output

        trial_rows.append(row)

        if is_correct:
            correct += 1
        else:
            print(f'Incorrect: {data[0]} vs {data[1]} - Model answered: {answer}')

print(f'Correct: {correct} out of {len(trials) * 2} trials')

output_path = Path('results') / f'trial_results_{args.seed}.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    'order',
    'animal_1',
    'animal_2',
    'expected_answer',
    'model_answer',
    'result',
    'memory',
]
if thinking_enabled:
    fieldnames.append('thinking')

with output_path.open('w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(trial_rows)

print(f'Saved trial results to: {output_path}')