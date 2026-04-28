import argparse
import json
import os
import random
from tqdm import tqdm
from ollama import chat
from pretty_chat import print_chat


def parse_args():
    parser = argparse.ArgumentParser(description='Run learning trials with deterministic randomness.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomization (default: 42).')
    return parser.parse_args()


args = parse_args()
random.seed(args.seed)


def no_four_consecutive_trials(trials):
    for i in range(len(trials) - 3):
        if trials[i] == trials[i + 1] == trials[i + 2] == trials[i + 3]:
            return False
    return True


def serialize_message(message):
    if isinstance(message, dict):
        return {
            'role': message.get('role', ''),
            'content': message.get('content', ''),
        }
    return {
        'role': getattr(message, 'role', ''),
        'content': getattr(message, 'content', str(message)),
    }


def write_messages(messages, seed):
    serialized_messages = [serialize_message(message) for message in messages]
    with open(f'results/memory_messages_{seed}.json', 'w', encoding='utf-8') as output_file:
        json.dump(serialized_messages, output_file, indent=2, ensure_ascii=True)

animals = ['house', 'water', 'ball', 'baby', 'fish', 'tree', 'car'] 
random.shuffle(animals)
os.makedirs('results', exist_ok=True)
with open(f'results/shuffled_animals_{args.seed}.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(str(animals))

correct = 0
mem_length = 100

messages = [{'role': 'user', 
             'content': 'You are a learning agent that is trying to learn the relationship between pairs of words. ' +
             'You do not have any memory. You context will be erased after every interation. ' +
             f'However, you do have a very short term memory that is a string of {mem_length} alpha-numeric characters. '+
             f'At the start the memory will be {"0"*mem_length}. You will participate in a number of trials. '+
             'In each trial you will be presented with exactly two words and need to select either the first word or second. '+
             f'You will then receive an indication of whether you were correct or not, after which you will respond with a string of {mem_length} '+
             'letters of alpha-numeric characters that will serve as your memory for future inputs.The goal is to learn the relationship'+
             ' between the words as quickly as possible. Remember, during the trial respond with only the first word or the second word '+
             'and no other text. Try to use the memory to keep track of any information that might be useful for learning the relationship between the words. '+
             'Keep in mind that parts of old memory might be worth keeping if it contains useful information, but you can also overwrite it with new information if you think that would be more useful for future trials.'}]

working_messages = messages.copy()
initial_message = messages.copy()
memory = "0" * mem_length


trials = []

for i in range(len(animals) - 1):
    trials.append((animals[i], animals[i + 1]))


training_data = trials * 10  # Repeat the trials to provide more training examples
random.shuffle(training_data)  # Shuffle the training data to ensure random order

while not no_four_consecutive_trials(training_data):
    random.shuffle(training_data)

for data in tqdm(training_data, desc="Trials"):

    order = random.choice([0, 1])
    animal1, animal2 = data if order == 0 else (data[1], data[0])

    
    messages.append({'role': 'user', 'content': f'{animal1} {animal2}, Memory: {memory}'})
    working_messages.append({'role': 'user', 'content': f'{animal1} {animal2}, Memory: {memory}'})

    response = chat(
        model='kimi-k2.5:cloud',
        messages=working_messages,
    )

    messages.append(response.message)
    working_messages.append(response.message)
    answer = response.message.content.strip().lower()
    print_chat(working_messages)
    if answer == data[0].lower():
        correct += 1
        messages.append({'role': 'user', 'content': 'Correct'})
        working_messages.append({'role': 'user', 'content': 'Correct'})
    else:
        messages.append({'role': 'user', 'content': 'Incorrect'})
        working_messages.append({'role': 'user', 'content': 'Incorrect'})

    response = chat(
        model='kimi-k2.5:cloud',
        messages=messages,
    )

    memory = response.message.content.strip()[:mem_length]  # Update memory with the first 7 characters of the response
    messages.append(response.message)
    working_messages = initial_message.copy()  # Reset working messages to the initial message for the next trial
    write_messages(messages, args.seed)
   
        
print_chat(messages)

