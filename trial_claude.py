import argparse
import json
import os
import random
from tqdm import tqdm
from anthropic import Anthropic
from pretty_chat import print_chat


def parse_args():
    parser = argparse.ArgumentParser(description='Run learning trials with deterministic randomness.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomization (default: 42).')
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-latest', help='Claude model to use.')
    return parser.parse_args()


args = parse_args()
random.seed(args.seed)
api_key = ''
if not api_key:
    raise RuntimeError('ANTHROPIC_API_KEY is not set. Please set it before running this script.')

client = Anthropic(api_key=api_key)


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


def call_claude(client, messages, model):
    system_prompts = []
    claude_messages = []

    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        if role == 'system':
            system_prompts.append(content)
        elif role in {'user', 'assistant'}:
            claude_messages.append({'role': role, 'content': content})
        else:
            claude_messages.append({'role': 'user', 'content': content})

    request = {
        'model': model,
        'max_tokens': 64,
        'messages': claude_messages,
    }
    if system_prompts:
        request['system'] = '\n\n'.join(system_prompts)

    response = client.messages.create(**request)
    text = ''.join(block.text for block in response.content if getattr(block, 'type', '') == 'text').strip()
    return {'role': 'assistant', 'content': text}

animals = ['house', 'water', 'ball', 'baby', 'fish', 'tree', 'car'] 
random.shuffle(animals)
os.makedirs('results', exist_ok=True)
with open(f'results/shuffled_animals_{args.seed}.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(str(animals))

correct = 0
messages = [{'role': 'user', 
             'content': 'You are a learning agent that is trying to learn the relationship between pairs of words. You will participate in a number of trials. In each trial you will be presented with exactly two words and need to select either the first word or second. You will then receive an indication of whether you were correct or not, after which you will respond with "[Next trial]". The goal is to learn the relationship between the words as quickly as possible. Remember, during the trial respond with only the first word or the second word and no other text.'}]



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
    messages.append({'role': 'user', 'content': f'{animal1} {animal2}'})

    response_message = call_claude(client=client, messages=messages, model=args.model)

    messages.append(response_message)

    answer = response_message['content'].strip().lower()
    if answer == data[0].lower():
        correct += 1
        messages.append({'role': 'user', 'content': 'Correct'})
    else:
        messages.append({'role': 'user', 'content': 'Incorrect'})

    response_message = call_claude(client=client, messages=messages, model=args.model)

    messages.append(response_message)

    while response_message['content'].strip() != '[Next trial]':
        messages.append({'role': 'user', 'content': 'Please respond with "[Next trial]" to proceed to the next trial.'})
        response_message = call_claude(client=client, messages=messages, model=args.model)
        messages.append(response_message)

        
print_chat(messages)

serialized_messages = [serialize_message(message) for message in messages]
with open(f'results/messages_{args.seed}.json', 'w', encoding='utf-8') as output_file:
    json.dump(serialized_messages, output_file, indent=2, ensure_ascii=True)

