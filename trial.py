import random
from tqdm import tqdm
from ollama import chat
from pretty_chat import print_chat


def no_four_consecutive_trials(trials):
    for i in range(len(trials) - 3):
        if trials[i] == trials[i + 1] == trials[i + 2] == trials[i + 3]:
            return False
    return True

animals = ['house', 'water', 'ball', 'baby', 'fish', 'tree', 'car'] 
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

    response = chat(
        model='kimi-k2.5:cloud',
        messages=messages,
    )

    messages.append(response.message)

    answer = response.message.content.strip().lower()
    if answer == data[0].lower():
        correct += 1
        messages.append({'role': 'user', 'content': 'Correct'})
    else:
        messages.append({'role': 'user', 'content': 'Incorrect'})

    response = chat(
        model='kimi-k2.5:cloud',
        messages=messages,
    )

    messages.append(response.message)

    while response.message.content.strip() != '[Next trial]':
        messages.append({'role': 'user', 'content': 'Please respond with "[Next trial]" to proceed to the next trial.'})
        response = chat(
            model='kimi-k2.5:cloud',
            messages=messages,
        )
        messages.append(response.message)
print_chat(messages)