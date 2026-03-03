import random

from ollama import chat
from pretty_chat import print_chat

animals = ['whale', 'elephant', 'horse', 'dog', 'duck', 'mouse', 'ant']
correct = 0

for trial in range(15):

    print('Trial', trial + 1)
    animal1, animal2 = random.sample(animals, 2)
    user_content = ['Consider the following relationships between abstract elements:\n\n'+
                    'whale is bigger than elephant, elephant is bigger than horse, horse is bigger than dog, dog is bigger than '+
                    'duck, duck is bigger than mouse, and mouse is bigger than ant\n\n'+
                    f'Now, between the elements {animal1} and {animal2}, which is bigger? Answer only with the element and not other text']

    response = chat(
        model='kimi-k2.5:cloud',
        messages=[{'role': 'user', 'content': user_content[0]}],
    )

    # print_chat(user_content, [response.message.content])

    animal1_index = animals.index(animal1)
    animal2_index = animals.index(animal2)

    if (animal1_index < animal2_index and response.message.content.strip().lower() == animal1) or \
       (animal2_index < animal1_index and response.message.content.strip().lower() == animal2):
        correct += 1

print(f'Correct: {correct} out of {trial + 1} trials, Accuracy: {correct / (trial + 1):.2f}')