import random
import string

from ollama import chat
from pretty_chat import print_chat



def random_string(length: int = 10) -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=length))


animals: list[str] = []
while len(animals) < 7:
    token = random_string(10)
    if token not in animals:
        animals.append(token)

correct = 0

for trial in range(15):

    print('Trial', trial + 1)
    animal1, animal2 = random.sample(animals, 2)
    chain_text = ', '.join(
        f'{animals[i]} is bigger than {animals[i + 1]}'
        for i in range(len(animals) - 1)
    )
    user_content = ['Consider the following relationships between abstract elements:\n\n'+
                    chain_text + '\n\n'+
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