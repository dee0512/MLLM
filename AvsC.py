from ollama import chat
from pretty_chat import print_chat

user_content = ['Consider the following relashionships between abstract elements: [A, B, C]\n\n'+
               '- A>B\n'+
               '- B>C\n\n'+
               'Now between elements A and C, which is greater? Answer only with the element and not other text']

response = chat(
    model='kimi-k2.5:cloud',
    messages=[{'role': 'user', 'content': user_content[0]}],
)



print_chat([
    {'role': 'user', 'content': user_content[0]},
    {'role': 'assistant', 'content': response.message.content},
])