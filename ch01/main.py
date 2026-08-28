import os

from dotenv import load_dotenv
load_dotenv()

from llm.chat_model import DeepSeekChat

chat = DeepSeekChat(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL")
)

def agent_loop(messages: list):
    while True:
        in_reasoning = False
        for kind, delta in chat.stream(
            messages=messages,
            model="deepseek-v4-pro",
            thinking=True
        ):
            if kind == 'reasoning':
                if not in_reasoning:
                    print('\n\033[94m-----------------[思考]------------------\033[0m\n')
                in_reasoning = True
            else:
                if in_reasoning:
                    print('\n\n\033[94m-----------------[回答]------------------\033[0m\n')
                in_reasoning = False
            print(delta, end='')
        print()

        if chat.finish_reason == 'stop':
            break

def main():
    print('====================================')
    print('=             Mini Agent           =')
    print('====================================')
    messages = []
    while True:
        query = input('\033[94m[USER] > \033[0m')
        messages.append({
            'role': 'user',
            'content': query
        })
        agent_loop(messages=messages)

if __name__ == '__main__':
    main()
