from dotenv import load_dotenv
load_dotenv()

import requests
import json
import logging

from tools.structed_tool import ToolCall

logger = logging.getLogger(__name__)

options = {
  # "model": "deepseek-v4-pro",
  # "thinking": {
  #   "type": "enabled"
  # },
  "reasoning_effort": "low",
  # "max_tokens": 4096,
  "response_format": {
    "type": "text"
  },
  "stop": None,
  # "stream": False,
  "stream_options": None,
  "temperature": 1,
  "top_p": 1,
  # "tools": None,
  # "tool_choice": "none",
  "logprobs": False,
  "top_logprobs": None
}

class DeepSeekLLM:
    def __init__(self,
                 api_key: str = None,
                 base_url: str = None
        ):
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.base_url = base_url
        self.api_key = api_key
        # 输出槽位：每次请求后记录最近一次结果（messages 由外部维护，不存实例）
        self.finish_reason = None
        self.tool_calls: list[ToolCall] = []

    def _build_options(self,
                       messages: list,
                       thinking: bool,
                       tools: list,
                       max_tokens: int,
                       model: str,
                       stream: bool) -> dict:
        # 组装请求体：以模块默认 options 为底，覆盖本次调用参数，返回新 dict（不污染全局 options）
        payload = dict(options)
        payload.update({
            'stream': stream,
            'thinking': {"type": "enabled" if thinking else "disabled"},
            'tools': tools or [],
            'tool_choice': "auto",
            'max_tokens': max_tokens,
            'model': model,
            'messages': messages,
        })
        return payload

    def invoke(self,
               messages: list,
               thinking: bool = False,
               tools: list = None,
               max_tokens: int = 4096,
               model: str = "deepseek-chat") -> dict:
        # 无状态非流式请求：messages 由外部传入，返回解析后的响应 dict
        payload = self._build_options(messages, thinking, tools, max_tokens, model, stream=False)

        response = requests.request("POST", self.base_url, headers=self.headers, data=json.dumps(payload))
        data = response.json()

        id = data['id']
        choices = data['choices']
        message = choices[0]['message']
        role = message['role']
        content = message['content']
        reasoning_content = message.get('reasoning_content', None)
        # 如果 tool_calls 存在，则取出 message['tool_calls']，否则为 None
        tool_calls = message['tool_calls'] if 'tool_calls' in message else None
        finish_reason = choices[0]['finish_reason']
        logger.info('收到响应: id=%s, finish_reason=%s', id, finish_reason)

        self.finish_reason = finish_reason

        return {
            'id': id,
            'role': role,
            'content': content,
            'reasoning_content': reasoning_content,
            'tool_calls': tool_calls,
            'finish_reason': finish_reason
        }

    def stream(self,
               messages: list,
               thinking: bool = False,
               tools: list = None,
               max_tokens: int = 4096,
               model: str = None):
        # 无状态流式请求：messages 由外部传入，逐段 yield ('reasoning'|'content', delta)
        payload = self._build_options(messages, thinking, tools, max_tokens, model, stream=True)
        response = requests.request("POST", self.base_url, headers=self.headers, data=json.dumps(payload), stream=True)

        if response.status_code == 400:
            resp_json = response.json()
            error = resp_json.get('error')
            print(f"\033[31mmessage: {error.get('message')}")
            print(f"type: {error.get('type')}")
            print(f"param: {error.get('param')}")
            print(f"code: {error.get('code')}\033[0m")

        response.raise_for_status()

        reasoning_content = ''
        content = ''
        tool_calls = []

        try:
            for raw in response.iter_lines():
                line = raw.decode('utf-8')
                # print(line)
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                choice = chunk.get('choices', [{}])[0]
                delta = choice.get('delta', {})
                self.finish_reason = choice.get('finish_reason', '')

                reasoning = delta.get('reasoning_content')
                if reasoning:
                    reasoning_content += reasoning
                    yield ('reasoning', reasoning)
                    continue

                piece = delta.get('content')
                if piece:
                    content = content + piece
                    yield ('content', piece)

                t_tool_calls = delta.get('tool_calls', None)
                if t_tool_calls:
                    for item in t_tool_calls:
                        i = item['index']
                        if len(tool_calls) > i:
                            tool_calls[i]['function']['arguments'] += item['function']['arguments']
                        else:
                            tool_calls.append(item)
        finally:
            self.tool_calls = []
            for tool_call in tool_calls:
                self.tool_calls.append(ToolCall(
                    tool_id=tool_call['id'],
                    name=tool_call['function']['name'],
                    tool_input=json.loads(tool_call['function']['arguments']),
                ))


# if __name__ == "__main__":
#     chat = DeepSeekChat(
#         api_key=os.environ.get("DEEPSEEK_API_KEY"),
#         base_url=os.environ.get("DEEPSEEK_BASE_URL")
#     )
#     messages = [{"role": "user", "content": "帮我查询广州海珠区今日的天气。"}]
#     in_reasoning = False
#     for kind, delta in chat.stream(
#         messages=messages,
#         model="deepseek-v4-pro",
#         thinking=True
#     ):
#         if kind == 'reasoning':
#             if not in_reasoning:
#                 print('\n-----------------[思考]------------------\n')
#             in_reasoning = True
#         else:
#             if in_reasoning:
#                 print('\n\n-----------------[回答]------------------\n')
#             in_reasoning = False
#         print(delta, end='')
    # print()
    # print(chat.tool_calls)
    #
    # response = chat.invoke(messages)
    # print(response['tool_calls'])