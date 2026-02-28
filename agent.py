# Context爆炸，没法用

import os
import json
from typing import Iterable

import curl_cffi
import markdownify
import openai
from openai.types.responses.tool_param import ToolParam
from dotenv import load_dotenv

NEWS_AGENT_INITIAL_PROMPT = """# 身份: 资深客观新闻聚合/编辑机器人

你是一个专业、高效、中立的资深新闻编辑机器人。你的主要任务是调用工具获取国外新闻，翻译到中文并进行客观、简洁的总结。你不表达个人观点，只陈述事实。
用户会提供新闻门户网站，你需要从中提取新闻，通过工具获取每段新闻网页的内容。

## 严格遵守:
1. **绝对客观**：保持中立态度，只陈述事实，**绝不**在总结中加入你的个人观点、评价或情感倾向。
2. **杜绝幻觉**：只基于用户提供的文本内容进行总结。如果提供的信息不完整，不要自行脑补或编造细节。

## 输出格式:
请严格按照以下结构输出你的总结：

**来源**: [新闻来源，如CNN、BBC等，附上链接]
**标题**：[生成一个精炼的客观标题]
**核心摘要**：[用一两句话（50字以内）概括这篇新闻最重要的事情]
**关键要点**：
 - [要点 1：如具体数据、重要决定等]
 - [要点 2：如相关方的回应、背景等]
 - [要点 3：如后续影响、未来规划等]
 - [按需添加更多要点]

对每篇新闻都要单独输出一段总结。
"""

TOOLS: Iterable[ToolParam] = [
    {
        "type": "function",
        "name": "get_webpage_markdown",
        "description": "Fetch a webpage and return its content in markdown format.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to fetch",
                },
            },
            "required": ["url"],
        },
    },  # pyright: ignore[reportAssignmentType]
]


def get_webpage_markdown(url: str) -> str:
    print(f"[debug] get_webpage_markdown({url})")
    response = curl_cffi.get(url, impersonate="chrome")
    if response.status_code != 200:
        return f"Error fetching webpage: {response.status_code}"
    html_content = response.text
    markdown_content = markdownify.markdownify(html_content, heading_style="ATX")
    return markdown_content


class Agent:
    def __init__(self):
        self.provider = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.messages = []
        self.model_name = os.getenv("MODEL", "gpt-4o")

        self.messages.append({"role": "system", "content": NEWS_AGENT_INITIAL_PROMPT})

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        response = self.call_model()
        have_toolcall = True
        while have_toolcall:
            have_toolcall = False
            for item in response.output:
                if item.type == "function_call":
                    have_toolcall = True
                    if item.name == "get_webpage_markdown":
                        url = json.loads(item.arguments)
                        markdown_content = get_webpage_markdown(url["url"])
                        self.messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps({"content": markdown_content}),
                            }
                        )
            if have_toolcall:
                response = self.call_model()

    def call_model(self):
        response = self.provider.responses.create(
            model=self.model_name, tools=TOOLS, input=self.messages
        )
        self.messages += response.output
        return response


def main():
    load_dotenv()
    agent = Agent()
    agent.chat("请帮我总结一下今日apnews头条：https://apnews.com/")
    print(agent.messages[-1]["content"])


if __name__ == "__main__":
    main()
