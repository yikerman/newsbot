import os
import json
from typing import Iterable

import curl_cffi
import markdownify
import openai
from dotenv import load_dotenv

NEWS_FETCHER_INITIAL_PROMPT = """# 身份: 新闻内容获取机器人

你是一个专业、高效的新闻内容获取机器人。你的主要任务是提取用户提供网页中的重要新闻URL。
严格遵守：准确提取，确保提取的URL是有效的新闻链接，避免提取无关的链接。

## 输入格式:
你将收到一个转换为markdown格式的网页内容。

## 输出格式:
请严格按照json数组格式输出提取的新闻URL列表：

[
    "https://example.com/news1",
    "https://example.com/news2",
    ...
]
"""

NEWS_AGENT_INITIAL_PROMPT = """# 身份: 资深客观新闻聚合/编辑机器人

你是一个专业、高效、中立的资深新闻编辑机器人。你的主要任务是调用工具获取国外新闻，翻译到中文并进行客观、简洁的总结。你不表达个人观点，只陈述事实。

## 严格遵守:
1. **绝对客观**：保持中立态度，只陈述事实，**绝不**在总结中加入你的个人观点、评价或情感倾向。
2. **杜绝幻觉**：只基于用户提供的文本内容进行总结。如果提供的信息不完整，不要自行脑补或编造细节。

## 输入格式:
你将收到一个转换为markdown格式的网页内容。

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
"""

def get_webpage_markdown(url: str) -> str:
    print(f"[debug] get_webpage_markdown({url})")
    response = curl_cffi.get(url, impersonate="chrome")
    if response.status_code != 200:
        return f"Error fetching webpage: {response.status_code}"
    html_content = response.text
    markdown_content = markdownify.markdownify(html_content, heading_style="ATX")
    return markdown_content

def call_llm(instructions: str, input: str):
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    response = client.responses.create(
        model=os.getenv("MODEL", "gpt-5.2"),
        instructions=instructions,
        input=input,
        temperature=0.2,
    )
    return response



def extract_news_urls(markdown_content: str) -> Iterable[str]:
    response = call_llm(NEWS_FETCHER_INITIAL_PROMPT, markdown_content)
    try:
        news_urls = json.loads(response.output_text)
        if isinstance(news_urls, list):
            return news_urls
        else:
            raise ValueError(f"LLM response is not a list: {response.output_text}")
    except json.JSONDecodeError:
        raise ValueError(f"LLM response is not valid JSON: {response.output_text}")

def summarize_news(markdown_content: str) -> str:
    response = call_llm(NEWS_AGENT_INITIAL_PROMPT, markdown_content)
    return response.output_text

def main():
    load_dotenv()
    APNEWS_HOMEPAGE = "https://apnews.com/"
    homepage_markdown = get_webpage_markdown(APNEWS_HOMEPAGE)
    news_urls = extract_news_urls(homepage_markdown)
    print(f"[debug] Extracted news URLs: {news_urls}")
    for url in news_urls:
        news_markdown = get_webpage_markdown(url)
        summary = summarize_news(news_markdown)
        print(summary)

if __name__ == "__main__":
    main()
