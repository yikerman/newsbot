import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable

import curl_cffi
import markdownify
import openai
from dotenv import load_dotenv

NEWS_FETCHER_INITIAL_PROMPT = """# 身份: 新闻内容获取机器人

你是一个专业、高效的新闻内容获取机器人。你的主要任务是提取用户提供网页中的新闻URL，选取最重要的5-10条新闻链接。
严格遵守：确保提取的URL是有效的新闻链接，避免提取无关的链接。

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

**来源**: [新闻来源，如CNN、BBC] [网页URL]
**重要程度**: [用1-5的数字评估这条新闻于世界的重要程度，5为最重要]
**标题**：[生成一个精炼的客观标题]
**核心摘要**：[200字以内概括这篇新闻最重要的事情]
**关键要点**：
 - [要点 1：如具体数据、重要决定等]
 - [要点 2：如相关方的回应、背景等]
 - [要点 3：如后续影响、未来规划等]
 - [按需添加更多要点，每个要点100字以内]
"""


def get_webpage_markdown(url: str) -> str:
    print(f"[debug] get_webpage_markdown({url})")
    response = curl_cffi.get(url, impersonate="chrome")
    if response.status_code != 200:
        return f"Error fetching webpage: {response.status_code}"
    html_content = response.text
    markdown_content = markdownify.markdownify(html_content, heading_style="ATX")
    return markdown_content


def call_llm(messages, **kwargs):
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-5.2"),
        messages=messages,
        temperature=0,
        reasoning_effort="low",
        **kwargs,
    )
    return response


def extract_news_urls(markdown_content: str) -> Iterable[str]:
    messages = [
        {"role": "system", "content": NEWS_FETCHER_INITIAL_PROMPT},
        {"role": "user", "content": markdown_content},
    ]
    response = call_llm(messages, response_format={"type": "json_object"})
    try:
        news_urls = json.loads(response.choices[0].message.content)  # type: ignore
        if isinstance(news_urls, list):
            return news_urls
        else:
            raise ValueError(f"LLM response is not a list: {response.choices[0].message.content}")  # type: ignore
    except json.JSONDecodeError:
        raise ValueError(f"LLM response is not valid JSON: {response.choices[0].message.content}")  # type: ignore


def summarize_news(markdown_content: str) -> str:
    messages = [
        {"role": "system", "content": NEWS_AGENT_INITIAL_PROMPT},
        {"role": "user", "content": markdown_content},
    ]
    response = call_llm(messages)
    return response.choices[0].message.content  # type: ignore


def is_today_news(markdown_content: str) -> bool:
    now = datetime.now()
    formatted_date = now.strftime("%B %d, %Y")  # APNews/Reuters的日期格式
    if formatted_date in markdown_content:
        return True
    if "LIVE" in markdown_content:
        return True
    return False


def process_news_url(url: str) -> str | None:
    """Fetch, filter, and summarize a single news URL. Returns summary or None."""
    news_markdown = get_webpage_markdown(url)
    if not is_today_news(news_markdown):
        print(f"[debug] Skipping non-today news: {url}")
        return None
    return summarize_news(news_markdown)


def main():
    load_dotenv()
    APNEWS_HOMEPAGE = "https://apnews.com/"
    homepage_markdown = get_webpage_markdown(APNEWS_HOMEPAGE)
    news_urls = extract_news_urls(homepage_markdown)
    print(f"[debug] Extracted news URLs: {news_urls}")
    summaries: list[str] = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_news_url, url): url for url in news_urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                summary = future.result()
                if summary:
                    summaries.append(summary)
            except Exception as e:
                print(f"[error] Failed to process {url}: {e}")

    today = datetime.now().strftime("%Y-%m-%d")
    output_file = f"news_{today}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"新闻摘要 - {today}\n")
        f.write("=" * 40 + "\n\n")
        f.write(("\n\n" + "-" * 40 + "\n\n").join(summaries))
        f.write("\n")
    print(f"[debug] Summarization complete. Output written to {output_file}")


if __name__ == "__main__":
    main()
