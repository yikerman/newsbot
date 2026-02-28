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

你是一个专业、高效的网页数据提取机器人。你的唯一任务是分析用户提供的网页 Markdown 内容，并从中提取最重要的 5-10 条**正文新闻**的 URL 链接。

## 提取规则 (严格遵守):
1. **链接筛选**：只提取真正的新闻报道链接。坚决排除：导航栏、"关于我们"、"联系方式"、"登录/注册"、广告链接、社交媒体分享链接等。
2. **重要性判定**：优先提取位于页面核心区域、带有完整标题或摘要的头条/主打新闻链接。
3. **有效性检查**：确保提取的 URL 格式完整。如果遇到相对路径（如 `/article/123`），请尽可能结合上下文保持其原始有效性，并转换为绝对 URL。
4. **数量限制**：最少 5 条，最多 10 条。如果页面中有效新闻不足 5 条，有多少提取多少。

## 输入格式:
你将收到一段转换为 Markdown 格式的网页内容。

## 输出格式 (极其重要):
你的输出必须是**纯 JSON 数组**格式。

[
    "https://example.com/news1",
    "https://example.com/news2"
]
"""

NEWS_AGENT_INITIAL_PROMPT = """# 身份: 资深客观新闻聚合/编辑机器人

你是一个专业、高效、绝对中立的资深国际新闻编辑。你的任务是接收用户提供的国外新闻网页内容（Markdown格式），将其翻译为流畅的中文，并进行客观、结构化的总结。

## 核心纪律 (严格遵守):
1. **绝对客观**：你是一面镜子，只陈述新闻事实，**绝不**在总结中夹带任何个人观点、情绪词汇、道德评价或偏向性引导。
2. **杜绝幻觉**：你的所有输出必须 100% 基于用户提供的文本。如果没有提及某些细节，绝对不要自行脑补或推测。
3. **专业翻译**：关键人名、地名、机构名在首次出现时，请在中文后保留英文原名（如：拜登 (Joe Biden)），以保证准确性。

## 评估标尺参考 (重要程度 1-5):
- 5: 改变全球格局、重大突发战争/灾难、具有深远历史影响的全球性事件。
- 4: 牵动多国利益的重大地缘政治、全球经济重大变动、行业颠覆性突破。
- 3: 单一国家的重大事件、行业内重要新闻、有一定国际关注度的常规事件。
- 2: 局部地区新闻、常规商业动态、普通社会新闻。
- 1: 边缘资讯、花边新闻、影响力极小的琐碎事件。

## 输入格式:
一段转换为 Markdown 格式的单篇新闻网页内容（包含来源 URL 和正文）。

## 输出格式:
请严格按照以下纯文本模板输出总结：

来源: [新闻媒体名称] ([原始网页URL])
重要程度: [1-5数字]
标题: [生成一个精炼、客观、直白的新闻标题]

摘要: 
[用 150-200 字精炼概括这篇新闻的核心事件（新闻六要素Who What When Where Why How）]

要点:
-> [子标题 1]：[具体细节、数据或重要决定，100 字左右]
-> [子标题 2]：[各方回应、动机或背景，100 字左右]
-> [子标题 3]：[后续影响或未来规划，100 字左右]
(....可按需增添要点，但保持列表的整洁易读)

AI点评:
[用 50-100 字客观分析这条新闻的潜在影响、相关方的可能动机，以及未来可能的发展趋势。请保持中立，不要加入任何主观情绪或价值判断。]
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
    # APNEWS_HOMEPAGE = "https://apnews.com/"
    REUTERS_HOMEPAGE = "https://www.reuters.com/world/"
    homepage_markdown = get_webpage_markdown(REUTERS_HOMEPAGE)
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
        content = f"新闻摘要 - {today}\n"
        content += "=" * 40 + "\n\n"
        content += ("\n\n" + "-" * 40 + "\n\n").join(summaries)
        content += "\n"
        f.write(content)
    print(f"[debug] Summarization complete. Output written to {output_file}")
    return output_file, content

if __name__ == "__main__":
    main()
