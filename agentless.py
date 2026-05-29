import logging
import os
import json
import random
import time
from typing import List
from datetime import datetime

import curl_cffi
import markdownify
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

NEWS_FETCHER_INITIAL_PROMPT = """# 身份: 新闻内容获取机器人

你是一个专业、高效的网页数据提取机器人。你的唯一任务是分析用户提供的新闻网站首页链接列表，并从中提取最重要的 5-12 条**正文新闻**的 URL 链接。

## 提取规则 (严格遵守):
1. **链接筛选**：只提取真正的新闻报道链接。坚决排除：导航栏、栏目首页、"关于我们"、"联系方式"、"登录/注册"、广告链接、社交媒体分享链接、播客/视频/通讯订阅栏目首页等。
2. **重要性判定**：根据链接文本判断新闻的重要性，优先选择文本看起来像完整新闻标题（包含具体事件、人物、地点等要素）的链接。
3. **数量限制**：最少 5 条，最多 12 条。如果有效新闻不足 5 条，有多少提取多少。
4. **内容多样化**：尽量提取涵盖不同事件的链接，而不是集中在同一事件上。若同一事件有多个报道，只保留最具代表性的一两个链接。

## 输入格式:
你将收到一个链接列表，每行格式为：

```
<链接文本> | <绝对 URL>
```

URL 已经是绝对路径，无需再做转换。

## 输出格式 (极其重要):
你的输出必须是**纯 JSON 数组**格式，每一行为一个绝对（包含协议和域名）的新闻 URL 字符串。例如：

[
    "https://example.com/news1",
    "https://example.com/news2"
]
"""

NEWS_AGENT_INITIAL_PROMPT = """# 身份: 资深客观新闻聚合/编辑机器人

你是一个专业、高效、绝对中立的资深国际新闻编辑。你的任务是接收用户提供的国外新闻网页内容（Markdown格式），将其翻译为流畅、简洁、易懂的中文，并进行客观、结构化的总结。

## 核心纪律 (严格遵守):
1. **绝对客观**：你只陈述新闻事实，**绝不**在总结中夹带任何个人观点、情绪词汇、道德评价或偏向性引导。
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

## 输出格式 (严格遵守 Markdown 语法):
请严格按照以下 Markdown 模板输出总结，不要额外添加代码块包裹（如 ```markdown）：

## [生成一个精炼、客观、直白的新闻标题]

**重要程度**: [1-5数字] / 5

### 摘要

[用 150-200 字精炼概括这篇新闻的核心事件（新闻六要素Who What When Where Why How）]

### 要点

- **[子标题 1]**：[具体细节、数据或重要决定，100 字左右]
- **[子标题 2]**：[各方回应、动机或背景，100 字左右]
- **[子标题 3]**：[后续影响或未来规划，100 字左右]

（可按需增添更多要点）
"""

NOISE_PATH_PREFIXES = (
    "/about",
    "/contact",
    "/career",
    "/careers",
    "/terms",
    "/privacy",
    "/cookie",
    "/accessibility",
    "/help",
    "/faq",
    "/login",
    "/signin",
    "/signup",
    "/subscribe",
    "/newsletter",
    "/tag/",
    "/author/",
    "/topic/",
    "/section/",
)


def extract_homepage_anchors(url: str) -> str:
    """Fetch a news homepage and return a compact `text | absolute_url` list of
    candidate article anchors. Site-agnostic — uses generic filters only.
    """
    logger.debug(f"Fetching homepage anchors: {url}")
    response = curl_cffi.get(url, impersonate="chrome")
    if response.status_code != 200:
        return f"Error fetching webpage: {response.status_code}"
    soup = BeautifulSoup(response.text, "html.parser")
    host = urlparse(url).netloc
    lines: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])  # type: ignore[arg-type]
        parsed = urlparse(href)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != host:
            continue
        if any(parsed.path.startswith(p) for p in NOISE_PATH_PREFIXES):
            continue
        text = " ".join(a.get_text(" ", strip=True).split())
        lines.append(f"{text} | {href}")
    logger.info(f"Extracted {len(lines)} candidate anchors from {url}")
    return "\n".join(lines)


def get_webpage_markdown(url: str) -> str:
    logger.debug(f"Fetching webpage: {url}")
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
        **kwargs,
    )
    return response


def extract_news_urls(anchor_list: str) -> List[str]:
    logger.debug("Extracting news URLs from anchor list")
    messages = [
        {"role": "system", "content": NEWS_FETCHER_INITIAL_PROMPT},
        {"role": "user", "content": anchor_list},
    ]
    response = call_llm(
        messages,
        response_format={"type": "json_object"},
        extra_body={"thinking_budget": 6144},
    )
    try:
        news_urls = json.loads(response.choices[0].message.content)  # type: ignore
        if isinstance(news_urls, list):
            logger.info(f"Extracted {len(news_urls)} news URLs")
            return news_urls
        else:
            raise ValueError(
                f"LLM response is not a list: {response.choices[0].message.content}"
            )  # type: ignore
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        raise ValueError(
            f"LLM response is not valid JSON: {response.choices[0].message.content}"
        )  # type: ignore


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
    logger.debug(f"Processing news URL: {url}")
    news_markdown = get_webpage_markdown(url)
    # if not is_today_news(news_markdown):
    #     logger.debug(f"Skipping non-today news: {url}")
    #     return None
    delay = random.uniform(0.5, 2)
    logger.debug(f"Waiting {delay:.2f}s to avoid rate limiting")
    time.sleep(delay)
    return summarize_news(news_markdown)


def main():
    load_dotenv()
    HOMEPAGE = "https://apnews.com/"
    # HOMEPAGE = "https://www.reuters.com/"
    homepage_anchors = extract_homepage_anchors(HOMEPAGE)
    news_urls = extract_news_urls(homepage_anchors)

    summaries: list[str] = []
    for i, url in enumerate(news_urls, 1):
        try:
            logger.info(f"[{i}/{len(news_urls)}] Processing: {url}")
            summary = process_news_url(url)
            if summary:
                summary = f"{summary.strip()}\n\n*[来源]({url})*"
                summaries.append(summary)
                logger.info(f"Successfully summarized: {url}")
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")

    today = datetime.now().strftime("%Y-%m-%d")
    output_file = f"news_{today}.md"
    content = f"# 新闻摘要 - {today}\n\n"
    content += "\n\n---\n\n".join(summaries)
    content += "\n"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(
        f"Summarization complete. Processed {len(summaries)}/{len(news_urls)} articles."
    )
    logger.info(f"Output written to {output_file}")
    return output_file, content


if __name__ == "__main__":
    main()
