import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import markdown as md
from dotenv import load_dotenv

from agentless import main as generate_news


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{subject}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    max-width: 720px;
    margin: 2em auto;
    padding: 0 1em;
    line-height: 1.7;
    color: #222;
}}
h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 0.3em; }}
h2 {{ color: #1a3a6c; margin-top: 2em; }}
h3 {{ color: #555; margin-top: 1.2em; }}
hr {{ border: none; border-top: 1px dashed #ccc; margin: 2.4em 0; }}
a {{ color: #1a6cb4; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
em {{ color: #777; font-size: 0.92em; }}
ul {{ padding-left: 1.4em; }}
li {{ margin-bottom: 0.5em; }}
strong {{ color: #111; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_html(subject: str, markdown_text: str) -> str:
    body = md.markdown(markdown_text, extensions=["extra", "sane_lists", "nl2br"])
    return HTML_TEMPLATE.format(subject=subject, body=body)


def send_email(subject: str, markdown_body: str) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    mail_from = os.getenv("MAIL_FROM", smtp_user)
    mail_to = os.environ["MAIL_TO"]  # comma-separated for multiple recipients

    recipients = [addr.strip() for addr in mail_to.split(",")]

    msg = MIMEMultipart("alternative")
    msg["From"] = mail_from
    msg["To"] = "undisclosed-recipients:;"
    msg["Bcc"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(markdown_body, "plain", "utf-8"))
    msg.attach(MIMEText(render_html(subject, markdown_body), "html", "utf-8"))

    use_ssl = os.getenv("SMTP_SSL", "true").lower() == "true"

    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(mail_from, recipients, msg.as_string())
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(mail_from, recipients, msg.as_string())

    print(f"[info] Email sent to {mail_to}")


def main() -> None:
    load_dotenv()

    _, content = generate_news()

    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"每日新闻摘要 - {today}"
    send_email(subject, content)


if __name__ == "__main__":
    main()
