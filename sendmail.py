import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

from agentless import main as generate_news


def send_email(subject: str, body: str) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    mail_from = os.getenv("MAIL_FROM", smtp_user)
    mail_to = os.environ["MAIL_TO"]  # comma-separated for multiple recipients

    recipients = [addr.strip() for addr in mail_to.split(",")]

    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = "undisclosed-recipients:;"
    msg["Bcc"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

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
