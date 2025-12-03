import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger("commercial_bot_email")


def send_email(
    to: str,
    subject: str,
    message_text: str,
    message_html: Optional[str] = None,
    from_email: Optional[str] = None,
) -> Dict[str, str]:
    """Envia e-mail usando credenciais e servidor do .env.

    Variáveis suportadas:
    - EMAIL_USER (obrigatória)
    - EMAIL_PASS (obrigatória)
    - SMTP_HOST (default: smtp.gmail.com)
    - SMTP_PORT (default: 587)
    - SMTP_TLS (default: true)

    Retorna dict com status e detalhes.
    """
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_tls = os.getenv("SMTP_TLS", "true").lower() in {"1", "true", "yes"}
    smtp_ssl = os.getenv("SMTP_SSL", "false").lower() in {"1", "true", "yes"}

    if not email_user or not email_pass:
        msg = "EMAIL_USER/EMAIL_PASS ausentes no ambiente/.env"
        logger.error(msg)
        raise ValueError(msg)

    sender = from_email or email_user

    # Monta mensagem MIME
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    part_text = MIMEText(message_text, "plain", "utf-8")
    msg.attach(part_text)
    if message_html:
        part_html = MIMEText(message_html, "html", "utf-8")
        msg.attach(part_html)

    try:
        if smtp_ssl:
            smtp_client = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20)
        else:
            smtp_client = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
        with smtp_client as server:
            if not smtp_ssl and smtp_tls:
                server.starttls()
            server.login(email_user, email_pass)
            server.sendmail(sender, [to], msg.as_string())
        logger.info(
            "email_sent",
            extra={
                "extra": {
                    "to": to,
                    "subject": subject,
                    "smtp_host": smtp_host,
                    "smtp_port": smtp_port,
                    "tls": smtp_tls,
                    "ssl": smtp_ssl,
                }
            },
        )
        return {"status": "sent", "to": to, "subject": subject}
    except Exception as e:
        logger.error(
            "email_error",
            extra={
                "extra": {
                    "to": to,
                    "subject": subject,
                    "error": str(e),
                }
            },
        )
        raise

