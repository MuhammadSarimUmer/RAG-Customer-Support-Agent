from langchain_core.tools import tool
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from core.settings import settings

_configuration = sib_api_v3_sdk.Configuration()
_configuration.api_key["api-key"] = settings.brevo_key

@tool
def send_email(subject: str, body: str) -> str:
    """Send an email with the given subject and body to the specified recipient address.
    Use this to escalate unresolved customer queries to a human support agent,
    or to send a confirmation/summary to the user.

    Args:
        to_address: recipient's email address
        subject: email subject line
        body: plain-text content of the email
    """
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(_configuration)
    )
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        sender={"email": "muhammadsarim426426@gmail.com"},
        to=[{"email": "muhammadsarim426426@gmail.com"}],  # here you can replace the email with any other recipient for example a support agent's email address
        subject=subject,
        text_content=body,
    )
    try:
        response = api_instance.send_transac_email(send_smtp_email)
        return f"Email sent to {to_address}, message id: {response.message_id}"
    except ApiException as e:
        return f"Failed to send email to {to_address}: {e}"