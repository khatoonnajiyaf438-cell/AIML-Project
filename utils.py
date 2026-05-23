from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import urlparse

import networkx as nx


G = nx.Graph()
EMAIL_TO_COMPANIES = defaultdict(set)
PHONE_TO_COMPANIES = defaultdict(set)
LINK_TO_COMPANIES = defaultdict(set)

FREE_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
}

SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".biz", ".work", ".info")
PAYMENT_WORDS = (
    "registration fee",
    "processing fee",
    "application fee",
    "deposit",
    "payment required",
    "pay now",
    "charge",
)
UNREALISTIC_WORDS = (
    "earn fast",
    "easy money",
    "quick money",
    "guaranteed income",
    "instant joining",
)


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _company_key(company: str) -> str:
    return _normalize(company) or "unknown_company"


def _extract_email_domain(email: str) -> str:
    email = _normalize(email)
    return email.split("@")[-1] if "@" in email else ""


def _extract_link_domain(link: str) -> str:
    link = _normalize(link)
    if not link:
        return ""
    parsed = urlparse(link if "://" in link else f"https://{link}")
    return parsed.netloc.replace("www.", "")


def _looks_like_email(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", value.strip()))


def _looks_like_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def _looks_like_link(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith(("http://", "https://", "www.")) or "." in value


def _has_payment_words(job_text: str) -> bool:
    text = _normalize(job_text)
    return any(word in text for word in PAYMENT_WORDS)


def _has_unrealistic_salary(job_text: str) -> bool:
    text = _normalize(job_text)
    if any(word in text for word in UNREALISTIC_WORDS):
        return True

    money_matches = re.findall(r"(\d[\d,]*)", text)
    for match in money_matches:
        amount = int(match.replace(",", ""))
        if "per day" in text or "daily" in text:
            if amount >= 5000:
                return True
        if "per week" in text or "weekly" in text:
            if amount >= 50000:
                return True
        if "per month" in text or "monthly" in text:
            if amount >= 200000:
                return True
    return False


def _duplicate_company_entity(counter_map: dict[str, set[str]], key: str) -> bool:
    return key in counter_map and len(counter_map[key]) > 1


def add_job(company: str, entity_value: str, entity_type: str) -> None:
    company_key = _company_key(company)
    entity_key = _normalize(entity_value)
    if not entity_key:
        return

    company_node = f"company:{company_key}"
    entity_node = f"{entity_type}:{entity_key}"
    G.add_node(company_node)
    G.add_node(entity_node)
    G.add_edge(company_node, entity_node)

    if entity_type == "email":
        EMAIL_TO_COMPANIES[entity_key].add(company_key)
    elif entity_type == "phone":
        PHONE_TO_COMPANIES[entity_key].add(company_key)
    elif entity_type == "link":
        LINK_TO_COMPANIES[entity_key].add(company_key)


def is_suspicious(entity_value: str, entity_type: str) -> bool:
    entity_key = _normalize(entity_value)
    if not entity_key:
        return False

    if entity_type == "email":
        return len(EMAIL_TO_COMPANIES[entity_key]) > 2
    if entity_type == "phone":
        return len(PHONE_TO_COMPANIES[entity_key]) > 2
    if entity_type == "link":
        return len(LINK_TO_COMPANIES[entity_key]) > 2
    return False


def run_checks(job_text: str, company: str, check_type: str, check_value: str) -> dict[str, object]:
    check_type = _normalize(check_type)
    check_value = check_value.strip()
    flags: list[str] = []

    ml_related_risk = _has_payment_words(job_text) or _has_unrealistic_salary(job_text)

    detected_type = ""
    if check_type == "email check":
        detected_type = "email"
    elif check_type == "phone check":
        detected_type = "phone"
    elif check_type == "link check":
        detected_type = "link"
    else:
        if _looks_like_email(check_value):
            detected_type = "email"
        elif _looks_like_phone(check_value):
            detected_type = "phone"
        elif _looks_like_link(check_value):
            detected_type = "link"
        else:
            detected_type = "text"

    if detected_type == "email":
        if not _looks_like_email(check_value):
            flags.append("Invalid email format detected.")
        domain = _extract_email_domain(check_value)
        if domain in FREE_EMAIL_DOMAINS:
            flags.append(f"Suspicious email provider detected: {domain}.")
        add_job(company, check_value, "email")
        if is_suspicious(check_value, "email"):
            flags.append("This email is linked with multiple companies in graph analysis.")

    elif detected_type == "phone":
        if not _looks_like_phone(check_value):
            flags.append("Invalid phone number format detected.")
        digits = re.sub(r"\D", "", check_value)
        if len(set(digits)) <= 2 and len(digits) >= 10:
            flags.append("Phone number pattern looks suspicious or repetitive.")
        add_job(company, digits, "phone")
        if is_suspicious(digits, "phone"):
            flags.append("This phone number is linked with multiple companies in graph analysis.")

    elif detected_type == "link":
        domain = _extract_link_domain(check_value)
        if not domain:
            flags.append("Invalid application link detected.")
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            flags.append(f"Suspicious link domain detected: {domain}.")
        company_slug = re.sub(r"[^a-z0-9]", "", _company_key(company))
        if company_slug and company_slug not in re.sub(r"[^a-z0-9]", "", domain):
            flags.append("Company name does not match the application link domain.")
        add_job(company, domain or check_value, "link")
        if is_suspicious(domain or check_value, "link"):
            flags.append("This link/domain is linked with multiple companies in graph analysis.")

    else:
        flags.append("Full check input should contain an email, phone number, or application link.")

    if _has_payment_words(job_text):
        flags.append("Payment demand detected in job description.")
    if _has_unrealistic_salary(job_text):
        flags.append("Unrealistic salary or earnings claim detected.")

    is_fake = bool(flags) or ml_related_risk
    return {
        "is_fake": is_fake,
        "flags": flags if flags else ["No strong suspicious signal detected."],
        "detected_type": detected_type,
    }
