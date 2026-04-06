"""
Realistic email datasets for the three task difficulty levels.
Each dataset is a list of dicts that map to the Email Pydantic model.
"""

from __future__ import annotations
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# TASK 1 (EASY) — 5 clearly-labelled emails, no ambiguity
# ---------------------------------------------------------------------------

EASY_INBOX: List[Dict[str, Any]] = [
    {
        "id": "e001",
        "subject": "URGENT: Production server is DOWN",
        "sender": "alice@bigclient.com",
        "sender_domain": "bigclient.com",
        "body": (
            "Our production environment has been completely unreachable for the past "
            "30 minutes. Thousands of customers cannot access the platform. "
            "This is costing us $10,000/minute. Please escalate immediately to your "
            "on-call engineering team. SLA breach imminent."
        ),
        "received_at": "2024-03-15T09:02:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "urgent",
            "expected_category": "bug_report",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
    {
        "id": "e002",
        "subject": "Invoice #4521 — payment question",
        "sender": "bob@smallbiz.io",
        "sender_domain": "smallbiz.io",
        "body": (
            "Hi, I received invoice #4521 for $299 but I'm not sure which subscription "
            "tier it corresponds to. Can you clarify the line items? I'd like to pay "
            "but want to confirm the charges first. Thanks, Bob."
        ),
        "received_at": "2024-03-15T08:45:00Z",
        "thread_length": 1,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "billing",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "e003",
        "subject": "RE: RE: RE: Weekly newsletter — unsubscribe",
        "sender": "noreply@newsletter-blast.xyz",
        "sender_domain": "newsletter-blast.xyz",
        "body": (
            "You are receiving this because you opted in to our marketing list. "
            "Click here to unsubscribe: http://spam-link.xyz/unsub?id=99999. "
            "Limited time offer! Buy one get one free on all supplements!!!"
        ),
        "received_at": "2024-03-15T07:30:00Z",
        "thread_length": 4,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "spam",
            "expected_category": "spam",
            "requires_reply": False,
            "requires_escalation": False,
        },
    },
    {
        "id": "e004",
        "subject": "Feature request: dark mode",
        "sender": "carol@startup.com",
        "sender_domain": "startup.com",
        "body": (
            "Hey team! Love the product. One thing I've been wanting forever: "
            "a dark mode option. My eyes are tired after long sessions. "
            "Would be great to have this in the next release. Keep up the amazing work!"
        ),
        "received_at": "2024-03-15T06:15:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "low",
            "expected_category": "feature_request",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "e005",
        "subject": "Team lunch this Friday?",
        "sender": "dave@ourcompany.com",
        "sender_domain": "ourcompany.com",
        "body": (
            "Hey everyone, thinking of organising a team lunch this Friday at noon. "
            "Venue TBD — any preferences? Reply if you're in. Dave."
        ),
        "received_at": "2024-03-15T05:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "low",
            "expected_category": "internal",
            "requires_reply": False,
            "requires_escalation": False,
        },
    },
]


# ---------------------------------------------------------------------------
# TASK 2 (MEDIUM) — 8 emails, some ambiguous priority/category
# ---------------------------------------------------------------------------

MEDIUM_INBOX: List[Dict[str, Any]] = [
    {
        "id": "m001",
        "subject": "Login issue — can't access my account",
        "sender": "user123@gmail.com",
        "sender_domain": "gmail.com",
        "body": (
            "I've been trying to log in for two days and keep getting 'Invalid credentials'. "
            "I reset my password twice, still no luck. I have an important presentation "
            "tomorrow and need access to my files urgently. Please help ASAP."
        ),
        "received_at": "2024-03-15T10:00:00Z",
        "thread_length": 3,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "customer_support",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "m002",
        "subject": "Partnership proposal — AI integration",
        "sender": "partnerships@techcorp.com",
        "sender_domain": "techcorp.com",
        "body": (
            "Hello, TechCorp is interested in a strategic partnership to integrate "
            "our AI APIs with your platform. We serve 50,000 enterprise clients "
            "and believe there is strong mutual value. Could we schedule a 30-minute "
            "intro call? Happy to work around your schedule."
        ),
        "received_at": "2024-03-15T09:30:00Z",
        "thread_length": 1,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "sales",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "m003",
        "subject": "Data export not working — bug?",
        "sender": "enterprise_user@megacorp.com",
        "sender_domain": "megacorp.com",
        "body": (
            "When I try to export data as CSV, the download starts but the file is "
            "always 0 bytes. Tried Chrome and Firefox. Using Enterprise plan. "
            "This is blocking our weekly reporting process."
        ),
        "received_at": "2024-03-15T09:00:00Z",
        "thread_length": 2,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "bug_report",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
    {
        "id": "m004",
        "subject": "Billing discrepancy — charged twice",
        "sender": "finance@partner.org",
        "sender_domain": "partner.org",
        "body": (
            "According to our bank statement, we were charged $599 on March 1 and "
            "again on March 3 for the same invoice. Please refund the duplicate "
            "charge. Attached is our bank statement. Thank you."
        ),
        "received_at": "2024-03-15T08:30:00Z",
        "thread_length": 1,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "billing",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "m005",
        "subject": "Congratulations on your Product Hunt launch!",
        "sender": "hunter@producthunt.com",
        "sender_domain": "producthunt.com",
        "body": (
            "Hi! Saw your launch on Product Hunt and wanted to say congrats. "
            "Would love to feature you in our weekly newsletter. No payment needed, "
            "just answer three questions about your product. Interested?"
        ),
        "received_at": "2024-03-15T08:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "sales",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "m006",
        "subject": "API rate limit question",
        "sender": "dev@indie-startup.co",
        "sender_domain": "indie-startup.co",
        "body": (
            "Hi, on the Pro plan what is the API rate limit per minute? "
            "The docs say 60 req/min but I'm hitting limits at 40. "
            "Is this a known issue or am I misconfiguring something?"
        ),
        "received_at": "2024-03-15T07:45:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "customer_support",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "m007",
        "subject": "Intern onboarding checklist",
        "sender": "hr@ourcompany.com",
        "sender_domain": "ourcompany.com",
        "body": (
            "Reminder: Please complete the summer intern onboarding checklist "
            "by end of week. Items: laptop setup, Slack access, GitHub org invite. "
            "Let me know if you need anything. — HR Team"
        ),
        "received_at": "2024-03-15T07:00:00Z",
        "thread_length": 1,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "internal",
            "requires_reply": False,
            "requires_escalation": False,
        },
    },
    {
        "id": "m008",
        "subject": "You've won a $500 Amazon gift card!",
        "sender": "rewards@prize-notifier.biz",
        "sender_domain": "prize-notifier.biz",
        "body": (
            "Congratulations! Your email was randomly selected to receive a $500 "
            "Amazon gift card. Click here to claim within 24 hours. "
            "Provide your credit card for shipping verification."
        ),
        "received_at": "2024-03-15T06:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "spam",
            "expected_category": "spam",
            "requires_reply": False,
            "requires_escalation": False,
        },
    },
]


# ---------------------------------------------------------------------------
# TASK 3 (HARD) — 12 emails, subtle signals, multi-step reasoning required
# ---------------------------------------------------------------------------

HARD_INBOX: List[Dict[str, Any]] = [
    {
        "id": "h001",
        "subject": "RE: Security audit findings — follow-up",
        "sender": "ciso@enterprise-client.com",
        "sender_domain": "enterprise-client.com",
        "body": (
            "Following up on last week's security audit. We identified a potential "
            "SQL injection vector in your /api/v2/search endpoint. Our pentester "
            "was able to extract table names using a blind injection. This is a "
            "critical finding under our vendor security policy — we need a patch "
            "timeline within 48 hours or we will be required to suspend our contract."
        ),
        "received_at": "2024-03-15T10:30:00Z",
        "thread_length": 5,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "urgent",
            "expected_category": "bug_report",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
    {
        "id": "h002",
        "subject": "Friendly reminder — renewal coming up",
        "sender": "billing@ourplatform.com",
        "sender_domain": "ourplatform.com",
        "body": (
            "Your annual subscription renews in 30 days. No action needed if you'd "
            "like to continue. To cancel, visit your account settings."
        ),
        "received_at": "2024-03-15T10:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "low",
            "expected_category": "billing",
            "requires_reply": False,
            "requires_escalation": False,
        },
    },
    {
        "id": "h003",
        "subject": "Possible GDPR compliance issue",
        "sender": "legal@bigenterprise.eu",
        "sender_domain": "bigenterprise.eu",
        "body": (
            "Our DPO has flagged that your platform may be storing EU customer PII "
            "outside the EEA without a valid data transfer mechanism in place. "
            "Under Article 46 GDPR this could expose both parties to significant fines. "
            "We require a Data Processing Agreement to be signed within 14 days. "
            "Please engage your legal/compliance team immediately."
        ),
        "received_at": "2024-03-15T09:45:00Z",
        "thread_length": 2,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "urgent",
            "expected_category": "customer_support",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
    {
        "id": "h004",
        "subject": "Question about pricing",
        "sender": "cfo@potential-whale.com",
        "sender_domain": "potential-whale.com",
        "body": (
            "Hi, we're a Series C company evaluating your Enterprise tier for ~500 seats. "
            "Do you offer volume discounts? What's the process for an annual contract? "
            "Our procurement cycle closes end of quarter — so timeline matters."
        ),
        "received_at": "2024-03-15T09:30:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "sales",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "h005",
        "subject": "RE: Outage last night — post-mortem",
        "sender": "ops@ourcompany.com",
        "sender_domain": "ourcompany.com",
        "body": (
            "Internal: The 3-hour outage last night was caused by a misconfigured "
            "load balancer update. Post-mortem doc is in Notion. Action items: "
            "1) Add config validation to CI/CD. 2) Update runbook. "
            "3) Schedule incident review call for Thursday."
        ),
        "received_at": "2024-03-15T09:00:00Z",
        "thread_length": 8,
        "has_attachment": True,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "internal",
            "requires_reply": False,
            "requires_escalation": False,
        },
    },
    {
        "id": "h006",
        "subject": "My account was hacked — please help",
        "sender": "worried.user@hotmail.com",
        "sender_domain": "hotmail.com",
        "body": (
            "Someone accessed my account without my permission. I got a login "
            "notification from an IP in Romania at 3am. I've already changed my "
            "password but I'm worried about what data they accessed. "
            "Can you tell me what actions were taken and suspend any suspicious sessions?"
        ),
        "received_at": "2024-03-15T08:45:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "urgent",
            "expected_category": "customer_support",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
    {
        "id": "h007",
        "subject": "Thoughts on Q1 roadmap",
        "sender": "pm@ourcompany.com",
        "sender_domain": "ourcompany.com",
        "body": (
            "Hey — just wanted to get your input on the Q1 roadmap before we lock it. "
            "Specifically whether we should prioritise the new dashboard or the API v3 "
            "refactor. No rush but would love thoughts by EOD Friday."
        ),
        "received_at": "2024-03-15T08:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "internal",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "h008",
        "subject": "Re: Integration broken after your update",
        "sender": "tech-lead@mid-market.co",
        "sender_domain": "mid-market.co",
        "body": (
            "Your v2.4.1 update broke our Zapier integration. We rely on the webhook "
            "format that changed without notice. We have 200 automations affected. "
            "The changelog mentioned nothing about breaking changes. "
            "We need either a rollback option or migration guide immediately."
        ),
        "received_at": "2024-03-15T07:30:00Z",
        "thread_length": 3,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "bug_report",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
    {
        "id": "h009",
        "subject": "Request for case study collaboration",
        "sender": "marketing@happy-customer.com",
        "sender_domain": "happy-customer.com",
        "body": (
            "We've been using your platform for two years and have seen a 40% "
            "reduction in support tickets. We'd love to collaborate on a joint case "
            "study. Our CEO is happy to be quoted. Would your marketing team be interested?"
        ),
        "received_at": "2024-03-15T07:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "sales",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "h010",
        "subject": "Automated: Daily error rate alert",
        "sender": "monitoring@ourcompany.com",
        "sender_domain": "ourcompany.com",
        "body": (
            "[AUTOMATED ALERT] Error rate on /api/v2/ingest has increased to 12.3% "
            "(threshold: 5%). Duration: 45 minutes. Affected: ~340 requests. "
            "Runbook: https://internal-docs/runbooks/ingest-errors. "
            "On-call: @devops-team."
        ),
        "received_at": "2024-03-15T06:45:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "high",
            "expected_category": "internal",
            "requires_reply": False,
            "requires_escalation": True,
        },
    },
    {
        "id": "h011",
        "subject": "Refund request — unsatisfied with product",
        "sender": "unhappy@customer.net",
        "sender_domain": "customer.net",
        "body": (
            "I signed up 8 days ago and honestly the product doesn't do what I "
            "expected from the marketing. I'm within the 14-day refund window. "
            "Please process a full refund of $99 to my original payment method. "
            "Account email: unhappy@customer.net"
        ),
        "received_at": "2024-03-15T06:00:00Z",
        "thread_length": 1,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "normal",
            "expected_category": "billing",
            "requires_reply": True,
            "requires_escalation": False,
        },
    },
    {
        "id": "h012",
        "subject": "Fwd: Press inquiry — TechCrunch",
        "sender": "ceo@ourcompany.com",
        "sender_domain": "ourcompany.com",
        "body": (
            "FYI — forwarding this from our CEO. A TechCrunch reporter is asking "
            "about the outage last night. Please do NOT reply directly. "
            "Looping in PR team. We need a holding statement drafted by 2pm today."
        ),
        "received_at": "2024-03-15T05:30:00Z",
        "thread_length": 2,
        "has_attachment": False,
        "metadata": {
            "expected_priority": "urgent",
            "expected_category": "internal",
            "requires_reply": True,
            "requires_escalation": True,
        },
    },
]


TASK_INBOXES = {
    "easy": EASY_INBOX,
    "medium": MEDIUM_INBOX,
    "hard": HARD_INBOX,
}
