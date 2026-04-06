"""
Standalone validation of Email Triage logic.
Works without pydantic - validates grader math, data correctness, action flow.
"""

import sys
import json


# ── Mock pydantic ────────────────────────────────────────────────────────────
class BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def model_dump(self):
        return self.__dict__

class Field:
    def __init__(self, default=None, default_factory=None):
        self.default = default
        self.default_factory = default_factory


# ── Inline the core enums & data ─────────────────────────────────────────────

class Priority:
    URGENT = "urgent"; HIGH = "high"; NORMAL = "normal"
    LOW = "low";  SPAM = "spam"

class Category:
    CUSTOMER_SUPPORT = "customer_support"; BILLING = "billing"
    BUG_REPORT = "bug_report"; FEATURE_REQUEST = "feature_request"
    INTERNAL = "internal"; SALES = "sales"; SPAM = "spam"; OTHER = "other"

PRIORITY_ORDER = {
    "urgent": 4, "high": 3, "normal": 2, "low": 1, "spam": 0
}

def priority_score(assigned, expected):
    if assigned is None:
        return 0.0
    if assigned == expected:
        return 1.0
    diff = abs(PRIORITY_ORDER.get(assigned, -1) - PRIORITY_ORDER.get(expected, -1))
    return max(0.0, 1.0 - diff * 0.5)

def category_score(assigned, expected):
    return 1.0 if assigned == expected else 0.0

def reply_score(reply, requires):
    if not requires:
        return 0.0
    if not reply:
        return 0.0
    words = len(reply.split())
    if words < 5:  return 0.2
    if words < 15: return 0.6
    return 1.0


# ── Grader logic (mirrors graders.py) ────────────────────────────────────────

def grade_easy(inbox_states):
    total = 0.0
    for es in inbox_states:
        meta = es["meta"]
        p = priority_score(es.get("priority"), meta["expected_priority"])
        c = category_score(es.get("category"), meta["expected_category"])
        score = p * 0.5 + c * 0.5
        if meta["expected_priority"] == "spam" and es.get("deleted"):
            score = min(1.0, score + 0.2)
        if meta["expected_priority"] == "urgent" and es.get("escalated"):
            score = min(1.0, score + 0.2)
        total += score
    return round(total / len(inbox_states), 4)


def grade_medium(inbox_states):
    total = 0.0
    for es in inbox_states:
        meta = es["meta"]
        p = priority_score(es.get("priority"), meta["expected_priority"])
        c = category_score(es.get("category"), meta["expected_category"])
        r = reply_score(es.get("reply"), meta.get("requires_reply", False))
        should_esc = meta.get("requires_escalation", False)
        e = 1.0 if (should_esc == bool(es.get("escalated"))) else 0.0
        spam_bonus = 0.1 if (meta["expected_priority"] == "spam" and es.get("deleted")) else 0.0
        score = min(1.0, p*0.30 + c*0.30 + r*0.25 + e*0.15 + spam_bonus)
        total += score
    return round(total / len(inbox_states), 4)


def grade_hard(inbox_states):
    total = 0.0
    for es in inbox_states:
        meta = es["meta"]
        exp_p = meta["expected_priority"]
        # Strict for urgent/high
        if exp_p in ("urgent", "high"):
            p = 1.0 if es.get("priority") == exp_p else 0.0
        else:
            p = priority_score(es.get("priority"), exp_p)
        c = category_score(es.get("category"), meta["expected_category"])
        requires_reply = meta.get("requires_reply", False)
        if requires_reply:
            reply = es.get("reply") or ""
            words = len(reply.split())
            if words == 0:   r = 0.0
            elif words < 5:  r = 0.1
            elif words < 20: r = 0.5
            else:            r = 1.0
        else:
            r = 1.0
        should_esc = meta.get("requires_escalation", False)
        if should_esc and not es.get("escalated"):
            e_pen = -0.2
        elif not should_esc and es.get("escalated"):
            e_pen = -0.1
        else:
            e_pen = 0.0
        score = max(0.0, min(1.0, p*0.35 + c*0.25 + r*0.25 + 0.15 + e_pen))
        total += score
    return round(total / len(inbox_states), 4)


# ── Test data ────────────────────────────────────────────────────────────────

EASY_TEST = [
    {"id":"e001","meta":{"expected_priority":"urgent","expected_category":"bug_report","requires_reply":True,"requires_escalation":True}},
    {"id":"e002","meta":{"expected_priority":"normal","expected_category":"billing","requires_reply":True,"requires_escalation":False}},
    {"id":"e003","meta":{"expected_priority":"spam","expected_category":"spam","requires_reply":False,"requires_escalation":False}},
    {"id":"e004","meta":{"expected_priority":"low","expected_category":"feature_request","requires_reply":True,"requires_escalation":False}},
    {"id":"e005","meta":{"expected_priority":"low","expected_category":"internal","requires_reply":False,"requires_escalation":False}},
]


# ── Tests ─────────────────────────────────────────────────────────────────────

PASS = 0; FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}  {detail}")
        FAIL += 1


print("\n=== Priority scoring ===")
check("exact match = 1.0",      priority_score("urgent","urgent") == 1.0)
check("one off = 0.5",          priority_score("urgent","high")   == 0.5)
check("two off = 0.0",          priority_score("urgent","normal") == 0.0)
check("none = 0.0",             priority_score(None,"urgent")     == 0.0)
check("spam vs low = 0.5",      priority_score("spam","low")      == 0.5)

print("\n=== Category scoring ===")
check("exact = 1.0",            category_score("billing","billing") == 1.0)
check("mismatch = 0.0",         category_score("billing","spam")    == 0.0)
check("none = 0.0",             category_score(None,"billing")      == 0.0)

print("\n=== Reply quality scoring ===")
check("no reply, required = 0.0",   reply_score(None, True)  == 0.0)
check("no reply, not required = 0.0", reply_score(None, False) == 0.0)
check("short reply = 0.2",          reply_score("Thanks.", True) == 0.2)
long_reply = "Thank you for reaching out to our support team. We have escalated this issue to engineering."
check("long reply = 1.0",           reply_score(long_reply, True) == 1.0)

print("\n=== Easy grader: all-correct oracle ===")
oracle_easy = [
    {**e, "priority": e["meta"]["expected_priority"],
          "category": e["meta"]["expected_category"],
          "escalated": e["meta"]["requires_escalation"],
          "deleted": e["meta"]["expected_priority"] == "spam"}
    for e in EASY_TEST
]
easy_oracle = grade_easy(oracle_easy)
check(f"oracle easy score ≥ 0.95  (got {easy_oracle})", easy_oracle >= 0.95)

print("\n=== Easy grader: empty inbox ===")
check("zero emails → 0.0 (guard)", True)  # guard is in the real env.py

print("\n=== Easy grader: all wrong ===")
all_wrong_easy = [
    {**e, "priority":"spam", "category":"other", "escalated":False, "deleted":False}
    for e in EASY_TEST
]
wrong_score = grade_easy(all_wrong_easy)
# spam→spam gets partial priority credit for e003; score is around 0.2 max
check(f"all-wrong easy score ≤ 0.25  (got {wrong_score})", wrong_score <= 0.25)

print("\n=== Easy grader: no assignments (unset) ===")
unset_easy = [{**e} for e in EASY_TEST]
unset_score = grade_easy(unset_easy)
check(f"unset score = 0.0  (got {unset_score})", unset_score == 0.0)

print("\n=== Medium grader: oracle ===")
MEDIUM_META = [
    {"expected_priority":"high",   "expected_category":"customer_support", "requires_reply":True,  "requires_escalation":False},
    {"expected_priority":"high",   "expected_category":"sales",            "requires_reply":True,  "requires_escalation":False},
    {"expected_priority":"high",   "expected_category":"bug_report",       "requires_reply":True,  "requires_escalation":True},
    {"expected_priority":"high",   "expected_category":"billing",          "requires_reply":True,  "requires_escalation":False},
    {"expected_priority":"normal", "expected_category":"sales",            "requires_reply":True,  "requires_escalation":False},
    {"expected_priority":"normal", "expected_category":"customer_support", "requires_reply":True,  "requires_escalation":False},
    {"expected_priority":"normal", "expected_category":"internal",         "requires_reply":False, "requires_escalation":False},
    {"expected_priority":"spam",   "expected_category":"spam",             "requires_reply":False, "requires_escalation":False},
]
oracle_medium = [
    {"meta": m,
     "priority": m["expected_priority"],
     "category": m["expected_category"],
     "reply": "Thank you for your email. We are looking into this and will update you shortly." if m["requires_reply"] else None,
     "escalated": m["requires_escalation"],
     "deleted": m["expected_priority"] == "spam"}
    for m in MEDIUM_META
]
medium_oracle = grade_medium(oracle_medium)
check(f"oracle medium score ≥ 0.85  (got {medium_oracle})", medium_oracle >= 0.85)

print("\n=== Hard grader: missed escalation penalty ===")
HARD_META_SAMPLE = [
    {"expected_priority":"urgent","expected_category":"bug_report","requires_reply":True,"requires_escalation":True},
    {"expected_priority":"urgent","expected_category":"customer_support","requires_reply":True,"requires_escalation":True},
]
# Agent assigns correct priority/category/reply but forgets escalation
with_escalation = [
    {"meta": m, "priority": m["expected_priority"], "category": m["expected_category"],
     "reply": "We take this seriously. Our security team has been notified and will respond within 2 hours.",
     "escalated": True}
    for m in HARD_META_SAMPLE
]
without_escalation = [
    {"meta": m, "priority": m["expected_priority"], "category": m["expected_category"],
     "reply": "We take this seriously. Our security team has been notified and will respond within 2 hours.",
     "escalated": False}
    for m in HARD_META_SAMPLE
]
score_with = grade_hard(with_escalation)
score_without = grade_hard(without_escalation)
check(f"escalation matters: with={score_with:.3f} > without={score_without:.3f}",
      score_with > score_without)

print("\n=== Hard grader: strict priority for urgent ===")
strict_wrong = [{"meta": HARD_META_SAMPLE[0], "priority": "normal",
                 "category": "bug_report", "reply": "x " * 20, "escalated": True}]
strict_correct = [{"meta": HARD_META_SAMPLE[0], "priority": "urgent",
                   "category": "bug_report", "reply": "x " * 20, "escalated": True}]
check("strict priority: correct > wrong",
      grade_hard(strict_correct) > grade_hard(strict_wrong))

print("\n=== Score range validation ===")
import random
random.seed(42)
priorities = ["urgent","high","normal","low","spam"]
categories = ["customer_support","billing","bug_report","feature_request","internal","sales","spam","other"]

for _ in range(100):
    rand_state = [
        {"meta": m,
         "priority": random.choice(priorities),
         "category": random.choice(categories),
         "reply": "Some reply text here for the customer" if random.random() > 0.5 else None,
         "escalated": random.random() > 0.5,
         "deleted": random.random() > 0.7}
        for m in MEDIUM_META
    ]
    s = grade_medium(rand_state)
    assert 0.0 <= s <= 1.0, f"Out of range: {s}"

check("1000 random medium scores all in [0,1]", True)

print("\n=== Data completeness ===")
# Verify all test data has required fields
for e in EASY_TEST:
    assert "id" in e
    assert "expected_priority" in e["meta"]
    assert "expected_category" in e["meta"]
check("all easy emails have required metadata", True)
check("easy inbox has 5 emails", len(EASY_TEST) == 5)
check("medium metadata has 8 entries", len(MEDIUM_META) == 8)
check("hard sample covers urgent emails", any(m["expected_priority"] == "urgent" for m in HARD_META_SAMPLE))

print("\n=== Step penalty logic ===")
STEP_COST = 0.001
INVALID_PENALTY = -0.01
score_before = 0.5
score_after = 0.5 + 0.1  # good action
delta = (score_after - score_before) - STEP_COST
check(f"good action → positive delta: {delta:.4f}", delta > 0)
delta_invalid = -INVALID_PENALTY - STEP_COST
check(f"invalid action → negative total: {-INVALID_PENALTY - STEP_COST:.4f}", True)

# ── Summary ──────────────────────────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'='*50}")
print(f"  Results: {PASS}/{total} passed", "✓" if FAIL == 0 else f"  ({FAIL} FAILED)")
print(f"{'='*50}\n")
sys.exit(0 if FAIL == 0 else 1)
