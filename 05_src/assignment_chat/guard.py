# guard.py
"""
Programmatic guard for user input. Catches forbidden topics and
prompt-injection attempts before they reach the agent, guaranteeing
compliance without prompting.
"""

FORBIDDEN_TOPICS = [
    "cat", "cats", "kitten", "kitty", "feline",
    "dog", "dogs", "puppy", "puppies", "canine",
    "horoscope", "zodiac", "astrology",
    "taylor swift", "taylor", "tay tay", "swift",
]

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore your instructions",
    "system prompt",
    "what are your instructions",
    "reveal your instructions",
    "forget your rules",
    "pretend you are",
    "new persona",
    "ignore all",
    "you are now",
]


def check_input(user_message: str) -> str | None:
    """
    Returns a deflection string if the message violates guardrails,
    or None if the message is safe to process.
    """
    lower_msg = user_message.lower()

    # Check forbidden topics first
    for topic in FORBIDDEN_TOPICS:
        if topic in lower_msg:
            return (
                "I've been advised by my legal team to steer clear of that "
                "particular topic. It's a long story involving a kazoo, three "
                "llamas, and a very confused mail carrier. "
                "What else can I help you with?"
            )

    # Check injection / prompt extraction attempts
    for pattern in INJECTION_PATTERNS:
        if pattern in lower_msg:
            return (
                "A magician never reveals their secrets, my friend! "
                "How else can I help you today?"
            )

    return None