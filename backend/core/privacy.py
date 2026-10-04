"""Personal data filter: applied to every transcript before it is stored or sent to a model.

Pilots talk while they fly and sometimes say things that have nothing to do with the line: a
phone number, an e-mail, a name. Those are replaced by a tag before the text reaches Claude,
the competence grid or the session files. Local regexes, no service call: nothing leaves the
machine unfiltered. Camera frames are rendered from the 3D scene only (the HUD and the panels are
DOM, never in the frame), so they hold no personal data.
"""

from __future__ import annotations

import re

MIN_DIGITS = 7  # "5 to 6 m", "12 m/s" or "25 30 metres" stay; phone, card and account numbers go


def _number(m: re.Match[str]) -> str:
    return "[NUMBER]" if sum(c.isdigit() for c in m.group(0)) >= MIN_DIGITS else m.group(0)


_PATTERNS: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"), "[IBAN]"),
    (re.compile(r"(?<!\w)\+?\d[\d ()./-]{5,}\d(?!\w)"), _number),
    # the introduction is matched in any case, the name only if capitalised ("call me at..." stays)
    (re.compile(r"\b((?i:my name is|i am called|i'm called|call me|je m'appelle|ich heiße|ich heisse))\s+[A-ZÀ-Ý][\w'-]+(\s+[A-ZÀ-Ý][\w'-]+)?"),
     r"\1 [NAME]"),
]


def redact(text: str) -> str:
    """The text with e-mails, long numbers and introduced names replaced by tags."""
    if not text:
        return text
    for pattern, tag in _PATTERNS:
        text = pattern.sub(tag, text)  # type: ignore[call-overload]
    return text
