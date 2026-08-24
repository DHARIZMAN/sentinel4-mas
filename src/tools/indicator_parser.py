"""Indicator-of-compromise extraction tool.

Unlike the two forensic simulators, this tool does real work: it parses free
text with regular expressions and returns the IOCs it finds. It is deliberately
deterministic and dependency-free, which makes it the natural anchor for the
unit-test suite.
"""

from __future__ import annotations

import re
from typing import Any

_IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_SHA256_RE = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)
_MD5_RE = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com|net|org|io|ru|cn|xyz|top|info|biz|co|dev|onion)\b",
    re.IGNORECASE,
)


def parse_indicators(text: str) -> dict[str, Any]:
    """Extract indicators of compromise from free text.

    Args:
        text: Any text — an incident brief, a log excerpt, or an agent's own
            narrative output.

    Returns:
        A dictionary with keys:
            ``ipv4`` (list[str]): unique IPv4 addresses, in first-seen order.
            ``domains`` (list[str]): unique lowercase domain names.
            ``cves`` (list[str]): unique uppercase CVE identifiers.
            ``hashes`` (list[str]): unique lowercase MD5/SHA-256 digests.
            ``total`` (int): count across all categories.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_indicators expects str, received {type(text).__name__}")

    def _unique(matches: list[str], transform: str = "lower") -> list[str]:
        """Deduplicate matches while preserving first-seen order.

        Args:
            matches: Raw regex matches.
            transform: ``"lower"`` or ``"upper"`` normalisation to apply.

        Returns:
            The normalised, order-preserving unique list.
        """
        seen: dict[str, None] = {}
        for match in matches:
            key = match.lower() if transform == "lower" else match.upper()
            seen.setdefault(key, None)
        return list(seen)

    ipv4 = _unique(_IPV4_RE.findall(text))
    # [HUMAN-REVIEW] The AI's domain regex also matched the trailing octet group
    # of every IPv4 address (e.g. "10.20.30.40" -> "30.40"). We now subtract any
    # domain that is a substring of an already-extracted IP, which removed all
    # false positives in our test corpus.
    domains = [d for d in _unique(_DOMAIN_RE.findall(text)) if not any(d in ip for ip in ipv4)]
    hashes = _unique(_SHA256_RE.findall(text)) + [
        h for h in _unique(_MD5_RE.findall(text))
        if not any(h in long_hash for long_hash in _unique(_SHA256_RE.findall(text)))
    ]

    result = {
        "ipv4": ipv4,
        "domains": domains,
        "cves": _unique(_CVE_RE.findall(text), transform="upper"),
        "hashes": hashes,
    }
    result["total"] = sum(len(v) for v in result.values() if isinstance(v, list))
    return result
