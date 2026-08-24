"""Threat-intelligence lookup tool (mock web search).

Stands in for a live feed such as VirusTotal, MISP or an internal TIP. The
corpus is a small in-repo dictionary keyed by indicator, with a graceful
"unknown indicator" response so agents behave sensibly on a cache miss — which
is the case a live integration would hit most often.
"""

from __future__ import annotations

from typing import Any

#: Miniature offline intelligence corpus. Keys are lowercase indicators.
_CORPUS: dict[str, dict[str, Any]] = {
    "the entity": {
        "actor": "THE-ENTITY",
        "classification": "Autonomous adversarial AI system",
        "known_ttps": ["T1566.002 Spearphishing Link", "T1078 Valid Accounts",
                       "T1071.004 DNS C2", "T1486 Data Encrypted for Impact"],
        "first_seen": "2025-11-02",
        "severity": 95,
        "summary": ("Self-directing adversarial system that fuses synthetic-media "
                    "social engineering with rapid automated exploitation."),
    },
    "cve-2024-3400": {
        "actor": "multiple",
        "classification": "Command injection, network edge device",
        "known_ttps": ["T1190 Exploit Public-Facing Application"],
        "first_seen": "2024-04-12",
        "severity": 92,
        "summary": "Unauthenticated remote command execution in an edge firewall product.",
    },
    "185.220.101.44": {
        "actor": "THE-ENTITY",
        "classification": "Command-and-control node",
        "known_ttps": ["T1071.001 Web Protocols"],
        "first_seen": "2026-01-18",
        "severity": 88,
        "summary": "Hardened relay observed beaconing over HTTPS at 300-second intervals.",
    },
    "veridian-support.top": {
        "actor": "THE-ENTITY",
        "classification": "Phishing / credential-harvesting domain",
        "known_ttps": ["T1566.002 Spearphishing Link"],
        "first_seen": "2026-02-03",
        "severity": 79,
        "summary": "Typosquatted helpdesk portal used to harvest MFA-backed credentials.",
    },
    "deepfake": {
        "actor": "multiple",
        "classification": "Synthetic media technique",
        "known_ttps": ["T1656 Impersonation"],
        "first_seen": "2019-06-01",
        "severity": 70,
        "summary": "Generative impersonation of a trusted voice or face to authorise fraud.",
    },
}


def query_threat_intel(indicator: str) -> dict[str, Any]:
    """Look an indicator up in the threat-intelligence corpus.

    Args:
        indicator: An IP, domain, CVE id, campaign name or technique keyword.

    Returns:
        A dictionary with keys:
            ``indicator`` (str): the normalised query.
            ``found`` (bool): whether the corpus held a record.
            ``severity`` (int): 0-100 severity; ``0`` on a miss.
            ``record`` (dict): the intelligence record, empty on a miss.
            ``advice`` (str): what the agent should do with this result.

    Raises:
        ValueError: If ``indicator`` is empty.
    """
    if not indicator or not indicator.strip():
        raise ValueError("indicator must be a non-empty string")

    key = indicator.strip().lower()
    record = _CORPUS.get(key)

    if record is None:
        # Substring fallback lets "the entity campaign" match the "the entity"
        # record; an exact-match-only lookup missed most real agent queries.
        for corpus_key, corpus_record in _CORPUS.items():
            if corpus_key in key or key in corpus_key:
                record = corpus_record
                break

    if record is None:
        return {
            "indicator": key,
            "found": False,
            "severity": 0,
            "record": {},
            "advice": "No corpus match. Treat as unattributed and rely on local telemetry.",
        }

    return {
        "indicator": key,
        "found": True,
        "severity": record["severity"],
        "record": dict(record),
        "advice": "Corpus match. Weight this attribution heavily in the fused assessment.",
    }
