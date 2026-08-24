"""Tests for the five custom tools and the registry's hallucination guard."""

from __future__ import annotations

import pytest

from src.tools.registry import (
    HallucinatedToolError,
    ToolExecutionError,
    build_default_registry,
)


@pytest.fixture()
def registry():
    """Provide a freshly built default tool registry.

    Returns:
        A :class:`~src.tools.registry.ToolRegistry` with all five tools.
    """
    return build_default_registry()


def test_registry_exposes_at_least_three_tools(registry):
    """The brief requires a minimum of three custom tools; we ship five."""
    assert len(registry.names()) >= 3
    assert set(registry.names()) == {
        "check_frame_consistency", "parse_indicators", "query_threat_intel",
        "read_evidence_file", "scan_audio_artifacts",
    }


def test_hallucinated_tool_is_refused_not_crashed(registry):
    """An unregistered tool name raises a typed error and is logged."""
    with pytest.raises(HallucinatedToolError):
        registry.invoke("run_nmap_scan", target="10.0.0.0/24")
    assert registry.invocation_log[-1]["status"] == "hallucinated"


def test_bad_arguments_are_reported_distinctly(registry):
    """Invented arguments on a real tool surface as a bad-arguments failure."""
    with pytest.raises(ToolExecutionError):
        registry.invoke("parse_indicators", wrong_kwarg="x")
    assert registry.invocation_log[-1]["status"] == "bad_arguments"


def test_indicator_parser_extracts_each_ioc_class(registry):
    """All four indicator classes are recovered from a realistic log line."""
    text = ("beacon 185.220.101.44 -> veridian-support.top, CVE-2024-3400, "
            "hash d41d8cd98f00b204e9800998ecf8427e")
    result = registry.invoke("parse_indicators", text=text)
    assert result["ipv4"] == ["185.220.101.44"]
    assert result["domains"] == ["veridian-support.top"]
    assert result["cves"] == ["CVE-2024-3400"]
    assert result["hashes"] == ["d41d8cd98f00b204e9800998ecf8427e"]
    assert result["total"] == 4


def test_indicator_parser_does_not_mistake_ip_octets_for_domains(registry):
    """Regression: the IPv4 tail must not be reported as a domain name."""
    result = registry.invoke("parse_indicators", text="traffic to 10.20.30.40 only")
    assert result["domains"] == []


def test_indicator_parser_rejects_non_string(registry):
    """A non-string argument is a contract breach, not a silent no-op."""
    with pytest.raises(ToolExecutionError):
        registry.invoke("parse_indicators", text=12345)


def test_audio_tool_is_deterministic(registry):
    """The same artefact reference always yields the same measurement."""
    first = registry.invoke("scan_audio_artifacts", evidence_ref="cfo_call.wav")
    second = registry.invoke("scan_audio_artifacts", evidence_ref="cfo_call.wav")
    assert first == second
    assert 0 <= first["synthesis_likelihood"] <= 100


def test_audio_tool_rejects_empty_reference(registry):
    """An empty artefact reference is refused rather than analysed."""
    with pytest.raises(ToolExecutionError):
        registry.invoke("scan_audio_artifacts", evidence_ref="   ")


def test_video_tool_reports_artefact_classes(registry):
    """The video detector returns a bounded likelihood and named artefacts."""
    result = registry.invoke("check_frame_consistency", evidence_ref="cctv.mp4")
    assert 0 <= result["manipulation_likelihood"] <= 100
    assert isinstance(result["artefact_classes"], list) and result["artefact_classes"]


def test_threat_intel_hit_and_miss(registry):
    """A corpus hit carries severity; a miss degrades gracefully."""
    hit = registry.invoke("query_threat_intel", indicator="The Entity")
    miss = registry.invoke("query_threat_intel", indicator="203.0.113.99")
    assert hit["found"] and hit["severity"] > 0
    assert not miss["found"] and miss["severity"] == 0


def test_evidence_reader_blocks_path_traversal(registry):
    """Reads outside the evidence sandbox are refused."""
    with pytest.raises(ToolExecutionError) as excinfo:
        registry.invoke("read_evidence_file", filename="../../etc/passwd")
    assert "PermissionError" in str(excinfo.value)


def test_evidence_reader_reports_missing_file_without_raising(registry):
    """A missing artefact is reported as absent, not as an error."""
    result = registry.invoke("read_evidence_file", filename="does_not_exist.txt")
    assert result["exists"] is False and result["content"] == ""
