"""Custom tools available to the SENTINEL-4 agents.

Agents never import these functions directly; they request them by name through
:class:`src.tools.registry.ToolRegistry`, which is where an invented ("hallucinated")
tool name is caught and refused.
"""
