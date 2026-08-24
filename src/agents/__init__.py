"""Specialist agents of the SENTINEL-4 countermeasure unit.

Each module defines exactly one agent with a mutually exclusive persona, system
prompt and output contract. All of them inherit resilient invocation, tool-error
absorption and blackboard patching from :mod:`src.agents.base`.
"""
