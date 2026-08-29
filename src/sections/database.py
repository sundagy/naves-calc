from __future__ import annotations

from src.config import load_sections
from src.sections.rhs import RHSSection, parse_rhs


def section_database() -> dict[str, RHSSection]:
    return {name: parse_rhs(name) for name in load_sections()}


def get_section(name: str) -> RHSSection:
    db = section_database()
    if name not in db:
        raise KeyError(f"Unknown section: {name}")
    return db[name]

