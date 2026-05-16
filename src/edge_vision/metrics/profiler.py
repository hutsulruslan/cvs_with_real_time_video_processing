from __future__ import annotations

from time import perf_counter
from typing import Callable


class Profiler:
    """Measure elapsed milliseconds for named processing sections."""

    def __init__(self, time_provider: Callable[[], float] = perf_counter) -> None:
        self._time_provider = time_provider
        self._active_sections: dict[str, float] = {}
        self._measured_sections: dict[str, float] = {}

    def start(self, section_name: str) -> None:
        """Start measuring a named section."""
        _validate_section_name(section_name)
        self._active_sections[section_name] = self._time_provider()

    def stop(self, section_name: str) -> float:
        """Stop measuring a named section and return elapsed milliseconds."""
        _validate_section_name(section_name)
        if section_name not in self._active_sections:
            raise ValueError(f"Profiler section was not started: {section_name}")

        elapsed_ms = (self._time_provider() - self._active_sections.pop(section_name)) * 1000.0
        self._measured_sections[section_name] = elapsed_ms
        return elapsed_ms

    def get_ms(self, section_name: str) -> float:
        """Return measured milliseconds for a section, or 0.0 if absent."""
        return self._measured_sections.get(section_name, 0.0)

    def as_dict(self) -> dict[str, float]:
        """Return a copy of measured section timings."""
        return dict(self._measured_sections)

    def reset(self) -> None:
        """Clear active and measured section timings."""
        self._active_sections.clear()
        self._measured_sections.clear()


def _validate_section_name(section_name: str) -> None:
    if not isinstance(section_name, str) or not section_name.strip():
        raise ValueError("Profiler section name must be a non-empty string.")
