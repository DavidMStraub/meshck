"""Check results and their text and JSON rendering."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(Enum):
    """How serious a finding is."""

    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    """One problem found by a check.

    Attributes:
        check: Identifier of the check that produced the finding.
        severity: ``WARN`` or ``ERROR``.
        message: Human-readable description, including measured values.
    """

    check: str
    severity: Severity
    message: str


Metrics = dict[str, int | float | list[float]]


@dataclass(frozen=True)
class RepairInfo:
    """What a repair changed, attached to the report of the repaired file.

    Attributes:
        source: The original file.
        before: Metrics of the original file.
        steps: Description of each repair step that changed the mesh.
        deviation: Estimated largest surface distance in mm between the
            original and the repaired mesh.
    """

    source: Path
    before: Metrics
    steps: list[str]
    deviation: float


@dataclass
class Report:
    """All findings for one mesh file.

    Attributes:
        path: The checked file.
        findings: Problems found, in the order the checks ran.
        metrics: Measured values (counts, size, volume), by name.
        repair: Set if ``path`` was produced by a repair.
    """

    path: Path
    findings: list[Finding] = field(default_factory=list)
    metrics: Metrics = field(default_factory=dict)
    repair: RepairInfo | None = None

    def add(self, check: str, severity: Severity, message: str) -> None:
        """Record a finding."""
        self.findings.append(Finding(check, severity, message))

    def passed(self, strict: bool = False) -> bool:
        """Return whether the file passes.

        Args:
            strict: If true, warnings count as failures too.
        """
        failing = {Severity.ERROR, Severity.WARN} if strict else {Severity.ERROR}
        return not any(f.severity in failing for f in self.findings)

    def to_dict(self, strict: bool = False) -> dict:
        """Return the report as a JSON-serializable dict.

        Args:
            strict: Passed on to ``passed``.
        """
        data: dict = {
            "path": str(self.path),
            "passed": self.passed(strict),
            "metrics": self.metrics,
            "findings": [
                {"check": f.check, "severity": f.severity.value, "message": f.message}
                for f in self.findings
            ],
        }
        if self.repair:
            data["repair"] = {
                "source": str(self.repair.source),
                "before": self.repair.before,
                "steps": self.repair.steps,
                "deviation_mm": self.repair.deviation,
            }
        return data

    def to_text(self, strict: bool = False) -> str:
        """Return the report as a human-readable, multi-line string.

        Args:
            strict: Passed on to ``passed``.
        """
        status = "PASS" if self.passed(strict) else "FAIL"
        if self.repair is None:
            lines = [f"{self.path}: {status}"]
            if self.metrics:
                lines.append("  " + _fmt_metrics(self.metrics))
        else:
            lines = [f"{self.repair.source} -> {self.path}: {status}",
                     "  before: " + _fmt_metrics(self.repair.before),
                     "  after:  " + _fmt_metrics(self.metrics)]
            lines += [f"  repair: {step}" for step in self.repair.steps]
            lines.append(f"  deviation_mm={self.repair.deviation:.4f}")
        lines += [f"  {f.severity.value:5}  {f.check}: {f.message}" for f in self.findings]
        return "\n".join(lines)


def _fmt_metrics(metrics: Metrics) -> str:
    """Format all metrics on one line."""
    return "  ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())


def _fmt(value: int | float | list[float]) -> str:
    """Format a metric value compactly for text output."""
    if isinstance(value, list):
        return "x".join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
