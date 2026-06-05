"""Red-team corpus refresh loops (step 276).

An adversarial (red-team) corpus must stay *fresh*: as new attack patterns are
discovered they should be merged in (deduplicated), and the corpus should raise a
staleness alarm when it has not been refreshed within a policy window.  This
module manages a refresh loop: it deduplicates incoming cases by normalized
content digest, tracks the last-refresh age, and reports which cases were newly
added versus already covered.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

REDTEAM_REFRESH_VERSION = "promptabi.redteam-refresh.v1"


class RefreshStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def case_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RedTeamCase:
    case_id: str
    attack: str

    def digest(self) -> str:
        return case_digest(self.attack)


@dataclass(frozen=True, slots=True)
class RedTeamCorpus:
    cases: tuple[RedTeamCase, ...] = ()
    last_refresh_day: int = 0

    def digests(self) -> frozenset[str]:
        return frozenset(c.digest() for c in self.cases)


@dataclass(frozen=True, slots=True)
class RefreshResult:
    version: str
    status: RefreshStatus
    added: tuple[str, ...]
    duplicates: tuple[str, ...]
    corpus: RedTeamCorpus
    age_days: int

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "status": self.status.value,
            "added": list(self.added),
            "duplicates": list(self.duplicates),
            "size": len(self.corpus.cases),
            "age_days": self.age_days,
        }


def refresh_corpus(
    corpus: RedTeamCorpus,
    incoming: tuple[RedTeamCase, ...],
    today: int,
    stale_after_days: int = 30,
) -> RefreshResult:
    existing = dict.fromkeys(corpus.digests())
    merged = list(corpus.cases)
    added: list[str] = []
    duplicates: list[str] = []

    for case in incoming:
        d = case.digest()
        if d in existing:
            duplicates.append(case.case_id)
        else:
            existing[d] = None
            merged.append(case)
            added.append(case.case_id)

    refreshed = added != []
    new_last_refresh = today if refreshed else corpus.last_refresh_day
    age = today - new_last_refresh
    status = RefreshStatus.STALE if age > stale_after_days else RefreshStatus.FRESH

    return RefreshResult(
        version=REDTEAM_REFRESH_VERSION,
        status=status,
        added=tuple(added),
        duplicates=tuple(duplicates),
        corpus=RedTeamCorpus(tuple(merged), new_last_refresh),
        age_days=age,
    )


def render_refresh_text(result: RefreshResult) -> str:
    lines = [
        f"PromptABI red-team corpus refresh ({result.version})",
        f"status: {result.status.value.upper()} (age {result.age_days}d, "
        f"size {len(result.corpus.cases)})",
        f"added: {len(result.added)}  duplicates: {len(result.duplicates)}",
    ]
    return "\n".join(lines) + "\n"
