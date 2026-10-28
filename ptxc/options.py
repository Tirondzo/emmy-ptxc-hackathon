"""The trade-offs the compiler decides on its own. Every option only picks between PTX programs
that pass the problem's check; the measurements behind the defaults are in tests/FINDINGS.md.
`compile.py` exposes `unroll`, `unroll_quota` and `precision`; the rest is set from Python
(tests, tests/bench.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Unroll = Literal["auto", "none", "force"]
Precision = Literal["auto", "exact", "fast"]


@dataclass(frozen=True)
class Options:
    unroll: Unroll = "auto"
    """auto: the IR's hints and the policy below; none: every loop rolled; force: every loop with a
    known trip count straight, whatever the quota."""

    precision: Precision = "auto"
    """auto: the float recipes may spend a quarter of the kernel's tolerance from meta.json on faster
    spellings (`ops.Budget`: fma contraction, approximate div / rcp / sqrt / tanh / sin / cos, `.ftz`),
    each priced by its worst case; exact: nothing, IEEE `.rn` everywhere, which also keeps ptxas from
    fusing mul + add; fast: no limit, whatever the tolerance."""

    raw_trig: bool = False
    """sin / cos through `sin.approx` on the raw argument when the budget allows its 2^-14.7 absolute
    error, which the ISA promises only within 100 pi: the caller vouches for the range. Off, the
    argument is reduced into [-pi, pi] first (three instructions), where the bound holds for any input."""

    unroll_quota: int = 2048
    """Statements a loop nest may expand to (`loops.sizes`). The register lever: ptxas spills when
    it can interleave too many copies."""

    unroll_auto: int = 8
    """Unmarked loops: copies per iteration, a power of two halved into the quota."""

    unroll_auto_full: int = 1024
    """Unmarked loops expanding to at most this many statements go straight."""

    unroll_auto_body: int = 64
    """Unmarked loops whose body, as written, is longer than this stay rolled."""

    cse: bool = True
    """Reuse an index subexpression or address base computed earlier in the scope. ptxas does the
    same elimination itself; this only halves the PTX text."""

    trace: tuple[str, ...] = ()
    """Topics reported on stderr: "unroll" prints each loop's decision."""
