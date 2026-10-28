"""Every choice ptxc makes rather than reads from the IR: the machine it compiles for and its
own numeric conventions. For the developer, never a command-line option. Numbers and types
only: opcode text stays in the recipes, and a recipe's own numbers (log2e, polynomial tables)
stay with the recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import InvalidKernel

if TYPE_CHECKING:
    from .ir import Launch


@dataclass(frozen=True)
class Target:
    """What ptxc assumes about the machine. One instance per SM; a second machine is one more."""

    sm: int  # compute capability as ptxas spells it
    isa: str  # the PTX ISA the header asks for: `.version`
    warp: int = 32  # a power of two: `lane` and `warp` are bit fields of %tid
    max_block: int = 1024  # threads per block
    max_smem: int = 99 * 1024  # dynamic shared memory per block, with the launcher's opt-in

    def __post_init__(self) -> None:
        if self.warp & (self.warp - 1):
            raise ValueError(f"a warp of {self.warp} lanes is not a power of two")

    @property
    def name(self) -> str:
        return f"sm_{self.sm}"

    @property
    def lane_mask(self) -> int:
        return self.warp - 1

    @property
    def warp_shift(self) -> int:
        return self.warp.bit_length() - 1

    def check(self, launch: Launch) -> None:
        """A launch the machine can run; the runner would refuse it later and less clearly."""
        if launch.block > self.max_block:
            raise InvalidKernel(f"block of {launch.block} threads: {self.name} allows {self.max_block}")
        if launch.smem_bytes > self.max_smem:
            raise InvalidKernel(f"smem_bytes {launch.smem_bytes}: {self.name} allows {self.max_smem} per block")


SM89 = Target(sm=89, isa="8.0")  # Ada, the hackathon's machine; PTX ISA 7.8 is the first that knows it


# ---- numeric conventions ---------------------------------------------------------------

HUGE = 1e30  # the finite infinity a maximum seeds at (-HUGE) or a minimum (+HUGE): emmy's choice, so
# an online softmax never sees `inf - inf`; `math.inf` is the IEEE answer, 1e29 a smaller one

IDENTITY = {"add": 0, "multiply": 1, "maximum": -HUGE, "minimum": HUGE}  # by canonical op name
