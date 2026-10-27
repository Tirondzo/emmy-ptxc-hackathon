class CompileError(Exception):
    """The input cannot be compiled. Reported as one line, exit 1."""


class InvalidKernel(CompileError):
    """The kernel breaks the IR's rules or disagrees with meta.json."""


class Unsupported(CompileError):
    """Valid IR that ptxc does not lower yet.

    Raised wherever lowering would otherwise have to guess, so that unfamiliar input
    fails loudly instead of becoming PTX that is silently wrong.
    """
