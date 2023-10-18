import re

__version__ = "0.1.1.dev0"


def is_canonical(version: str) -> bool:
    """Return True if `version` is a PEP440 conformant version."""
    match = re.match(
        (
            r"^([1-9]\d*!)?(0|[1-9]\d*)"
            r"(\.(0|[1-9]\d*))"
            r"*((a|b|rc)(0|[1-9]\d*))"
            r"?(\.post(0|[1-9]\d*))"
            r"?(\.dev(0|[1-9]\d*))?$"
        ),
        version,
    )

    return match is not None


assert is_canonical(__version__)
