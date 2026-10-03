import re

VERSION_PATTERN = re.compile(
    r"Version\s+([0-9]+(?:\.[0-9]+)*)",
    re.IGNORECASE,
)

PAGE_TEXT_PATTERN = re.compile(
    r"P\s*a\s*g\s*e\s+(\d+)\s*\|\s*(\d+)",
    re.IGNORECASE,
)


FOOTER_PATTERN = re.compile(
    r"abc\s+technology\s+limited" r".*?" r"[-–—]\s*version\s+[0-9]+(?:\.[0-9]+)*",
    re.IGNORECASE,
)


def extract_version(text: str) -> str | None:
    """
    Extract document version from the document text.

    Example:
        Version 2.0
        -> 2.0
    """

    match = VERSION_PATTERN.search(text)

    if not match:
        return None

    return match.group(1)


def remove_footer(text: str) -> str:
    """
    Remove repeated company footer.

    Example:
        abc technology Limited Access Card Policy – Version 2.0

    is removed, while:

        Access Card Policy – Version 2.0

    remains.
    """

    return FOOTER_PATTERN.sub("", text)


def remove_page_marker(text: str) -> str:
    """
    Remove page marker from the actual content.

    Handles:
        Page 3 | 5
        P a g e 3 | 5
    """

    return re.sub(
        r"^[ \t]*P\s*a\s*g\s*e\s+\d+\s*\|\s*\d+[ \t]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )


import re


def normalize_whitespace(text: str) -> str:
    """
    Remove unwanted whitespace while preserving Markdown structure.

    - Removes trailing spaces/tabs from each line
    - Converts CRLF/CR line endings to LF
    - Removes excessive consecutive blank lines
    - Removes whitespace-only lines
    - Preserves meaningful Markdown newlines
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
