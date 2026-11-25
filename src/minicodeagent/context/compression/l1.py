"""Minimal deterministic L1 cleanup for the edges of ordinary text."""


def _display_bounds(text: str) -> tuple[int, int]:
    if not text or text.isspace() or "```" in text or "~~~" in text:
        return 0, len(text)

    start = 0
    while text.startswith(("\n", "\r\n"), start):
        start += 2 if text[start] == "\r" else 1

    # Keep the line terminator of the last substantive line. Only subsequent
    # physically empty lines are redundant. Preserve LF and CRLF as written.
    end = len(text)
    keep_end = end
    while end > start and text[end - 1] == "\n":
        keep_end = end
        end -= 1
        if end > start and text[end - 1] == "\r":
            end -= 1
    return start, keep_end


def trim_display_edges(text: str) -> str:
    start, end = _display_bounds(text)
    return text[start:end]


def is_display_edge_trim(original: str, replacement: str) -> bool:
    """Validate a stored replacement without generating a candidate."""
    start, end = _display_bounds(original)
    return (start, end) != (0, len(original)) and end - start == len(replacement) and original.startswith(
        replacement, start, end,
    )
