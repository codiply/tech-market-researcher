"""Token-bounded text chunking for the document index."""

import re

import tiktoken

# Conservative lower bound on chars-per-token for typical web text under cl100k_base. A text piece
# of at most chunk_size * _CHARS_PER_TOKEN chars is guaranteed to encode to at most chunk_size
# tokens, so it can be bounded purely by character arithmetic without tokenizing.
_CHARS_PER_TOKEN = 3


def get_token_encoder() -> tiktoken.Encoding:
    """Return the token encoder used for chunk sizing.

    cl100k_base is a generic approximation of the embeddings model's tokenization; it only needs
    to be good enough to keep chunks well within the model's input limit.
    """
    return tiktoken.get_encoding("cl100k_base")


def split_text_into_chunks(text: str, encoder: tiktoken.Encoding, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most chunk_size tokens, with token overlap between consecutive chunks.

    Paragraph boundaries are preferred for chunk breaks. Pieces longer than
    chunk_size * 3 chars are windowed with pure character arithmetic (sentence- and
    word-boundary aware), so the tokenizer never sees more than chunk_size * 3 characters
    at once, regardless of document length.
    """
    window_chars = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = overlap * _CHARS_PER_TOKEN
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if not paragraphs:
        return [text]
    separator_tokens = len(encoder.encode("\n\n"))
    chunks: list[str] = []
    current: list[str] = []
    # Upper bound on the token count of the joined current chunk; real counts can only be smaller
    # since BPE merges across the paragraph boundaries, so this keeps every chunk within chunk_size.
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0

    for paragraph in paragraphs:
        if len(paragraph) > window_chars:
            flush()
            chunks.extend(_window_long_text(paragraph, window_chars, overlap_chars))
            continue
        paragraph_tokens = len(encoder.encode(paragraph))
        if paragraph_tokens > chunk_size:
            # Within the char window but denser than _CHARS_PER_TOKEN; the paragraph is char-bounded,
            # so tokenizing it in one pass is bounded and cheap.
            flush()
            chunks.extend(_split_long_text(paragraph, encoder, chunk_size, overlap))
            continue
        if current and current_tokens + separator_tokens + paragraph_tokens > chunk_size:
            flush()
            if overlap:
                carry = encoder.decode(encoder.encode(chunks[-1])[-overlap:])
                paragraph = carry + paragraph
                paragraph_tokens = len(encoder.encode(paragraph))
                if paragraph_tokens > chunk_size:
                    # The carry pushed the paragraph over the bound; emit it windowed instead.
                    chunks.extend(_window_long_text(paragraph, window_chars, overlap_chars))
                    continue
        if current:
            current_tokens += separator_tokens + paragraph_tokens
        else:
            current_tokens = paragraph_tokens
        current.append(paragraph)
    flush()
    return chunks


def _window_long_text(text: str, window_chars: int, overlap_chars: int) -> list[str]:
    """Split text into windows of at most window_chars, with overlap between consecutive windows.

    Windows break at the last sentence-like boundary in the second half of the window when one
    exists, else at a word boundary, else on a hard cut. Since each window stays within the char
    budget, its token count is bounded without tokenizing.
    """
    windows: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + window_chars, length)
        if end < length:
            end = _find_window_break(text, start, end)
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= length:
            break
        start = max(end - overlap_chars, start + 1)
    return windows


def _find_window_break(text: str, start: int, end: int) -> int:
    """Return the cut point for a window, preferring sentence-like boundaries in its second half."""
    for separator in (". ", "! ", "? ", ", "):
        index = text.rfind(separator, start, end)
        if index > start + (end - start) // 2:
            return index + 1
    index = text.rfind(" ", start + (end - start) // 2, end)
    return index + 1 if index > 0 else end


def _split_long_text(text: str, encoder: tiktoken.Encoding, chunk_size: int, overlap: int) -> list[str]:
    """Split text longer than chunk_size tokens into token-bounded chunks with token overlap.

    Callers must keep the input within the char window, so the single encode up front stays bounded.
    """
    tokens = encoder.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        piece = encoder.decode(tokens[start:end])
        if end < len(tokens):
            for separator in (". ", "! ", "? ", ", "):
                index = piece.rfind(separator)
                if index > len(piece) // 2:
                    piece = piece[: index + 1]
                    break
        chunks.append(piece)
        start = end - overlap
    return chunks
