"""Chunking strategies for RAG pipelines.

This module implements four progressively sophisticated chunking strategies,
each suited for different document types and retrieval requirements.

Strategy overview (in order of complexity):

1. fixed_size       — Simple, predictable. Good baseline.
2. recursive        — Paragraph-aware. Better at preserving meaning.
3. parent_child     — Two-level hierarchy. Best precision + context.
4. tokenizer_aware  — Token-budget based. Ensures LLM context limits.

When to use which:
------------------
| Strategy        | Best for                                   |
|-----------------|--------------------------------------------|
| fixed_size      | Quick prototypes, homogeneous text         |
| recursive       | Blog posts, documentation, web content     |
| parent_child    | Long reports, contracts, manuals           |
| tokenizer_aware | When feeding chunks directly into LLM      |

Design Notes
------------
- All strategies return ``Chunk`` objects with a ``chunk_id``, ``text``,
  ``metadata``, and optional ``parent_id`` (for parent-child).
- ``parent_child`` returns BOTH parent and child chunks. Store all of them
  in the vector DB — children for retrieval, parents for context injection.
- Tokenizer-aware chunking requires ``tiktoken`` (OpenAI tokenizer). It falls
  back to character-based estimation if tiktoken is not installed.

Usage
-----
    from src.chunker import chunk_text, ChunkStrategy

    chunks = chunk_text(
        text="...",
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=500,
        overlap=50,
    )
    for c in chunks:
        print(c.chunk_id, c.text[:80])
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ChunkStrategy(str, Enum):
    """Available chunking strategies.

    Use the string value when selecting from a Gradio dropdown.
    """
    FIXED = "fixed_size"
    RECURSIVE = "recursive"
    PARENT_CHILD = "parent_child"
    TOKENIZER = "tokenizer_aware"


@dataclass
class Chunk:
    """A single text chunk ready for embedding and indexing.

    Attributes
    ----------
    chunk_id:
        Unique ID (UUID4 string). Used as the Qdrant point ID.
    text:
        The actual text content of this chunk.
    chunk_index:
        0-indexed position of this chunk in the original document.
    strategy:
        Which strategy produced this chunk.
    parent_id:
        For parent-child: the chunk_id of the parent chunk. None otherwise.
    is_parent:
        True if this chunk is a parent (should be stored but not retrieved
        directly — only used to inject context).
    metadata:
        Arbitrary key-value pairs passed through to Qdrant payload.
    """

    chunk_id: str
    text: str
    chunk_index: int
    strategy: str
    parent_id: Optional[str] = None
    is_parent: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy,
            "parent_id": self.parent_id,
            "is_parent": self.is_parent,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking
# ---------------------------------------------------------------------------


def _chunk_fixed(
    text: str,
    chunk_size: int,
    overlap: int,
    metadata: dict,
) -> list[Chunk]:
    """Split text into fixed-size character chunks with overlap.

    This is the simplest strategy. It does NOT respect word or sentence
    boundaries — it just counts characters. This often cuts mid-sentence.

    When to use:
    - Homogeneous text (all similar length, no important structure)
    - Quick prototype where quality matters less than speed
    - When you need perfectly predictable chunk sizes

    Trade-offs:
    - Pro: Fast, deterministic, no dependencies
    - Con: Cuts mid-sentence, destroys semantic boundaries

    Overlap explanation:
    - Without overlap: chunks A|B|C — retrieval misses info at boundaries
    - With overlap=100: chunk1 ends at char 500, chunk2 starts at char 400
    - The 100-char tail of chunk1 is repeated at the start of chunk2
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    chunk_index=idx,
                    strategy=ChunkStrategy.FIXED,
                    metadata=dict(metadata),
                )
            )
            idx += 1

        # Move forward by (chunk_size - overlap)
        step = chunk_size - overlap
        if step <= 0:
            break
        start += step

    logger.debug(
        "Fixed-size chunking done",
        total_chars=len(text),
        chunk_size=chunk_size,
        overlap=overlap,
        chunks=len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Recursive character splitter
# ---------------------------------------------------------------------------

# Priority-ordered separators (try double-newline first, then single, etc.)
# This mimics LangChain's RecursiveCharacterTextSplitter but without the dep.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_by_separator(text: str, separator: str, chunk_size: int) -> list[str]:
    """Split text by separator, keeping chunks under chunk_size."""
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)  # character-by-character fallback

    good: list[str] = []
    current = ""

    for split in splits:
        candidate = (current + separator + split).strip() if current else split.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                good.append(current)
            # If a single split is too large, recurse with next separator
            current = split.strip()

    if current:
        good.append(current)

    return [g for g in good if g]


def _chunk_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    metadata: dict,
) -> list[Chunk]:
    """Split text recursively by paragraph → sentence → word → char.

    This strategy tries to keep semantic units together:
    1. First split by \\n\\n (paragraphs)
    2. If a paragraph is still too large, split by \\n (lines)
    3. If a line is still too large, split by ". " (sentences)
    4. Fall back to spaces, then characters

    This is the default strategy for most text content. It produces
    more natural chunks than fixed-size.

    When to use:
    - Documentation, articles, blog posts, web content
    - When you want readable chunks that make sense on their own

    Trade-offs:
    - Pro: Chunks respect paragraph/sentence boundaries
    - Con: Variable chunk size (harder to predict token usage)
    """
    if not text.strip():
        return []

    # Phase 1: split into blocks using separator hierarchy
    blocks: list[str] = [text]
    for sep in _SEPARATORS:
        new_blocks: list[str] = []
        for block in blocks:
            if len(block) <= chunk_size:
                new_blocks.append(block)
            else:
                new_blocks.extend(_split_by_separator(block, sep, chunk_size))
        blocks = new_blocks
        if all(len(b) <= chunk_size for b in blocks):
            break

    # Phase 2: merge small blocks with overlap
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block.strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=current,
                        chunk_index=idx,
                        strategy=ChunkStrategy.RECURSIVE,
                        metadata=dict(metadata),
                    )
                )
                idx += 1
                # Overlap: carry the tail of the previous chunk
                tail = current[-overlap:] if overlap else ""
                current = (tail + "\n\n" + block).strip() if tail else block.strip()
            else:
                current = block.strip()

    if current:
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=current,
                chunk_index=idx,
                strategy=ChunkStrategy.RECURSIVE,
                metadata=dict(metadata),
            )
        )

    logger.debug(
        "Recursive chunking done",
        total_chars=len(text),
        chunk_size=chunk_size,
        chunks=len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Parent-child chunking
# ---------------------------------------------------------------------------


def _chunk_parent_child(
    text: str,
    parent_size: int,
    child_size: int,
    metadata: dict,
) -> list[Chunk]:
    """Two-level chunking: large parents + small children.

    How it works:
    1. Split text into PARENT chunks (large, ~1000 chars)
    2. For each parent, split into CHILD chunks (small, ~200 chars)
    3. Each child gets a ``parent_id`` pointing to its parent
    4. ALL chunks (parent + child) are returned

    At index time:
    - Store BOTH parents and children in Qdrant
    - Mark parents with is_parent=True in payload

    At retrieval time:
    - Query only child chunks (small = more precise match)
    - When a child is retrieved, look up its parent_id
    - Inject the PARENT text into the LLM context (more context)

    This gives you:
    - High PRECISION: small child chunks match queries tightly
    - High CONTEXT: large parent gives the LLM enough surrounding text

    Example:
        Parent (800 chars): Full paragraph about supplier payment terms
        Child A (200 chars): "Payment due within 30 days"
        Child B (200 chars): "Late fees apply after 45 days"

        User query: "when are payments due?"
        → Child A matched (high precision)
        → Parent injected (LLM sees full context)

    When to use:
    - Long reports, contracts, manuals, SAP documentation
    - When individual sentences are too small to provide context
    - When retrieval precision matters most

    Trade-offs:
    - Pro: Best precision + context combination
    - Con: 2x storage, more complex indexing/retrieval logic
    """
    if not text.strip():
        return []

    # Step 1: create parent chunks
    parent_chunks = _chunk_recursive(
        text,
        chunk_size=parent_size,
        overlap=0,  # No overlap between parents (children handle boundaries)
        metadata=metadata,
    )

    all_chunks: list[Chunk] = []

    for parent_chunk in parent_chunks:
        # Reassign parent properties
        parent_chunk.is_parent = True
        parent_chunk.strategy = ChunkStrategy.PARENT_CHILD

        # Step 2: create child chunks from this parent
        child_chunks = _chunk_recursive(
            parent_chunk.text,
            chunk_size=child_size,
            overlap=20,  # Small overlap between children
            metadata=metadata,
        )

        for child in child_chunks:
            child.strategy = ChunkStrategy.PARENT_CHILD
            child.parent_id = parent_chunk.chunk_id
            child.is_parent = False

        all_chunks.append(parent_chunk)
        all_chunks.extend(child_chunks)

    parents = sum(1 for c in all_chunks if c.is_parent)
    children = sum(1 for c in all_chunks if not c.is_parent)
    logger.debug(
        "Parent-child chunking done",
        parents=parents,
        children=children,
        total=len(all_chunks),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Strategy 4: Tokenizer-aware chunking
# ---------------------------------------------------------------------------


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken (OpenAI BPE tokenizer).

    Falls back to character-based estimation (chars / 4) if tiktoken
    is not installed.

    Why token-based chunking?
    - LLMs have a TOKEN limit, not a CHARACTER limit
    - "Hello" = 1 token, but "supercalifragilistic" = ~5 tokens
    - 1 token ≈ 4 chars on average (English), but varies a lot
    - Token-based chunking ensures you never exceed the model's context window
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except ImportError:
        # Fallback: rough approximation
        return max(1, len(text) // 4)


def _chunk_tokenizer(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    metadata: dict,
) -> list[Chunk]:
    """Split text by token count using OpenAI's tiktoken.

    Unlike character-based strategies, this guarantees chunks won't
    exceed the LLM's context window in tokens.

    When to use:
    - When you're feeding chunks directly into a model call (not just retrieval)
    - When working with models that have strict token limits (e.g., 4096)
    - When character-based chunking produces chunks that are too large in tokens

    Trade-offs:
    - Pro: Precise token budget, no context overflow
    - Con: Requires tiktoken, slightly slower (encoding step)
    """
    if not text.strip():
        return []

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        use_tiktoken = True
    except ImportError:
        logger.warning(
            "tiktoken not installed, falling back to char-based estimation "
            "(4 chars ≈ 1 token). Install with: pip install tiktoken"
        )
        # Simulate token list as character positions
        tokens = list(range(len(text) // 4))
        use_tiktoken = False

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        if use_tiktoken:
            chunk_text = enc.decode(chunk_tokens).strip()
        else:
            # Back-calculate character range
            char_start = start * 4
            char_end = min(end * 4, len(text))
            chunk_text = text[char_start:char_end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    chunk_index=idx,
                    strategy=ChunkStrategy.TOKENIZER,
                    metadata={**metadata, "token_count": len(chunk_tokens)},
                )
            )
            idx += 1

        step = max_tokens - overlap_tokens
        if step <= 0:
            break
        start += step

    logger.debug(
        "Tokenizer-aware chunking done",
        total_tokens=len(tokens),
        max_tokens=max_tokens,
        chunks=len(chunks),
    )
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    strategy: ChunkStrategy | str = ChunkStrategy.RECURSIVE,
    *,
    # Shared params
    chunk_size: int = 500,
    overlap: int = 50,
    # Parent-child specific
    parent_size: int = 1000,
    child_size: int = 200,
    # Tokenizer specific
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    # Metadata pass-through
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk a text string using the specified strategy.

    This is the single entry point. The caller picks a strategy and the
    function dispatches to the appropriate implementation.

    Parameters
    ----------
    text:
        The raw text to chunk.
    strategy:
        One of ChunkStrategy enum values or their string equivalents.
        Default: ChunkStrategy.RECURSIVE (best for most content).
    chunk_size:
        Target chunk size in characters. Used by FIXED and RECURSIVE.
    overlap:
        Overlap between consecutive chunks in characters. Used by FIXED
        and RECURSIVE.
    parent_size:
        Parent chunk size in characters. Used by PARENT_CHILD only.
    child_size:
        Child chunk size in characters. Used by PARENT_CHILD only.
    max_tokens:
        Maximum tokens per chunk. Used by TOKENIZER only.
    overlap_tokens:
        Token overlap between chunks. Used by TOKENIZER only.
    metadata:
        Dict of extra metadata to attach to every Chunk (e.g. source URL,
        doc_type, page_num).

    Returns
    -------
    list[Chunk]
        List of Chunk objects ready for embedding and indexing.

    Example
    -------
    >>> chunks = chunk_text("Long document...", strategy="recursive")
    >>> print(f"{len(chunks)} chunks created")
    """
    if metadata is None:
        metadata = {}

    # Normalize strategy to enum
    if isinstance(strategy, str):
        strategy = ChunkStrategy(strategy)

    if strategy == ChunkStrategy.FIXED:
        return _chunk_fixed(text, chunk_size, overlap, metadata)

    elif strategy == ChunkStrategy.RECURSIVE:
        return _chunk_recursive(text, chunk_size, overlap, metadata)

    elif strategy == ChunkStrategy.PARENT_CHILD:
        return _chunk_parent_child(text, parent_size, child_size, metadata)

    elif strategy == ChunkStrategy.TOKENIZER:
        return _chunk_tokenizer(text, max_tokens, overlap_tokens, metadata)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def chunk_pages(
    pages: list,  # list[DocPage]
    strategy: ChunkStrategy | str = ChunkStrategy.RECURSIVE,
    **kwargs,
) -> list[Chunk]:
    """Chunk a list of DocPage objects from doc_loader.

    Convenience wrapper: iterates over pages, chunks each one, and
    attaches page-level metadata (source, page_num, doc_type) to every chunk.

    Parameters
    ----------
    pages:
        Output from src.doc_loader.load_document().
    strategy:
        Chunking strategy to apply.
    **kwargs:
        Extra keyword arguments passed to chunk_text().

    Returns
    -------
    list[Chunk]
        All chunks from all pages, in order.
    """
    all_chunks: list[Chunk] = []

    for page in pages:
        page_metadata = {
            "source": page.source,
            "page_num": page.page_num,
            "doc_type": page.doc_type,
            **page.metadata,
        }

        page_chunks = chunk_text(
            page.content,
            strategy=strategy,
            metadata=page_metadata,
            **kwargs,
        )

        # Offset chunk_index by how many came before
        offset = len(all_chunks)
        for chunk in page_chunks:
            chunk.chunk_index += offset

        all_chunks.extend(page_chunks)

    total_parents = sum(1 for c in all_chunks if c.is_parent)
    total_children = sum(1 for c in all_chunks if not c.is_parent)
    logger.info(
        "Pages chunked",
        pages=len(pages),
        strategy=strategy,
        total_chunks=len(all_chunks),
        parents=total_parents,
        children=total_children,
    )
    return all_chunks


def strategy_summary(strategy: ChunkStrategy | str) -> str:
    """Return a human-readable explanation of the strategy.

    Used in the Gradio UI tooltip / info panel.
    """
    summaries = {
        ChunkStrategy.FIXED: (
            "**Fixed-size**: Splits text every N characters regardless of word "
            "or sentence boundaries. Fast and predictable, but may cut mid-sentence. "
            "Good for homogeneous text."
        ),
        ChunkStrategy.RECURSIVE: (
            "**Recursive**: Tries to split at paragraph boundaries first (\\\\n\\\\n), "
            "then newlines, then sentences. Produces natural chunks that preserve meaning. "
            "Best for articles, documentation, web content."
        ),
        ChunkStrategy.PARENT_CHILD: (
            "**Parent-Child**: Creates two levels — large parent chunks (~1000 chars) "
            "for context and small child chunks (~200 chars) for precise retrieval. "
            "Children are retrieved; parents are injected into the LLM context. "
            "Best for long reports and contracts."
        ),
        ChunkStrategy.TOKENIZER: (
            "**Tokenizer-aware**: Uses OpenAI's tiktoken to split by token count "
            "(not characters). Ensures chunks never exceed the LLM's context window. "
            "Best when feeding chunks directly into model calls."
        ),
    }
    if isinstance(strategy, str):
        strategy = ChunkStrategy(strategy)
    return summaries.get(strategy, "No description available.")
