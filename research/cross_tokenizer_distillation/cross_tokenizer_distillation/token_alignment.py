# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token alignment across different tokenizers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AlignmentChunk:
    """A contiguous text region that maps to whole tokens on both sides.

    Each chunk covers ``[byte_start, byte_end)`` in the original text and
    corresponds to one or more teacher tokens and one or more student tokens.
    """

    byte_start: int
    byte_end: int
    teacher_token_indices: list[int] = field(default_factory=list)
    student_token_indices: list[int] = field(default_factory=list)


@dataclass
class AlignmentResult:
    """Complete alignment between two tokenizations of the same text.

    Attributes:
        text: The original text that was tokenized.
        teacher_token_ids: Token IDs produced by the teacher tokenizer.
        student_token_ids: Token IDs produced by the student tokenizer.
        chunks: Ordered, non-overlapping chunks covering the full text.
    """

    text: str
    teacher_token_ids: list[int] = field(default_factory=list)
    student_token_ids: list[int] = field(default_factory=list)
    chunks: list[AlignmentChunk] = field(default_factory=list)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_teacher_tokens(self) -> int:
        return len(self.teacher_token_ids)

    @property
    def num_student_tokens(self) -> int:
        return len(self.student_token_ids)


def _empty_alignment_stats() -> dict[str, int]:
    return {
        "student_fast_path_hits": 0,
        "student_fast_path_misses": 0,
        "student_visible_piece_path_hits": 0,
        "piece_greedy_hits": 0,
        "piece_span_fallback_hits": 0,
    }


def _get_token_byte_spans(
    text: str,
    tokenizer,
    token_ids: list[int] | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Get byte-level spans for each token.

    Tries ``tokenizer(text, return_offsets_mapping=True)`` first (fast
    tokenizers).  Falls back to decoding each token individually and matching
    against the text.

    Returns:
        (token_ids, spans) where spans[i] = (byte_start, byte_end).
    """
    # --- fast path: offset_mapping -----------------------------------------
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        # offset_mapping is in *character* offsets for HF tokenizers.
        # Convert to byte offsets via the encoded text.
        text_bytes = text.encode("utf-8")
        char_to_byte = []
        byte_idx = 0
        for ch in text:
            char_to_byte.append(byte_idx)
            byte_idx += len(ch.encode("utf-8"))
        char_to_byte.append(byte_idx)  # sentinel for end

        byte_spans: list[tuple[int, int]] = []
        for start_char, end_char in offsets:
            byte_spans.append((char_to_byte[start_char], char_to_byte[end_char]))
        return ids, byte_spans
    except Exception:
        pass

    # --- slow path: decode each token --------------------------------------
    if token_ids is None:
        enc = tokenizer(text, add_special_tokens=False)
        token_ids = enc["input_ids"]

    text_bytes = text.encode("utf-8")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tid in token_ids:
        decoded = tokenizer.decode([tid])
        token_bytes = decoded.encode("utf-8")
        # find in remaining text
        idx = text_bytes.find(token_bytes, cursor)
        if idx == -1:
            # fallback: assign zero-width span at cursor
            spans.append((cursor, cursor))
        else:
            spans.append((idx, idx + len(token_bytes)))
            cursor = idx + len(token_bytes)
    return token_ids, spans


def get_visible_pieces_from_original_ids(
    tokenizer,
    original_ids: list[int],
) -> tuple[list[int], list[bytes], list[int]]:
    """Extract visible text pieces from *original* generated token IDs.

    Uses prefix decoding with ``skip_special_tokens=True`` so the pieces
    exactly reproduce the text obtained by ``tokenizer.decode(original_ids,
    skip_special_tokens=True)``.  Tokens that contribute no visible text
    (EOS, BOS, ``<unused*>``, …) are dropped.

    Returns:
        ``(visible_ids, pieces, original_indices)`` where
        ``original_indices[i]`` is the position of ``visible_ids[i]``
        in *original_ids*.
    """
    visible_ids: list[int] = []
    pieces: list[bytes] = []
    original_indices: list[int] = []

    prefix_ids: list[int] = []
    prev_decoded = ""
    for i, tid in enumerate(original_ids):
        prefix_ids.append(tid)
        decoded = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded) > len(prev_decoded) and decoded.startswith(prev_decoded):
            piece_text = decoded[len(prev_decoded):]
            visible_ids.append(tid)
            pieces.append(piece_text.encode("utf-8"))
            original_indices.append(i)
            prev_decoded = decoded
        # Tokens that don't grow the decoded string are invisible (special)
        # and are silently skipped.

    return visible_ids, pieces, original_indices


def align_tokens_from_original_student_ids(
    text: str,
    teacher_tokenizer,
    student_tokenizer,
    original_student_token_ids: list[int],
) -> AlignmentResult:
    alignment, _ = align_tokens_from_original_student_ids_with_stats(
        text=text,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        original_student_token_ids=original_student_token_ids,
    )
    return alignment


def align_tokens_from_original_student_ids_with_stats(
    text: str,
    teacher_tokenizer,
    student_tokenizer,
    original_student_token_ids: list[int],
) -> tuple[AlignmentResult, dict[str, int]]:
    """Align decoded text while preserving indices into the original student ids."""
    stats = _empty_alignment_stats()
    if not text or not original_student_token_ids:
        return (
            AlignmentResult(
                text=text,
                teacher_token_ids=[],
                student_token_ids=list(original_student_token_ids),
                chunks=[],
            ),
            stats,
        )

    special_ids = set(getattr(student_tokenizer, "all_special_ids", []) or [])
    visible_original_ids = [
        tid for tid in original_student_token_ids if tid not in special_ids
    ]
    visible_original_indices = [
        i for i, tid in enumerate(original_student_token_ids) if tid not in special_ids
    ]

    # Fast path: if the visible text re-tokenizes to the same visible student ids
    # (ignoring skipped special tokens), align directly from byte spans and remap
    # chunk indices back to the original generated sequence.
    try:
        roundtrip_student_ids, roundtrip_student_spans = _get_token_byte_spans(
            text, student_tokenizer
        )
        if roundtrip_student_ids == visible_original_ids:
            teacher_ids, teacher_spans = _get_token_byte_spans(text, teacher_tokenizer)
            alignment = _build_alignment_from_spans(
                text=text,
                teacher_ids=teacher_ids,
                teacher_spans=teacher_spans,
                student_ids=list(original_student_token_ids),
                student_spans=roundtrip_student_spans,
            )
            for chunk in alignment.chunks:
                chunk.student_token_indices = [
                    visible_original_indices[idx] for idx in chunk.student_token_indices
                ]
            alignment.student_token_ids = list(original_student_token_ids)
            stats["student_fast_path_hits"] += 1
            return alignment, stats
    except Exception:
        pass
    stats["student_fast_path_misses"] += 1

    visible_ids, visible_pieces, original_indices = get_visible_pieces_from_original_ids(
        student_tokenizer, original_student_token_ids
    )
    if not visible_ids:
        return (
            AlignmentResult(
                text=text,
                teacher_token_ids=[],
                student_token_ids=list(original_student_token_ids),
                chunks=[],
            ),
            stats,
        )

    alignment, piece_stats = align_tokens_by_decoded_pieces(
        text,
        teacher_tokenizer,
        student_tokenizer,
        student_token_ids=visible_ids,
        student_pieces=visible_pieces,
        return_stats=True,
    )
    stats["student_visible_piece_path_hits"] += 1
    for key, value in piece_stats.items():
        stats[key] = stats.get(key, 0) + value
    for chunk in alignment.chunks:
        chunk.student_token_indices = [
            original_indices[idx] for idx in chunk.student_token_indices
        ]
    alignment.student_token_ids = list(original_student_token_ids)
    return alignment, stats


def _get_piece_byte_spans(pieces: list[bytes]) -> list[tuple[int, int]]:
    """Convert sequential decoded pieces into byte spans over the shared text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        next_cursor = cursor + len(piece)
        spans.append((cursor, next_cursor))
        cursor = next_cursor
    return spans


def _build_alignment_from_spans(
    text: str,
    teacher_ids: list[int],
    teacher_spans: list[tuple[int, int]],
    student_ids: list[int],
    student_spans: list[tuple[int, int]],
) -> AlignmentResult:
    """Build alignment chunks from byte spans on both sides."""

    teacher_end_map: dict[int, list[int]] = {}
    for idx, (_, end) in enumerate(teacher_spans):
        teacher_end_map.setdefault(end, []).append(idx)

    student_end_map: dict[int, list[int]] = {}
    for idx, (_, end) in enumerate(student_spans):
        student_end_map.setdefault(end, []).append(idx)

    all_boundaries = sorted(
        set(
            [s for s, _ in teacher_spans]
            + [e for _, e in teacher_spans]
            + [s for s, _ in student_spans]
            + [e for _, e in student_spans]
        )
    )

    chunks: list[AlignmentChunk] = []
    chunk_start = all_boundaries[0] if all_boundaries else 0
    pending_teacher: list[int] = []
    pending_student: list[int] = []

    for boundary in all_boundaries:
        if boundary in teacher_end_map:
            pending_teacher.extend(teacher_end_map[boundary])
        if boundary in student_end_map:
            pending_student.extend(student_end_map[boundary])

        if pending_teacher and pending_student:
            chunks.append(
                AlignmentChunk(
                    byte_start=chunk_start,
                    byte_end=boundary,
                    teacher_token_indices=sorted(pending_teacher),
                    student_token_indices=sorted(pending_student),
                )
            )
            chunk_start = boundary
            pending_teacher = []
            pending_student = []

    if pending_teacher or pending_student:
        if chunks:
            last = chunks[-1]
            last.byte_end = all_boundaries[-1] if all_boundaries else chunk_start
            last.teacher_token_indices.extend(pending_teacher)
            last.student_token_indices.extend(pending_student)
        elif pending_teacher and pending_student:
            chunks.append(
                AlignmentChunk(
                    byte_start=chunk_start,
                    byte_end=all_boundaries[-1] if all_boundaries else chunk_start,
                    teacher_token_indices=sorted(pending_teacher),
                    student_token_indices=sorted(pending_student),
                )
            )

    return AlignmentResult(
        text=text,
        teacher_token_ids=teacher_ids,
        student_token_ids=student_ids,
        chunks=chunks,
    )


def _align_from_piece_spans(
    text: str,
    teacher_ids: list[int],
    teacher_pieces: list[bytes],
    student_ids: list[int],
    student_pieces: list[bytes],
) -> AlignmentResult:
    """Align two token-piece sequences using byte spans without re-tokenizing."""
    return _build_alignment_from_spans(
        text=text,
        teacher_ids=teacher_ids,
        teacher_spans=_get_piece_byte_spans(teacher_pieces),
        student_ids=student_ids,
        student_spans=_get_piece_byte_spans(student_pieces),
    )


def _get_token_ids_and_pieces(
    text: str,
    tokenizer,
) -> tuple[list[int], list[bytes]]:
    """Tokenize text and recover per-token visible text pieces as UTF-8 bytes."""
    enc = tokenizer(text, add_special_tokens=False)
    token_ids = enc["input_ids"]
    pieces: list[bytes] = []

    prefix_ids: list[int] = []
    prev_decoded = ""
    for token_id in token_ids:
        prefix_ids.append(token_id)
        decoded = tokenizer.decode(
            prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded.startswith(prev_decoded):
            piece = decoded[len(prev_decoded) :]
        else:
            piece = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        pieces.append(piece.encode("utf-8"))
        prev_decoded = decoded

    return token_ids, pieces


def _get_token_ids_and_pieces_from_spans(
    text: str,
    tokenizer,
) -> tuple[list[int], list[bytes]]:
    """Tokenize text and recover per-token pieces directly from byte spans."""
    token_ids, spans = _get_token_byte_spans(text, tokenizer)
    text_bytes = text.encode("utf-8")
    pieces = [text_bytes[start:end] for start, end in spans]
    return token_ids, pieces


def align_tokens_by_byte_offset(
    text: str,
    teacher_tokenizer,
    student_tokenizer,
) -> AlignmentResult:
    """Align two tokenizations of the same text via byte offsets.

    Algorithm:
        1. Tokenize *text* with both tokenizers → byte spans per token.
        2. Collect all boundary points from both span sets → sorted unique set.
        3. Greedily merge adjacent sub-intervals so that each resulting
           *chunk* contains at least one complete token from each side.

    Returns:
        An ``AlignmentResult`` with the ordered chunks.
    """
    teacher_ids, teacher_spans = _get_token_byte_spans(text, teacher_tokenizer)
    student_ids, student_spans = _get_token_byte_spans(text, student_tokenizer)

    return _build_alignment_from_spans(
        text=text,
        teacher_ids=teacher_ids,
        teacher_spans=teacher_spans,
        student_ids=student_ids,
        student_spans=student_spans,
    )


def align_tokens_by_decoded_pieces(
    text: str,
    teacher_tokenizer,
    student_tokenizer,
    student_token_ids: list[int] | None = None,
    student_pieces: list[bytes] | None = None,
    return_stats: bool = False,
) -> AlignmentResult | tuple[AlignmentResult, dict[str, int]]:
    """Align two tokenizations by greedily matching decoded token-piece groups.

    This follows the same spirit as GOLD: each side accumulates decoded token
    pieces until both represent the same visible text span, then emits one
    shared alignment group.

    When *student_token_ids* and *student_pieces* are provided they are used
    directly (no re-tokenisation on the student side).  This is critical for
    correctness: the indices in the returned chunks then reference positions in
    the **original** generated sequence rather than a re-encoded copy.
    """
    stats = {"piece_greedy_hits": 0, "piece_span_fallback_hits": 0}
    teacher_ids, teacher_pieces = _get_token_ids_and_pieces_from_spans(text, teacher_tokenizer)
    if student_token_ids is not None and student_pieces is not None:
        student_ids = student_token_ids
    else:
        student_ids, student_pieces = _get_token_ids_and_pieces(text, student_tokenizer)

    chunks: list[AlignmentChunk] = []
    matched_bytes = 0
    teacher_idx = 0
    student_idx = 0
    teacher_buf = b""
    student_buf = b""
    teacher_group: list[int] = []
    student_group: list[int] = []

    while teacher_idx < len(teacher_pieces) or student_idx < len(student_pieces):
        if teacher_buf == student_buf and teacher_buf and teacher_group and student_group:
            span_len = len(teacher_buf)
            chunks.append(
                AlignmentChunk(
                    byte_start=matched_bytes,
                    byte_end=matched_bytes + span_len,
                    teacher_token_indices=list(teacher_group),
                    student_token_indices=list(student_group),
                )
            )
            matched_bytes += span_len
            teacher_buf = b""
            student_buf = b""
            teacher_group = []
            student_group = []

        take_teacher = (
            student_idx >= len(student_pieces)
            or (teacher_idx < len(teacher_pieces) and len(teacher_buf) <= len(student_buf))
        )
        if take_teacher and teacher_idx < len(teacher_pieces):
            teacher_buf += teacher_pieces[teacher_idx]
            teacher_group.append(teacher_idx)
            teacher_idx += 1
        elif student_idx < len(student_pieces):
            student_buf += student_pieces[student_idx]
            student_group.append(student_idx)
            student_idx += 1

    if teacher_buf == student_buf and teacher_buf and teacher_group and student_group:
        span_len = len(teacher_buf)
        chunks.append(
            AlignmentChunk(
                byte_start=matched_bytes,
                byte_end=matched_bytes + span_len,
                teacher_token_indices=list(teacher_group),
                student_token_indices=list(student_group),
            )
        )
    elif teacher_buf or student_buf or teacher_group or student_group:
        # Greedy matching failed. Fall back to byte-span alignment over the
        # already-decoded token pieces so we preserve the caller-provided
        # student token index space.
        stats["piece_span_fallback_hits"] += 1
        alignment = _align_from_piece_spans(
            text=text,
            teacher_ids=teacher_ids,
            teacher_pieces=teacher_pieces,
            student_ids=student_ids,
            student_pieces=student_pieces,
        )
        return (alignment, stats) if return_stats else alignment

    stats["piece_greedy_hits"] += 1
    alignment = AlignmentResult(
        text=text,
        teacher_token_ids=teacher_ids,
        student_token_ids=student_ids,
        chunks=chunks,
    )
    return (alignment, stats) if return_stats else alignment


def compute_chunk_logprobs(
    token_logprobs: torch.Tensor,
    chunks: list[AlignmentChunk],
    side: str,
) -> torch.Tensor:
    """Aggregate per-token log-probs into per-chunk log-probs.

    Args:
        token_logprobs: Shape ``(seq_len,)`` — log-prob of each token.
        chunks: Alignment chunks from ``align_tokens_by_byte_offset``.
        side: ``"teacher"`` or ``"student"`` — which indices to use.

    Returns:
        Tensor of shape ``(num_chunks,)`` with summed log-probs per chunk.
    """
    chunk_lps = []
    for chunk in chunks:
        indices = (
            chunk.teacher_token_indices
            if side == "teacher"
            else chunk.student_token_indices
        )
        if indices:
            chunk_lps.append(token_logprobs[indices].sum())
        else:
            chunk_lps.append(torch.tensor(0.0, device=token_logprobs.device))
    if not chunk_lps:
        return torch.zeros(0, device=token_logprobs.device)
    return torch.stack(chunk_lps)


def merge_alignment_chunks(
    alignment: AlignmentResult,
    min_bytes: int = 0,
) -> AlignmentResult:
    """Merge adjacent alignment chunks into coarser shared text spans.

    Minimal byte-level chunks are often too fine for cross-tokenizer training and
    create thousands of boundaries per sample. Coarsening them gives a more
    stable shared event space for sequence-level distillation.
    """
    if min_bytes <= 0 or alignment.num_chunks <= 1:
        return alignment

    merged_chunks: list[AlignmentChunk] = []
    pending: AlignmentChunk | None = None

    for chunk in alignment.chunks:
        if pending is None:
            pending = AlignmentChunk(
                byte_start=chunk.byte_start,
                byte_end=chunk.byte_end,
                teacher_token_indices=list(chunk.teacher_token_indices),
                student_token_indices=list(chunk.student_token_indices),
            )
        else:
            pending.byte_end = chunk.byte_end
            pending.teacher_token_indices.extend(chunk.teacher_token_indices)
            pending.student_token_indices.extend(chunk.student_token_indices)

        if pending.byte_end - pending.byte_start >= min_bytes:
            merged_chunks.append(pending)
            pending = None

    if pending is not None:
        if merged_chunks:
            merged_chunks[-1].byte_end = pending.byte_end
            merged_chunks[-1].teacher_token_indices.extend(pending.teacher_token_indices)
            merged_chunks[-1].student_token_indices.extend(pending.student_token_indices)
        else:
            merged_chunks.append(pending)

    return AlignmentResult(
        text=alignment.text,
        teacher_token_ids=alignment.teacher_token_ids,
        student_token_ids=alignment.student_token_ids,
        chunks=merged_chunks,
    )


def batch_align(
    texts: list[str],
    teacher_tokenizer,
    student_tokenizer,
) -> list[AlignmentResult]:
    """Align a batch of texts."""
    return [
        align_tokens_by_byte_offset(text, teacher_tokenizer, student_tokenizer)
        for text in texts
    ]
