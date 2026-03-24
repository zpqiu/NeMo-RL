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

"""Token alignment across different tokenizers via byte-offset mapping."""

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

    # Build mapping: byte_boundary → set of token indices that END at that boundary.
    # We use *end* boundaries because a chunk can only be closed when both sides
    # have completed at least one token.
    teacher_end_map: dict[int, list[int]] = {}
    for idx, (_, end) in enumerate(teacher_spans):
        teacher_end_map.setdefault(end, []).append(idx)

    student_end_map: dict[int, list[int]] = {}
    for idx, (_, end) in enumerate(student_spans):
        student_end_map.setdefault(end, []).append(idx)

    # Collect all byte boundaries and sort
    all_boundaries = sorted(
        set(
            [s for s, _ in teacher_spans]
            + [e for _, e in teacher_spans]
            + [s for s, _ in student_spans]
            + [e for _, e in student_spans]
        )
    )

    # Greedy merge: walk boundaries, accumulate tokens, emit chunk when both
    # sides have ≥1 token.
    chunks: list[AlignmentChunk] = []
    chunk_start = all_boundaries[0] if all_boundaries else 0
    pending_teacher: list[int] = []
    pending_student: list[int] = []

    for boundary in all_boundaries:
        if boundary in teacher_end_map:
            pending_teacher.extend(teacher_end_map[boundary])
        if boundary in student_end_map:
            pending_student.extend(student_end_map[boundary])

        # Emit a chunk when both sides have accumulated ≥1 token
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

    # Flush any remaining tokens into the last chunk
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
