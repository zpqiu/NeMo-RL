from __future__ import annotations

import torch

from cross_tokenizer_distillation.gold_algorithm import pack_gold_alignment_into_data
from cross_tokenizer_distillation.gold_loss import VocabMapping
from cross_tokenizer_distillation.token_alignment import AlignmentChunk, AlignmentResult


def test_pack_gold_alignment_uses_predictor_slots_and_safe_unmatched_indices():
    alignment = AlignmentResult(
        text="hello",
        teacher_token_ids=[10],
        student_token_ids=[20],
        chunks=[
            AlignmentChunk(
                byte_start=0,
                byte_end=5,
                teacher_token_indices=[0],
                student_token_indices=[0],
            )
        ],
    )
    vocab_mapping = VocabMapping(
        matched_student_ids=[1],
        matched_teacher_ids=[0],
        student_matched_mask=torch.tensor([False, True, False]),
        teacher_matched_mask=torch.tensor([True, False, False, False, False, True]),
        teacher_to_student_map={0: 1},
        mapping_tensor=torch.tensor([1]),
        num_matched=1,
        student_vocab_size=3,
        teacher_vocab_size=6,
        jaccard_index=1 / 8,
    )

    teacher_topk_logits = torch.zeros(1, 6, 2)
    teacher_topk_logits[0, 4] = torch.tensor([3.0, 1.0])
    teacher_topk_indices = torch.zeros(1, 6, 2, dtype=torch.long)
    teacher_topk_indices[0, 4] = torch.tensor([0, 5], dtype=torch.long)

    packed = pack_gold_alignment_into_data(
        alignments=[alignment],
        teacher_topk_logits=teacher_topk_logits,
        teacher_topk_indices=teacher_topk_indices,
        teacher_gen_logprobs=[torch.zeros(1)],
        student_prev_logprobs=torch.zeros(1, 5),
        teacher_input_lengths=[5],
        student_prompt_lengths=[3],
        seq_len=6,
        topk_k=2,
        vocab_mapping=vocab_mapping,
    )

    predictor_slot = 2  # prompt_len - 1 + first_generated_pos
    assert packed["gold_position_mask"][0, predictor_slot].item() == 1.0
    assert packed["gold_position_mask"][0, predictor_slot + 1].item() == 0.0
    assert torch.equal(
        packed["teacher_topk_logits"][0, predictor_slot],
        torch.tensor([3.0, 1.0]),
    )
    assert torch.equal(
        packed["gold_teacher_topk_indices_original"][0, predictor_slot],
        torch.tensor([0, 5]),
    )
    assert torch.equal(
        packed["teacher_topk_indices"][0, predictor_slot],
        torch.tensor([1, 0]),
    )
