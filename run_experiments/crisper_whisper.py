from typing import Any, Optional

import torch
from transformers import WhisperForConditionalGeneration


class WhisperForConditionalGenerationWithAttentionLoss(WhisperForConditionalGeneration):

    generation_config = None

    def average_heads(self, cross_attentions):
        heads = self.generation_config.alignment_heads
        # Initialize an accumulator tensor with zeros with the shape of the 0th head's attention from the first tensor
        accumulator = torch.zeros_like(cross_attentions[0][:, 0, :, :])

        for head in heads:
            accumulator += cross_attentions[head[0]].to(torch.float32)[:, head[1], :, :]
        # Calculate the average across all tensors in the list
        average_across_list = accumulator / len(heads)

        return average_across_list

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask_for_loss: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Any:
        original_output = super().forward(
            input_features,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            decoder_inputs_embeds,
            decoder_position_ids,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        cross_attentions = original_output.cross_attentions

        if original_output.loss is not None and attention_labels is not None:
            cross_attention_matrices_b_t_a = self.average_heads(cross_attentions).to(
                torch.float32
            )
            standardized_attentions_b_t_a = cross_attention_matrices_b_t_a
            standardized_attentions_b_t_a = torch.mul(
                standardized_attentions_b_t_a, attention_mask_for_loss
            )  # ignore attentions that are far out from words...
            norms = (
                torch.norm(standardized_attentions_b_t_a, p=2, dim=-1, keepdim=True)
                + 1e-8
            )
            l2_normalized_attentions = standardized_attentions_b_t_a / norms
            gt_attention_matrix = attention_labels
            norms_gt = torch.norm(gt_attention_matrix, p=2, dim=-1, keepdim=True) + 1e-8
            l2_normalized_gt = gt_attention_matrix / norms_gt
            similarity = torch.bmm(
                l2_normalized_attentions, l2_normalized_gt.transpose(1, 2)
            )
            batch_size, token_size, _ = cross_attention_matrices_b_t_a.shape
            identity_shapes = torch.sum(labels != -100, dim=1)
            for b in range(
                cross_attention_matrices_b_t_a.shape[0]
            ):  # Iterate over the batch dimension
                for t in range(
                    cross_attention_matrices_b_t_a.shape[1]
                ):  # Iterate over the token dimension
                    # Zero out rows beyond the max row specified by label_dimension for each [b, t] pair
                    max_row = identity_shapes[b]
                    cross_attention_matrices_b_t_a[b, max_row:, :] = 0
            # Create a batch of identity matrices

            identity_matrices = (
                torch.eye(token_size)
                .repeat(batch_size, 1, 1)
                .to(standardized_attentions_b_t_a.device)
            )
            # Create a mask for the diagonal elements based on how many tokens are in the ground truth
            for i, dim in enumerate(identity_shapes):
                # Generate a diagonal matrix with 1s up to the specified dim for each matrix in the batch
                diag_matrix = torch.diag(torch.ones(dim)).to(identity_matrices.device)
                # Expand dimensions to fit the batch shape and token size
                padded_diag_matrix = torch.nn.functional.pad(
                    diag_matrix, (0, token_size - dim, 0, token_size - dim)
                )
                # Update the identity_matrices with the new diagonal matrix for the specific batch index
                identity_matrices[i] = padded_diag_matrix
            diff = identity_matrices - similarity
            traces = torch.einsum("bii->b", diff)
            num_tokens = torch.sum(labels != -100)
            attention_loss = traces.sum() / num_tokens
            print(f"attention_loss: {attention_loss}")
            original_output.loss += 0.12 * attention_loss

        return original_output
