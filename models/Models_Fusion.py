import torch
import torch.nn as nn

class Align_Model(nn.Module):

    def __init__(self, input_dim=7, attn_dim=32):
        super().__init__()

        self.q_proj = nn.Linear(input_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, attn_dim, bias=False)

        self.use_bpda = False

    def get_attention_scores(self, seq_traj):
        # seq_traj: [B, T, D]
        Q = self.q_proj(seq_traj)
        K = self.k_proj(seq_traj)

        d_k = Q.size(-1)
        attn_matrix = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / (d_k ** 0.5)

        attn_matrix = torch.softmax(attn_matrix, dim=-1)
        scores = attn_matrix.mean(dim=1)  # [B, T]

        return scores

    def forward(
        self,
        seq_traj=None,
        img_traj=None,
        pixel_index=None,
        compressed_size=None,
    ):
        k = compressed_size
        batch_size, seq_len, _ = seq_traj.shape
        repeat_times = seq_len // k

        scores = self.get_attention_scores(seq_traj)

        _, topk_indices = torch.topk(scores, k=k, dim=1)

        topk_indices, _ = torch.sort(topk_indices, dim=1)

        row_idx = torch.arange(
            batch_size,
            device=seq_traj.device,
        ).view(-1, 1)

        compressed_seq = seq_traj[row_idx, topk_indices]

        repeated_seq = compressed_seq.repeat_interleave(
            repeat_times,
            dim=1,
        )

        compressed_pixels = pixel_index[row_idx, topk_indices]

        out_img = torch.zeros_like(img_traj)

        b_idx = torch.arange(
            batch_size,
            device=img_traj.device,
        ).view(batch_size, 1).expand(batch_size, k)

        x_coords = compressed_pixels[:, :, 0].long()
        y_coords = compressed_pixels[:, :, 1].long()

        for c in range(out_img.shape[1]):
            out_img[b_idx, c, x_coords, y_coords] = (
                img_traj[b_idx, c, x_coords, y_coords]
            )

        return repeated_seq, out_img
