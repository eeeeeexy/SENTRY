import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Estimator(nn.Module):
    def __init__(self, seq_model, img_model, output_dims, *args) -> None:
        super().__init__()
        self.seq_model = seq_model
        self.img_model = img_model
        self.output_dims = output_dims

        self.fc_merge = nn.Linear(128 + 16, self.output_dims)

    def forward(self, traj_sample, data_map19):
        # traj model
        x_traj = self.seq_model(traj_sample)

        # map model
        x_img = self.img_model(data_map19)

        x = self.fc_merge(torch.concat((x_traj, x_img), 1))

        return x
    


class Align_Model(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.args = args

    def get_attention_scores(self, seq_traj):

        d_k = Q.size(-1)
        attn_matrix = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_matrix = torch.softmax(attn_matrix, dim=-1)
        
        scores = attn_matrix.mean(dim=1) 
        return scores


    def forward(self, seq_traj=None, img_traj=None, pixel_index=None, compressed_size=None):

        k = compressed_size 
        batch_size, seq_len, _ = seq_traj.shape
        repeat_times = seq_len // k

        scores = self.get_attention_scores(seq_traj)

        _, topk_indices = torch.topk(scores, k=k, dim=1)

        topk_indices, _ = torch.sort(topk_indices, dim=1)


        row_idx = torch.arange(batch_size).view(-1, 1).to(seq_traj.device)

        compressed_seq = seq_traj[row_idx, topk_indices]

        repeated_seq = compressed_seq.repeat_interleave(repeat_times, dim=1)

        compressed_pixels = pixel_index[row_idx, topk_indices]

        out_img = torch.zeros_like(img_traj)
        b_idx = torch.arange(batch_size).view(batch_size, 1).expand(batch_size, k)
        x_coords = compressed_pixels[:, :, 0].long()
        y_coords = compressed_pixels[:, :, 1].long()

        for c in range(out_img.shape[1]):
            out_img[b_idx, c, x_coords, y_coords] = img_traj[b_idx, c, x_coords, y_coords]

        return repeated_seq, out_img

