import torch
import torch.nn as nn


class LANGER(nn.Module):
    """Protein LM + ion (+ optional GGL) fusion network for REE binding affinity."""

    def __init__(
        self,
        prot_input_dim: int = 1280,
        ion_input_dim: int = 11,
        ggl_input_dim: int = 0,
        prot_latent_dim: int = 512,
        ion_latent_dim: int = 128,
        ggl_latent_dim: int = 256,
        ffn_hidden_dims: tuple[int, int, int, int] = (768, 384, 128, 32),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.prot_input_dim = prot_input_dim
        self.ion_input_dim = ion_input_dim
        self.ggl_input_dim = ggl_input_dim
        self.use_prot = prot_input_dim > 0
        self.use_ggl = ggl_input_dim > 0

        latent_chunks = []
        if self.use_prot:
            self.prot_norm = nn.LayerNorm(prot_input_dim)
            self.prot_linear = nn.Linear(prot_input_dim, prot_latent_dim)
            latent_chunks.append(prot_latent_dim)

        self.ion_norm = nn.LayerNorm(ion_input_dim)
        self.ion_linear = nn.Linear(ion_input_dim, ion_latent_dim)
        latent_chunks.append(ion_latent_dim)

        if self.use_ggl:
            self.ggl_norm = nn.LayerNorm(ggl_input_dim)
            self.ggl_linear = nn.Linear(ggl_input_dim, ggl_latent_dim)
            latent_chunks.append(ggl_latent_dim)

        total_latent_dim = sum(latent_chunks)
        self.fusion_norm = nn.LayerNorm(total_latent_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        h1, h2, h3, h4 = ffn_hidden_dims
        self.linear1 = nn.Linear(total_latent_dim, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, h3)
        self.linear4 = nn.Linear(h3, h4)
        self.final_linear = nn.Linear(h4, 1)

    def _encode_branches(self, prot, ion, ggl=None):
        latent_chunks: list[torch.Tensor] = []
        if self.use_prot:
            latent_chunks.append(torch.relu(self.prot_linear(self.prot_norm(prot))))
        latent_chunks.append(torch.relu(self.ion_linear(self.ion_norm(ion))))
        if ggl is not None and self.use_ggl:
            latent_chunks.append(torch.relu(self.ggl_linear(self.ggl_norm(ggl))))
        return torch.cat(latent_chunks, dim=1)

    def forward(self, prot, ion, ggl=None):
        x = self.dropout(self.fusion_norm(self._encode_branches(prot, ion, ggl)))
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        return self.final_linear(x)
