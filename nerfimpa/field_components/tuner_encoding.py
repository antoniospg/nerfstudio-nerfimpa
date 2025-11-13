import torch
import math
from torch import nn, Tensor
from nerfstudio.field_components.encodings import Encoding

def sample_high_int_freqs(m_high: int, d: int, B: int, b: int) -> torch.Tensor:
    K_high = []
    while len(K_high) < m_high:
        draw = m_high - len(K_high)
        cand = torch.randint(-B, B + 1, (draw, d), dtype=torch.int32)
        mask = cand.abs().amax(dim=1) > b
        taken = cand[mask]
        if taken.numel():
            K_high.append(taken)
    return torch.cat(K_high, dim=0)[:m_high]


def sample_integer_frequencies(
    d: int,
    m: int,
    B: int,
    b: int,
    low_frac: float,
) -> tuple[Tensor, Tensor, Tensor]:
    assert 0 < b <= B
    assert 0.0 < low_frac < 1.0

    m_low = max(1, int(round(low_frac * m)))
    m_high = m - m_low

    K_low = torch.randint(-b, b+1, (m_low, d), dtype=torch.int32)
    K_high = sample_high_int_freqs(m_high, d, B, b) if m_high > 0 else torch.empty(0, d, dtype=torch.int32)
    K = torch.cat([K_low, K_high], dim=0).to(torch.float32)

    omega = (2.0 * math.pi) * K
    phi = (torch.rand(K.shape[0]) - 0.5) * math.pi
    is_low = (K.abs().amax(dim=1) <= b)

    return omega, phi, is_low.bool()


class TunerEncoding(Encoding):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_width: int = 256,
        m: int = 128,
        B: int = 64,
        b: int = 21,
        low_frac: float = 0.7,
        learned_bounds: bool = True,
        c_low: float = 1.0,
        c_high: float = 0.05,
        reg_lambda: float = 0.05,
        include_input: bool = False,
    ):
        super().__init__(in_dim=in_dim)

        self.include_input = include_input

        omega, phi, is_low = sample_integer_frequencies(
            d=in_dim, m=m, B=B, b=b, low_frac=low_frac
        )

        self.register_buffer("omega", omega)
        self.register_buffer("phi", phi)
        self.register_buffer("is_low_mask", is_low)
        self.m = m

        self.hidden_width = hidden_width
        self._out_dim = hidden_width
        self.reg_lambda = reg_lambda

        if learned_bounds:
            self.W_raw = nn.Parameter(torch.empty(hidden_width, m))
            nn.init.normal_(self.W_raw, mean=0.0, std=0.33)
            c = torch.full((m,), float(c_high))
            c[is_low] = float(c_low)
            self.c = nn.Parameter(c)
            self._learned = True
        else:
            self.W = nn.Parameter(torch.empty(hidden_width, m))
            nn.init.normal_(self.W_raw, mean=0.0, std=0.1)
            cb = torch.full((m,), float(c_high))
            cb[is_low] = float(c_low)
            self.register_buffer("col_bounds", cb)
            self._learned = False

        self.b = nn.Parameter(torch.zeros(hidden_width))


    def forward(self, x: Tensor) -> Tensor:
        x_flat = x.reshape(-1, self.omega.shape[1])
        D = torch.sin(x_flat @ self.omega.T + self.phi)

        if self._learned:
            W_eff = torch.tanh(self.W_raw) * self.c.unsqueeze(0)
        else:
            W_eff = torch.clamp(self.W, -self.col_bounds, self.col_bounds)

        H = torch.sin(D @ W_eff.T + self.b)
        encoded_inputs =  H.reshape(*x.shape[:-1], self._out_dim)

        if self.include_input:
            return torch.cat([encoded_inputs, x], dim=-1)
        else:
            return encoded_inputs


    def get_out_dim(self) -> int:
        if self.include_input:
            return self._out_dim + self.in_dim
        else:
            return self._out_dim


    def regularizer(self) -> Tensor:
        if self._learned:
            return self.reg_lambda * self.c.abs().sum()
        return torch.zeros((), device=self.b.device)
