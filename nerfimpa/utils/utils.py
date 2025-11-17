import torch
import torch.nn.functional as F

def to_gray(img: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor to grayscale [H, W] float32.

    Supports shapes:
      - [H, W, C]   (C=1 or 3)
    """
    if img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = img[..., 0]
    return gray.float()


def fft2_power_spectrum_gray(gray: torch.Tensor) -> torch.Tensor:
    """gray: [H,W] float.
    Returns [H,W] in [0,1] representing log power spectrum (low freq in center).
    """
    F = torch.fft.fft2(gray)
    F_shift = torch.fft.fftshift(F)

    P = (F_shift.real ** 2 + F_shift.imag ** 2)

    P_log = torch.log1p(P)  # log(1 + P) to avoid log(0)

    P_min = P_log.min()
    P_max = P_log.max()
    spec = (P_log - P_min) / (P_max - P_min + 1e-8)

    return spec


def gray_to_rgb(gray: torch.Tensor) -> torch.Tensor:
    return gray.unsqueeze(-1).repeat(1, 1, 3)


def laplacian_2d(img: torch.Tensor) -> torch.Tensor:
    """
    img: [H, W] (grayscale)
    returns: Laplacian image [H, W]
    """
    if img.dim() != 2:
        raise ValueError(f"Expected [H, W], got {img.shape}")

    H, W = img.shape

    kernel = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3)  # [out_channels, in_channels, kH, kW]

    img_bchw = img.unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(img_bchw, kernel, padding=1)  # [1,1,H,W]
    lap = lap.squeeze(0).squeeze(0)              # [H,W]

    lap_abs  = lap.abs()
    lap_norm = (lap_abs - lap_abs.min()) / (lap_abs.max() - lap_abs.min() + 1e-8)

    return lap_norm
