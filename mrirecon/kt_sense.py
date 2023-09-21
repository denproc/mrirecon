r"""
   Copyright 2022 Denis Prokopenko

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import torch
from mrirecon.functional.domain_conversion import kt2xf, k2x, x2k, f2t
from mrirecon.sense import sense


def slice_wise_kt_sense(
    kt_acq: torch.Tensor,
    kt_trn: torch.Tensor,
    csm: torch.Tensor,
    noise_cov: torch.Tensor,
    lambda_0: float = 0.0014,
    real_prior: bool = True,
    fft_norm: str = None,
) -> torch.Tensor:
    r"""kt-SENSE dynamic MRI reconstruction. The function performs dynamic MRI reconstruction from undersampled data
    using densely sampled prior, coil sensitivities and coil noise covariance matrix.

    The implementation allows to process data slice by slice using GPU resources.

    Args:
        kt_acq: Undersampled data in kt-space :math:`(C, T, H, W)`.
        kt_trn: Fully-sampled prior data in kt-space :math:`(C, T, H, W)`.
        csm: Coil sensitivity maps in image domain :math:`(C, 1, H, W)`.
        noise_cov: Coil noise covariance matrix :math:`(C, C)`.
        lambda_0: Regularisation coefficient.
        real_prior: flag to use real-valued or complex-valued prior.
        fft_norm: FFT normalisation.

    Returns:
        Reconstructed dynamic MRI :math:`(C, T, H, W)`.

    References:
        Tsao, Jeffrey, Peter Boesiger, and Klaas P. Pruessmann.
        "k‐t BLAST and k‐t SENSE: dynamic MRI with high frame rate exploiting spatiotemporal correlations."
        Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in
        Medicine 50.5 (2003): 1031-1042.

    Notes:
        The performance of the kt-SENSE reconstruction matches the results obtained by
        https://github.com/mriphysics/ktrecon.
    """
    n_coils, n_freq, n_H, n_W = kt_acq.size()

    kt_sampling = kt_acq.sum(dim=(0, -1), keepdims=True) != 0
    kt_bln = kt_acq.sum(dim=-3, keepdim=True) / kt_sampling.sum(dim=-3, keepdim=True)
    kt_diff = kt_acq - kt_bln * kt_sampling
    xf_diff = kt2xf(kt_diff, norm=fft_norm)

    xf_trn = kt2xf(kt_trn, norm=fft_norm)
    xf_prior = sense(
        data=xf_trn[None], csm=csm[None], noise_cov=noise_cov, acceleration_rate=1
    )[0]
    xf_prior[:, xf_prior.size(-3) // 2, :, :] = 0

    xf_sampling = kt2xf(kt_sampling, norm=fft_norm)
    xf_psf = xf_sampling.abs() > 0.7 * xf_sampling.abs().max()
    n_aliases = xf_psf.sum()

    mask_coil, mask_freq, mask_H, mask_W = torch.where(xf_psf == 1)

    shift_H = (mask_H - mask_H[0]) % n_H
    shift_freq = (mask_freq - mask_freq[0]) % n_freq

    alias_sign = torch.sign(xf_sampling[:, mask_freq, mask_H, mask_W].real)

    safety_margin = 1 / lambda_0**0.5
    scale_factor = safety_margin

    xf_recon = torch.zeros_like(xf_prior)

    idx_H = torch.arange(int(n_H / n_aliases), device=csm.device)

    alias_H = (idx_H.view(1, -1, 1) - shift_H.view(1, 1, -1)) % n_H
    sensitivity = csm[:, 0, alias_H].permute(4, 2, 1, 0, 3)
    sensitivity_h = torch.conj(sensitivity.transpose(-1, -2))

    idx_freq = torch.arange(n_freq, device=csm.device)
    alias_freq = (idx_freq.view(-1, 1, 1) - shift_freq.view(1, 1, -1)) % n_freq
    rho_diff = n_aliases * xf_diff[:, :, idx_H].unsqueeze(0)

    rho_prior = xf_prior[:, alias_freq, alias_H].permute(4, 2, 1, 3, 0)

    M_2 = scale_factor**2 * torch.matmul(
        rho_prior, torch.conj(rho_prior.transpose(-1, -2))
    )

    if real_prior:
        M_2 = M_2 * torch.eye(M_2.size(-1), device=M_2.device)

    prior = torch.matmul(M_2, sensitivity_h)
    prior = torch.matmul(sensitivity, prior)
    prior = torch.matmul(
        M_2, torch.matmul(sensitivity_h, torch.inverse(prior + noise_cov))
    )

    prior = alias_sign.T * prior

    rho_recon = torch.matmul(prior, rho_diff.T)

    xf_recon[:, alias_freq, alias_H] = rho_recon.permute(4, 2, 1, 3, 0)

    xf_bln = k2x(kt_bln, norm=fft_norm) * n_freq

    xf_bln = sense(
        data=xf_bln[None], csm=csm[None], noise_cov=noise_cov, acceleration_rate=1
    )[0]

    xf_recon[0, n_freq // 2] = xf_bln[0, 0]
    xt_recon = f2t(xf_recon, norm=fft_norm)
    kt_recon = x2k(xt_recon, norm=fft_norm)

    return kt_recon


def kt_sense(
    kt_acq: torch.Tensor,
    kt_trn: torch.Tensor,
    csm: torch.Tensor,
    noise_cov: torch.Tensor,
    lambda_0: float = 0.0014,
    real_prior: bool = True,
    fft_norm: str = None,
) -> torch.Tensor:
    r"""kt-SENSE dynamic MRI reconstruction. The function performs dynamic MRI reconstruction from undersampled data
    using densely sampled prior, coil sensitivities and coil noise covariance matrix.

    The implementation allows to process data slice by slice using GPU resources.

    Args:
        kt_acq: Undersampled data in kt-space :math:`(N, C, T, H, W)`.
        kt_trn: Fully-sampled prior data in kt-space :math:`(N, C, T, H, W)`.
        csm: Coil sensitivity maps in image domain :math:`(N, C, 1, H, W)`.
        noise_cov: Coil noise covariance matrix :math:`(C, C)`.
        lambda_0: Regularisation coefficient.
        real_prior: flag to use real-valued or complex-valued prior.
        fft_norm: FFT normalisation.

    Returns:
        Reconstructed dynamic MRI :math:`(N, C, T, H, W)`.

    References:
        Tsao, Jeffrey, Peter Boesiger, and Klaas P. Pruessmann.
        "k‐t BLAST and k‐t SENSE: dynamic MRI with high frame rate exploiting spatiotemporal correlations."
        Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in
        Medicine 50.5 (2003): 1031-1042.

    Notes:
        The performance of the kt-SENSE reconstruction matches the results obtained by
        https://github.com/mriphysics/ktrecon.
    """
    n_b, n_coils, n_freq, n_H, n_W = kt_acq.size()

    kt_sampling_slice = kt_acq.sum(dim=(1, -1), keepdims=True) != 0
    kt_sampling = kt_sampling_slice.sum(dim=0) / n_b
    assert torch.all(
        kt_sampling_slice[0] == kt_sampling
    ), "In case of multiple slices, slices should have the same undersampling pattern"
    kt_bln = kt_acq.sum(dim=-3, keepdim=True) / kt_sampling.sum(dim=-3, keepdim=True)
    kt_diff = kt_acq - kt_bln * kt_sampling
    xf_diff = kt2xf(kt_diff, norm=fft_norm)

    xf_trn = kt2xf(kt_trn, norm=fft_norm)
    xf_prior = sense(data=xf_trn, csm=csm, noise_cov=noise_cov, acceleration_rate=1)
    xf_prior[:, :, xf_prior.size(-3) // 2, :, :] = 0

    xf_sampling = kt2xf(kt_sampling, norm=fft_norm)
    xf_psf = xf_sampling.abs() > 0.7 * xf_sampling.abs().max()
    n_aliases = xf_psf.sum()

    mask_coil, mask_freq, mask_H, mask_W = torch.where(xf_psf == 1)

    shift_H = (mask_H - mask_H[0]) % n_H
    shift_freq = (mask_freq - mask_freq[0]) % n_freq
    alias_sign = torch.sign(xf_sampling[:, mask_freq, mask_H, mask_W].real)

    safety_margin = 1 / lambda_0**0.5
    scale_factor = safety_margin

    xf_recon = torch.zeros_like(xf_prior)

    idx_H = torch.arange(int(n_H / n_aliases), device=csm.device)
    alias_H = (idx_H.view(1, -1, 1) - shift_H.view(1, 1, -1)) % n_H

    sensitivity = csm[:, :, 0, alias_H].permute(0, 5, 3, 2, 1, 4)

    sensitivity_h = torch.conj(sensitivity.transpose(-1, -2))

    idx_freq = torch.arange(n_freq, device=csm.device)
    alias_freq = (idx_freq.view(-1, 1, 1) - shift_freq.view(1, 1, -1)) % n_freq

    rho_prior = xf_prior[:, :, alias_freq, alias_H].permute(0, 5, 3, 2, 4, 1)

    M_2 = scale_factor**2 * torch.matmul(
        rho_prior, torch.conj(rho_prior.transpose(-1, -2))
    )

    if real_prior:
        M_2 = M_2 * torch.eye(M_2.size(-1), device=M_2.device)

    prior = torch.matmul(M_2, sensitivity_h)
    prior = torch.matmul(sensitivity, prior)
    prior = torch.matmul(
        M_2, torch.matmul(sensitivity_h, torch.inverse(prior + noise_cov))
    )

    prior = alias_sign.T * prior

    rho_diff = n_aliases * xf_diff[:, :, :, idx_H]
    rho_diff = rho_diff.unsqueeze(0)
    rho_recon = torch.matmul(prior, rho_diff.permute(1, 5, 4, 3, 2, 0))

    xf_recon[:, :, alias_freq, alias_H] = rho_recon.permute(0, 5, 3, 2, 4, 1)

    xf_bln = k2x(kt_bln, norm=fft_norm) * n_freq

    xf_bln = sense(data=xf_bln, csm=csm, noise_cov=noise_cov, acceleration_rate=1)

    xf_recon[:, 0, n_freq // 2] = xf_bln[:, 0, 0]
    xt_recon = f2t(xf_recon, norm=fft_norm)
    kt_recon = x2k(xt_recon, norm=fft_norm)

    return kt_recon
