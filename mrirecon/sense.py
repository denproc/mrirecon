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


def sense(
    data: torch.Tensor,
    csm: torch.Tensor,
    noise_cov: torch.Tensor,
    acceleration_rate: int = 1,
) -> torch.Tensor:
    r"""SENSE reconstruction for parallel imaging in MRI. Combines complex-valued image signal acquired from set of
    coils.
    :math:`(S^H \Psi^{-1} S)^{-1} S^H \Psi^{-1} D`, where :math:`S` is a coil sensitivity maps, :math:`\Psi` - noise
    covariance, :math:`D` - signal acquired from coils.
    :math:`N` - number of slices, :math:`C` - number of coils, :math:`T` - number of time frames,
    :math:`F` - number of temporal frequencies, :math:`H` - image resolution in phase-encoding direction,
    :math:`W` - image resolution in readout direction.

    Args:
        data: input image data from set of coils. Size :math:`(N, C, T, H, W)` or :math:`(N, C, F, H, W)`.
        csm: coil sensitivity data. Size :math:`(N, C, 1, H, W)`.
        noise_cov: noise covariance matrix between coils. Size :math:`(C, C)`.
        acceleration_rate: acceleration factor in phase-encoding direction

    Returns:
        Combined image from all coils. Size :math:`(N, 1, T, H, W)` or :math:`(N, 1, F, H, W)`.

    References:
        Pruessmann, Klaas P., et al.
        "SENSE: sensitivity encoding for fast MRI."
        Magnetic Resonance in Medicine:
        An Official Journal of the International Society for Magnetic Resonance in Medicine 42.5 (1999): 952-962.
    """
    assert data.dim() == 5, f"Expected 5D tensor, got {data.dim()}D tensor"
    sense_matrix = _get_sense_matrix(
        csm=csm, noise_cov=noise_cov, acceleration_rate=acceleration_rate
    )
    sense_data = data * sense_matrix
    return sense_data.sum(dim=-4, keepdim=True)


def _get_sense_matrix(
    csm: torch.Tensor, noise_cov: torch.Tensor, acceleration_rate: int = 1
) -> torch.Tensor:
    r"""Create the operator matrix, which help to combine the signal from coils into signal SENSE reconstruction.

    Args:
        csm: coil sensitivity data. Size :math:`(N, C, 1, H, W)`.
        noise_cov: noise covariance matrix between coils. Size :math:`(C, C)`.
        acceleration_rate: acceleration factor in phase encoding direction
    Returns:
        Matrix to use for SENSE reconstruction. Size :math:`(N, C, 1, H, W)`

    References:
        Pruessmann, Klaas P., et al.
        "SENSE: sensitivity encoding for fast MRI."
        Magnetic Resonance in Medicine:
        An Official Journal of the International Society for Magnetic Resonance in Medicine 42.5 (1999): 952-962.
    """
    assert (
        acceleration_rate > 0
    ), f"Expected acceleration rate greater 0, got {acceleration_rate}"

    sense_matrix = torch.zeros_like(csm)
    psi_inv = torch.inverse(noise_cov)
    pe_size = csm.size(-2)

    aliases = torch.arange(
        start=0, end=pe_size, step=pe_size // acceleration_rate, device=csm.device
    ).view(-1, 1)
    mask_for_reduction = (
        torch.arange(pe_size // acceleration_rate, device=csm.device) + aliases
    )
    mask_csm = torch.where(csm[:, :, :, mask_for_reduction].sum(dim=(1, -3)) != 0)

    slice_locations = mask_csm[0].repeat(acceleration_rate, 1)
    frame_locations = mask_csm[1].repeat(acceleration_rate, 1)
    pe_locations = mask_csm[2] + aliases
    ro_locations = mask_csm[3].repeat(acceleration_rate, 1)

    s = csm[slice_locations, :, frame_locations, pe_locations, ro_locations].permute(
        1, 2, 0
    )
    s_h = torch.conj(s).transpose(-1, -2)
    unmix = torch.matmul(
        torch.pinverse(torch.matmul(s_h, torch.matmul(psi_inv, s))),
        torch.matmul(s_h, psi_inv),
    )

    sense_matrix[
        slice_locations, :, frame_locations, pe_locations, ro_locations
    ] = unmix.transpose(0, 1)

    return sense_matrix
