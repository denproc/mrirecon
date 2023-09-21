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


def k2x(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Transfers tensor from k-space into image space.
    Args:
        x: input tensor with at least 2 axes. Inverse fft is applied to last two dimensions.
        norm: normalisation, see torch.fft

    Returns:
        Tensor of the same dimensionality as input.
    """

    out = torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-1, -2)), dim=(-1, -2), norm=norm)
    return out


def x2k(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Transfers tensor from image space into k-space .
    Args:
        x: input tensor with at least 2 axes. Fourier is applied to last two dimensions.
        norm: normalisation, see torch.fft

    Returns:
        Tensor of the same dimensionality as input.
    """

    out = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1), norm=norm), dim=(-2, -1))
    return out


def t2f(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Transfers tensor from time domain into temporal frequency.
    Args:
        x: input tensor with at least 3 axes. Fourier is applied to third dimension from the end.
        norm: normalisation, see torch.fft

    Returns:
        Tensor of the same dimensionality as input.
    """

    out = torch.fft.fftshift(torch.fft.fftn(x, dim=(-3,), norm=norm), dim=(-3,))
    return out


def f2t(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Transfers tensor from temporal frequency domain into time .
    Args:
        x: input tensor with at least 3 axes. Inverse Fourier is applied to third dimension from the end.
        norm: normalisation, see torch.fft

    Returns:
        Tensor of the same dimensionality as input.
    """

    out = torch.fft.ifftn(torch.fft.ifftshift(x, dim=(-3,)), dim=(-3,), norm=norm)
    return out


def kt2xf(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Transfers tensor from kt-space into xf-space, i.e. image temporal frequency domain.

    Args:
        x: input tensor with at least 3 axes. Fourier is applied to third dimension from the end.
            Inverse Fourier is applied to last two dimensions.
        norm: normalisation, see torch.fft

    Returns:
        Tensor of the same dimensionality as input.
    """
    return t2f(k2x(x, norm=norm), norm=norm)


def xf2kt(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Transfers tensor from xf-space into kt-space.

    Args:
        x: input tensor with at least 3 axes. Inverse Fourier is applied to third dimension from the end.
            Fourier is applied to last two dimensions.
        norm: normalisation, see torch.fft

    Returns:
        Tensor of the same dimensionality as input.
    """
    return x2k(f2t(x, norm=norm), norm=norm)
