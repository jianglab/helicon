import math

import torch
import numpy as np
from dataclasses import dataclass, field


# These two conversions were the only reason this module depended on kornia,
# which is a heavy dependency (kornia + kornia-rs) for ~20 lines of standard
# quaternion algebra.  Both match kornia's WXYZ convention exactly; verified
# against kornia over random inputs and the degenerate cases.
def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Unit-normalised (w, x, y, z) quaternions -> rotation matrices.

    (..., 4) -> (..., 3, 3).  Matches the former
    ``kornia.geometry.conversions.quaternion_to_rotation_matrix``.
    """
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        dim=-2,
    )


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rotation vectors (axis * angle) -> (w, x, y, z) quaternions.

    (..., 3) -> (..., 4).  Matches the former
    ``kornia.geometry.conversions.axis_angle_to_quaternion``, including the
    small-angle limit where sin(theta/2)/theta -> 1/2.
    """
    theta_sq = (axis_angle * axis_angle).sum(-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp_min(0))
    half = 0.5 * theta
    # away from zero use sin(theta/2)/theta; at zero use the 1/2 limit
    k = torch.where(
        theta_sq > 0,
        torch.sin(half) / theta.clamp_min(1e-20),
        0.5 * torch.ones_like(theta),
    )
    w = torch.where(theta_sq > 0, torch.cos(half), torch.ones_like(theta))
    return torch.cat([w, axis_angle * k], dim=-1)


@dataclass
class AnisotropicGaussian:
    amplitude: float = 1.0
    center: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0], metadata={"length": 3}
    )
    sigma: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0], metadata={"length": 3}
    )
    quaternion: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0], metadata={"length": 4}
    )  # (w, x, y, z)

    def __post_init__(self):
        if isinstance(self.center, (int, float)):
            self.center = [self.center] * 3
        elif len(self.center) != 3:
            raise ValueError("center must be a single value or a list of 3 values.")
        if isinstance(self.sigma, (int, float)):
            self.sigma = [self.sigma] * 3
        elif len(self.sigma) != 3:
            raise ValueError("sigma must be a single value or a list of 3 values.")


class AnisotropicGaussianSet:
    def __init__(
        self,
        amplitudes: torch.Tensor,
        centers: torch.Tensor,
        sigmas: torch.Tensor,
        quaternions: torch.Tensor,
        device: str = "cpu",
    ):
        assert (
            amplitudes.shape[0]
            == centers.shape[0]
            == sigmas.shape[0]
            == quaternions.shape[0]
        )
        assert amplitudes.dim() == 1
        assert centers.dim() == 2 and centers.shape[1] == 3
        assert sigmas.dim() == 2 and sigmas.shape[1] == 3
        assert quaternions.dim() == 2 and quaternions.shape[1] == 4
        self.amplitudes = amplitudes.to(torch.float32)
        self.centers = centers.to(torch.float32)
        self.sigmas = sigmas.to(torch.float32)
        self.quaternions = quaternions.to(torch.float32)
        try:
            assert (
                amplitudes.device
                == centers.device
                == sigmas.device
                == quaternions.device
            )
            self.device = amplitudes.device
        except:
            self.device = None
        self.to(device)

    def __len__(self):
        return len(self.amplitudes)

    @classmethod
    def from_list(cls, gaussians: list[AnisotropicGaussian] = [], device: str = "cpu"):
        amplitudes = torch.tensor(
            [g.amplitude for g in gaussians], dtype=torch.float32, device=device
        )
        centers = torch.tensor(
            np.array([g.center for g in gaussians]), dtype=torch.float32, device=device
        )
        sigmas = torch.tensor(
            np.array([g.sigma for g in gaussians]), dtype=torch.float32, device=device
        )
        quaternions = torch.tensor(
            np.array([g.quaternion for g in gaussians]),
            dtype=torch.float32,
            device=device,
        )
        assert (amplitudes > 0).all()
        assert (sigmas > 0).all()
        return cls(amplitudes, centers, sigmas, quaternions, device)

    def mask(
        self,
        nx: int,
        ny: int,
        nz: int,
        apix: float,
        min_dist_sigma: float = 5.0,
        inplace: bool = False,
    ):
        """Remove Gaussians too close to edge"""
        rotation_matrices = quaternion_to_rotation_matrix(self.quaternions)
        cov_3d = torch.bmm(
            rotation_matrices,
            torch.bmm(
                torch.diag_embed(self.sigmas**2), rotation_matrices.transpose(1, 2)
            ),
        )
        sigma_x = torch.sqrt(cov_3d[:, 0, 0])
        sigma_y = torch.sqrt(cov_3d[:, 1, 1])
        sigma_z = torch.sqrt(cov_3d[:, 2, 2])

        distances = torch.empty((self.centers.shape[0], 6), device=self.device)
        distances[:, 0] = (
            self.centers[:, 0] > -nx // 2 * apix + min_dist_sigma * sigma_x
        )  # Left edge (x = -nx//2)
        distances[:, 1] = (
            self.centers[:, 0] < nx // 2 * apix - min_dist_sigma * sigma_x
        )  # Right edge (x = nx//2)
        distances[:, 2] = (
            self.centers[:, 1] > -ny // 2 * apix + min_dist_sigma * sigma_y
        )  # Bottom edge (y = -ny//2)
        distances[:, 3] = (
            self.centers[:, 1] < ny // 2 * apix - min_dist_sigma * sigma_y
        )  # Top edge (y = ny//2)
        distances[:, 4] = (
            self.centers[:, 2] > -nz // 2 * apix + min_dist_sigma * sigma_z
        )  # Back edge (z = -nz//2)
        distances[:, 5] = (
            self.centers[:, 2] < nz // 2 * apix - min_dist_sigma * sigma_z
        )  # Front edge (z = nz//2)
        mask = (distances > 0).all(dim=1)
        num_true_elements = mask.sum().item()
        if num_true_elements < 1:
            raise ValueError(
                "All gaussians have been masked out in AnisotropicGaussianSet.mask()!"
            )

        if inplace:
            self.amplitudes = self.amplitudes[mask]
            self.centers = self.centers[mask]
            self.sigmas = self.sigmas[mask]
            self.quaternions = self.quaternions[mask]
            return self
        else:
            amplitudes = self.amplitudes[mask]
            centers = self.centers[mask]
            sigmas = self.sigmas[mask]
            quaternions = self.quaternions[mask]
            return AnisotropicGaussianSet(
                amplitudes, centers, sigmas, quaternions, self.device
            )

    def to(self, device: str):
        if self.device != device:
            self.amplitudes = self.amplitudes.clone().to(device)
            self.centers = self.centers.clone().to(device)
            self.sigmas = self.sigmas.clone().to(device)
            self.quaternions = self.quaternions.clone().to(device)
            self.device = device
        return self

    def apply_helical_symmetry(
        self,
        twist: float,
        rise: float,
        csym: int,
        zmin: float,
        zmax: float,
        min_dist_sigma: float = 3.0,
    ):
        assert rise > 0, "Rise must be positive"
        assert (
            isinstance(csym, int) and csym > 0
        ), "Cyclic symmetry must be a positive integer"
        assert zmin < zmax, "zmin must be less than zmax"
        z0, z1 = self.centers[:, 2].min(), self.centers[:, 2].max()

        rotation_matrices = quaternion_to_rotation_matrix(self.quaternions)
        cov_3d = torch.bmm(
            rotation_matrices,
            torch.bmm(
                torch.diag_embed(self.sigmas**2), rotation_matrices.transpose(1, 2)
            ),
        )
        sigma_min = torch.sqrt(cov_3d[:, 2, 2].min())
        # math, not numpy: the operands here are torch tensors, and handing a
        # tensor to np.floor/np.ceil goes through __array_wrap__, which numpy 2
        # deprecates and will eventually refuse. Pulling the scalars out first
        # keeps the arithmetic in plain Python and says what is meant.
        n_min = math.floor(
            (zmin + min_dist_sigma * float(sigma_min) - float(z1)) / rise
        )
        n_max = math.ceil((zmax - min_dist_sigma * float(sigma_min) - float(z0)) / rise)
        n_range = torch.arange(
            n_min, n_max + 1, device=self.device, dtype=torch.float32
        )

        n_grid, i_grid = torch.meshgrid(
            n_range,
            torch.arange(csym, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        n_flat = n_grid.flatten()
        i_flat = i_grid.flatten()

        angles = (
            torch.deg2rad(torch.tensor(twist, device=self.device, dtype=torch.float32))
            * n_flat
            + 2 * np.pi * i_flat / csym
        )
        rotations = axis_angle_to_quaternion(
            torch.stack(
                [torch.zeros_like(angles), torch.zeros_like(angles), angles], dim=-1
            )
        )

        new_centers = self.centers.clone().repeat(len(n_flat), 1)
        new_centers[:, 2] = (
            self.centers[:, 2].clone() + (rise * n_flat.unsqueeze(1))
        ).reshape(-1)
        rotation_matrices = quaternion_to_rotation_matrix(
            rotations
        )  # Shape: (len(n_flat) * csym, 3, 3)
        new_centers[:, :2] = (
            (
                self.centers[:, :2].clone()
                @ (rotation_matrices[:, :2, :2].reshape(-1, 2, 2).transpose(1, 2))
            )
        ).reshape(-1, 2)

        new_quaternions = (rotations * self.quaternions.unsqueeze(1)).reshape(-1, 4)

        expanded_amplitudes = self.amplitudes.clone().repeat(len(n_flat), 1).reshape(-1)
        expanded_sigmas = self.sigmas.clone().repeat(len(n_flat), 1)

        sigmas_z = torch.sqrt(cov_3d[:, 2, 2]).repeat(len(n_flat))
        expanded_min_dist_sigma = torch.tensor(
            min_dist_sigma, dtype=torch.float32, device=self.device
        ).repeat(len(self.amplitudes) * len(n_flat))
        mask = zmin + expanded_min_dist_sigma * sigmas_z < new_centers[:, 2]
        mask &= new_centers[:, 2] < zmax - expanded_min_dist_sigma * sigmas_z

        return AnisotropicGaussianSet(
            expanded_amplitudes[mask],
            new_centers[mask],
            expanded_sigmas[mask],
            new_quaternions[mask],
        )

    def sample_volume(
        self, nx: int, ny: int, nz: int, apix: float, batch_size: int = 100
    ):
        z = (torch.arange(nz, dtype=torch.float32, device=self.device) - nz // 2) * apix
        y = (torch.arange(ny, dtype=torch.float32, device=self.device) - ny // 2) * apix
        x = (torch.arange(nx, dtype=torch.float32, device=self.device) - nx // 2) * apix
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        XYZ = torch.stack(
            [X.flatten(), Y.flatten(), Z.flatten()], dim=-1
        )  # Shape: (nz*ny*nx, 3)

        vol = torch.zeros(nz * ny * nx, dtype=torch.float32, device=self.device)

        for i in range(0, len(self.amplitudes), batch_size):
            batch_amplitudes = self.amplitudes[i : i + batch_size]
            batch_centers = self.centers[i : i + batch_size]
            batch_sigmas = self.sigmas[i : i + batch_size]
            batch_quaternions = self.quaternions[i : i + batch_size]

            rotation_matrices = quaternion_to_rotation_matrix(batch_quaternions)

            cov_3d = torch.bmm(
                rotation_matrices,
                torch.bmm(
                    torch.diag_embed(batch_sigmas**2), rotation_matrices.transpose(1, 2)
                ),
            )
            inv_covs = torch.inverse(cov_3d)

            diff = XYZ.unsqueeze(1) - batch_centers.unsqueeze(0)

            exponent = -0.5 * torch.einsum("ngi,gij,ngj->ng", diff, inv_covs, diff)

            vol += torch.sum(batch_amplitudes * torch.exp(exponent), dim=-1)

        vol = vol.reshape(nz, ny, nx)

        return vol

    def projection_z(self, nx: int, ny: int, apix: float, batch_size: int = 10_000):
        y = (torch.arange(ny, dtype=torch.float32, device=self.device) - ny // 2) * apix
        x = (torch.arange(nx, dtype=torch.float32, device=self.device) - nx // 2) * apix
        Y, X = torch.meshgrid(y, x, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)

        proj = torch.zeros(ny * nx, dtype=torch.float32, device=self.device)

        for i in range(0, len(self.amplitudes), batch_size):
            batch_amplitudes = self.amplitudes[i : i + batch_size]
            batch_centers = self.centers[i : i + batch_size][:, :2]
            batch_sigmas = self.sigmas[i : i + batch_size]
            batch_quaternions = self.quaternions[i : i + batch_size]

            rotation_matrices = quaternion_to_rotation_matrix(batch_quaternions)

            cov_3d = torch.bmm(
                rotation_matrices,
                torch.bmm(
                    torch.diag_embed(batch_sigmas**2), rotation_matrices.transpose(1, 2)
                ),
            )
            cov_2d = cov_3d[:, :2, :2]
            inv_covs_2d = torch.inverse(cov_2d)

            diff = XY.unsqueeze(1) - batch_centers.unsqueeze(0)
            exponent = -0.5 * torch.einsum("ngi,gij,ngj->ng", diff, inv_covs_2d, diff)
            # Integrating an unnormalised 3D Gaussian along z gives an
            # unnormalised 2D Gaussian with the (x, y) marginal covariance and
            # prefactor sqrt(2*pi * det(Sigma)/det(Sigma_xy)) -- the Schur
            # complement 1/(Sigma^-1)_zz.  The simpler sqrt(2*pi * Sigma_zz) is
            # only correct when Sigma has no z cross-terms, i.e. for isotropic
            # or axis-aligned Gaussians; for a rotated anisotropic Gaussian it
            # can be wrong by tens of percent.
            schur = torch.det(cov_3d) / torch.det(cov_2d).clamp_min(1e-12)
            normalizations = (
                batch_amplitudes * torch.sqrt((2 * torch.pi) * schur) / apix
            )
            proj += torch.sum(normalizations * torch.exp(exponent), dim=-1)

        proj = proj.reshape(ny, nx)

        return proj


@dataclass
class IsotropicGaussian:
    amplitude: float = 1.0
    center: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0], metadata={"length": 3}
    )
    sigma: float = 1.0


class IsotropicGaussianSet:
    def __init__(
        self,
        amplitudes: torch.Tensor,
        centers: torch.Tensor,
        sigmas: float,
        device: str = "cpu",
    ):
        assert amplitudes.shape[0] == centers.shape[0] == sigmas.shape[0]
        assert amplitudes.dim() == 1
        assert centers.dim() == 2 and centers.shape[1] == 3
        assert sigmas.dim() == 1
        self.amplitudes = amplitudes.to(torch.float32)
        self.centers = centers.to(torch.float32)
        self.sigmas = sigmas.to(torch.float32)
        try:
            assert amplitudes.device == centers.device == sigmas.device
            self.device = amplitudes.device
        except:
            self.device = None
        self.to(device)

    def __len__(self):
        return len(self.amplitudes)

    @classmethod
    def from_list(cls, gaussians: list[IsotropicGaussian] = [], device: str = "cpu"):
        amplitudes = torch.tensor(
            [g.amplitude for g in gaussians], dtype=torch.float32, device=device
        )
        centers = torch.tensor(
            np.array([g.center for g in gaussians]), dtype=torch.float32, device=device
        )
        sigmas = torch.tensor(
            np.array([g.sigma for g in gaussians]), dtype=torch.float32, device=device
        )
        assert (amplitudes > 0).all()
        assert (sigmas > 0).all()
        return cls(amplitudes, centers, sigmas, device)

    def mask(
        self,
        nx: int,
        ny: int,
        nz: int,
        apix: float,
        min_dist_sigma: float = 3.0,
        inplace: bool = False,
    ):
        """Remove Gaussians too close to edge"""

        distances = torch.empty((self.centers.shape[0], 6), device=self.device)
        distances[:, 0] = (
            self.centers[:, 0] > -nx // 2 * apix + min_dist_sigma * self.sigmas
        )  # Left edge (x = -nx//2)
        distances[:, 1] = (
            self.centers[:, 0] < nx // 2 * apix - min_dist_sigma * self.sigmas
        )  # Right edge (x = nx//2)
        distances[:, 2] = (
            self.centers[:, 1] > -ny // 2 * apix + min_dist_sigma * self.sigmas
        )  # Bottom edge (y = -ny//2)
        distances[:, 3] = (
            self.centers[:, 1] < ny // 2 * apix - min_dist_sigma * self.sigmas
        )  # Top edge (y = ny//2)
        distances[:, 4] = (
            self.centers[:, 2] > -nz // 2 * apix + min_dist_sigma * self.sigmas
        )  # Back edge (z = -nz//2)
        distances[:, 5] = (
            self.centers[:, 2] < nz // 2 * apix - min_dist_sigma * self.sigmas
        )  # Front edge (z = nz//2)
        mask = (distances > 0).all(dim=1)

        if inplace:
            self.amplitudes = self.amplitudes[mask]
            self.centers = self.centers[mask]
            self.sigmas = self.sigmas[mask]
            return self
        else:
            amplitudes = self.amplitudes[mask]
            centers = self.centers[mask]
            sigmas = self.sigmas[mask]
            return IsotropicGaussianSet(amplitudes, centers, sigmas, self.device)

    def to(self, device: str):
        if self.device != device:
            self.amplitudes = self.amplitudes.clone().to(device)
            self.centers = self.centers.clone().to(device)
            self.sigmas = self.sigmas.clone().to(device)
            self.device = device
        return self

    def apply_helical_symmetry(
        self,
        twist: float,
        rise: float,
        csym: int,
        zmin: float,
        zmax: float,
        min_dist_sigma: float = 3.0,
    ):
        assert rise > 0, "Rise must be positive"
        assert (
            isinstance(csym, int) and csym > 0
        ), "Cyclic symmetry must be a positive integer"
        assert zmin < zmax, "zmin must be less than zmax"

        z0, z1 = self.centers[:, 2].min(), self.centers[:, 2].max()
        sigma_min = self.sigmas.min()
        # math, not numpy: the operands here are torch tensors, and handing a
        # tensor to np.floor/np.ceil goes through __array_wrap__, which numpy 2
        # deprecates and will eventually refuse. Pulling the scalars out first
        # keeps the arithmetic in plain Python and says what is meant.
        n_min = math.floor(
            (zmin + min_dist_sigma * float(sigma_min) - float(z1)) / rise
        )
        n_max = math.ceil((zmax - min_dist_sigma * float(sigma_min) - float(z0)) / rise)
        n_range = torch.arange(
            n_min, n_max + 1, device=self.device, dtype=torch.float32
        )

        n_grid, i_grid = torch.meshgrid(
            n_range,
            torch.arange(csym, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        n_flat = n_grid.flatten()
        i_flat = i_grid.flatten()

        angles = (
            torch.deg2rad(torch.tensor(twist, device=self.device, dtype=torch.float32))
            * n_flat
            + 2 * np.pi * i_flat / csym
        )
        rotations = axis_angle_to_quaternion(
            torch.stack(
                [torch.zeros_like(angles), torch.zeros_like(angles), angles], dim=-1
            )
        )

        new_centers = self.centers.clone().repeat(len(n_flat), 1)
        new_centers[:, 2] = (
            self.centers[:, 2].clone() + (rise * n_flat.unsqueeze(1))
        ).reshape(-1)
        rotation_matrices = quaternion_to_rotation_matrix(rotations)
        new_centers[:, :2] = (
            (
                self.centers[:, :2].clone()
                @ (rotation_matrices[:, :2, :2].reshape(-1, 2, 2).transpose(1, 2))
            )
        ).reshape(-1, 2)

        expanded_amplitudes = self.amplitudes.clone().repeat(len(n_flat), 1).reshape(-1)
        expanded_sigmas = self.sigmas.clone().repeat(len(n_flat), 1).reshape(-1)

        mask = zmin + min_dist_sigma * expanded_sigmas < new_centers[:, 2]
        mask &= new_centers[:, 2] < zmax - min_dist_sigma * expanded_sigmas

        return IsotropicGaussianSet(
            expanded_amplitudes[mask], new_centers[mask], expanded_sigmas[mask]
        )

    def sample_volume(
        self,
        nx: int,
        ny: int,
        nz: int,
        apix: float,
        batch_size: int = 100,
        device="cpu",
    ):
        z = (torch.arange(nz, dtype=torch.float32, device=device) - nz // 2) * apix
        y = (torch.arange(ny, dtype=torch.float32, device=device) - ny // 2) * apix
        x = (torch.arange(nx, dtype=torch.float32, device=device) - nx // 2) * apix
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        XYZ = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

        vol = torch.zeros((nz * ny * nx), dtype=torch.float32, device=device)

        for i in range(0, len(self.amplitudes), batch_size):
            batch_amplitudes = self.amplitudes[i : i + batch_size]
            batch_centers = self.centers[i : i + batch_size]
            batch_sigmas = self.sigmas[i : i + batch_size]

            diff = XYZ.unsqueeze(1) - batch_centers.unsqueeze(0)
            exponent = -0.5 * (diff / batch_sigmas.unsqueeze(0).unsqueeze(2)) ** 2.0
            vol += torch.sum(
                batch_amplitudes.unsqueeze(0) * torch.exp(torch.sum(exponent, dim=-1)),
                dim=-1,
            )

        return vol.reshape((nz, ny, nx))

    def projection_z(self, nx: int, ny: int, apix: float, batch_size: int = 10_000):
        y = (torch.arange(ny, dtype=torch.float32, device=self.device) - ny // 2) * apix
        x = (torch.arange(nx, dtype=torch.float32, device=self.device) - nx // 2) * apix
        Y, X = torch.meshgrid(y, x, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)

        proj = torch.zeros(ny * nx, dtype=torch.float32, device=self.device)
        sqrt_2pi_apix = torch.sqrt(torch.tensor(2 * np.pi, device=self.device)) / apix

        for i in range(0, len(self.amplitudes), batch_size):
            batch_amplitudes = self.amplitudes[i : i + batch_size]
            batch_centers = self.centers[i : i + batch_size][:, :2]
            batch_sigmas = self.sigmas[i : i + batch_size]

            diff = XY.unsqueeze(1) - batch_centers.unsqueeze(0)
            exponent = -0.5 * (diff / batch_sigmas.unsqueeze(0).unsqueeze(2)) ** 2.0
            proj += torch.sum(
                batch_amplitudes.unsqueeze(0)
                * sqrt_2pi_apix
                * batch_sigmas.unsqueeze(0)
                * torch.exp(torch.sum(exponent, dim=-1)),
                dim=-1,
            )

        return proj.reshape(ny, nx)
