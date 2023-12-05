import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
from abc import abstractmethod
from typing import Optional, Tuple
from torch import Tensor, nn

class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Output dimension to module.
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor):
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError

class DeformMLP(FieldComponent):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, in_tensor):
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input

        Returns:
            MLP network output
        """
        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

class SE3DeformationFieldConfig:
    n_freq_pos = 7
    warp_code_dim: int = 8
    mlp_num_layers: int = 6
    mlp_layer_width: int = 128
    skip_connections: Tuple[int] = (4,)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    return torch.concat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    return v[..., :3] / v[..., -1:]

def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square

def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V

def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```

    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform.permute(0, 2, 1)

class MultiScaleSinousidalEncoding(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 num_frequencies: int, 
                 min_freq_exp: float, 
                 max_freq_exp: float,
                 include_input: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def forward(self,
                x,
                covs: Optional[torch.Tensor] = None,
                windows_param: Optional[float] = None) -> torch.Tensor:
        x = 2 * torch.pi * x # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(x.device)
        scaled_x = x[..., None] * freqs
        scaled_x = scaled_x.view(*scaled_x.shape[:-2], -1)

        if covs is None:
            encoded_x = torch.sin(torch.cat([scaled_x, scaled_x + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            x_means = torch.cat([scaled_x, scaled_x + torch.pi / 2.0], dim=-1)
            x_vars = torch.cat(2 * [input_var], dim=-1)
            encoded_x = torch.exp(-0.5 * x_vars) * torch.sin(x_means)

        if windows_param is not None:
            window = self.encode_window(windows_param).to(x.device)[None, :].repeat(x.shape[-1], 1).reshape(-1).repeat(2)
            encoded_x = window * encoded_x
        
        if self.include_input:
            encoded_x = torch.cat([encoded_x, x], dim=-1)
        
        return encoded_x
    
    def encode_window(self, windows_param):
        bands = torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)
        x = torch.clamp(windows_param - bands, 0.0, 1.0)
        return 0.5 * (1 - torch.cos(torch.pi * x))
    
    def get_out_dim(self):
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim
        
# Warping field
class SE3WarpingField(nn.Module):

    def __init__(
            self,
            config: SE3DeformationFieldConfig,
    ) -> None:
        super().__init__()

        self.position_encoding = MultiScaleSinousidalEncoding(
            in_dim=3,
            num_frequencies=config.n_freq_pos,
            min_freq_exp=0.0,
            max_freq_exp=config.n_freq_pos - 1,
            include_input=False
        )

        # in_dim = self.position_encoding.get_out_dim() + config.warp_code_dim
        in_dim = self.position_encoding.get_out_dim()

        self.mlp_stem = DeformMLP(
            in_dim=in_dim,
            out_dim=config.mlp_layer_width,
            num_layers=config.mlp_num_layers,
            layer_width=config.mlp_layer_width,
            skip_connections=config.skip_connections,
            out_activation=nn.ReLU(),
        )
        self.mlp_r = DeformMLP(
            in_dim=config.mlp_layer_width,
            out_dim=3,
            num_layers=1,
            layer_width=config.mlp_layer_width,
        )
        self.mlp_v = DeformMLP(
            in_dim=config.mlp_layer_width,
            out_dim=3,
            num_layers=1,
            layer_width=config.mlp_layer_width,
        )

        # diminish the last layer of SE3 Field to approximate an identity transformation
        nn.init.uniform_(self.mlp_r.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.uniform_(self.mlp_v.layers[-1].weight, a=-1e-5, b=1e-5)
        nn.init.zeros_(self.mlp_r.layers[-1].bias)
        nn.init.zeros_(self.mlp_v.layers[-1].bias)

    def get_transform(self, positions: torch.Tensor,
                      warp_code: torch.Tensor = None,
                      windows_param=None):
        encoded_xyz = self.position_encoding(
            positions,
            windows_param=windows_param,
        )  # (R, S, 3)

        if warp_code is not None:
            feat = self.mlp_stem(torch.cat([encoded_xyz, warp_code], dim=-1))  # (R, S, D)
        else:
            feat = self.mlp_stem(encoded_xyz)

        r = self.mlp_r(feat).reshape(-1, 3)  # (R*S, 3)
        v = self.mlp_v(feat).reshape(-1, 3)  # (R*S, 3)

        screw_axis = torch.concat([v, r], dim=-1)  # (R*S, 6)
        screw_axis = screw_axis.to(positions.dtype)
        transforms = se3_exp_map(screw_axis)
        return transforms.permute(0, 2, 1)

    def apply_transform(self, positions: torch.Tensor, transforms: torch.Tensor):
        p = positions.reshape(-1, 3)

        warped_p = from_homogenous((transforms @ to_homogenous(p).unsqueeze(-1)).squeeze(-1))
        warped_p = warped_p.to(positions.dtype)

        idx_nan = warped_p.isnan()
        warped_p[idx_nan] = p[idx_nan]  # if deformation is NaN, just use original point

        # Reshape to shape of input positions tensor
        warped_p = warped_p.reshape(*positions.shape[: positions.ndim - 1], 3)

        return warped_p

    def forward(self, positions: torch.Tensor,
                warp_code: Optional[torch.Tensor] = None,
                windows_param: Optional[float] = None):
        # if warp_code is None:
        #     return None

        transforms = self.get_transform(positions, warp_code, windows_param)
        return self.apply_transform(positions, transforms)

class SE3DeformationField(nn.Module):

    def __init__(self,
                 deformation_field_config: SE3DeformationFieldConfig,
                 max_n_samples_per_batch: int = -1,
                 ):
        super(SE3DeformationField, self).__init__()

        self.se3_field = SE3WarpingField(deformation_field_config)
        self.max_n_samples_per_batch = max_n_samples_per_batch

    def forward(self,
                ray_samples,
                warp_code = None,
                windows_param: Optional[float] = None):
        
        warped_points = self.se3_field(
            positions=ray_samples, 
            warp_code=warp_code, 
            windows_param=windows_param)
        
        return warped_points
