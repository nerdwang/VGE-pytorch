import torch
import numpy as np

radians_to_degrees = 180.0 / np.pi

def pytorch_angular_error_from_pitchyaw(y_true, y_pred):
    """PyTorch method to calculate angular loss from head angles."""
    def angles_to_unit_vectors(y):
        sin = torch.sin(y)
        cos = torch.cos(y)
        return torch.stack([
            torch.mul(cos[:, 0], sin[:, 1]),
            sin[:, 0],
            torch.mul(cos[:, 0], cos[:, 1]),
        ], dim=1)

    v_true = angles_to_unit_vectors(y_true)
    v_pred = angles_to_unit_vectors(y_pred)
    return pytorch_angular_error_from_vector(v_true, v_pred)


def pytorch_angular_error_from_vector(v_true, v_pred):
    """PyTorch method to calculate angular loss from 3D vector."""
    v_true_norm = torch.sqrt(torch.sum(torch.square(v_true), dim=1))
    v_pred_norm = torch.sqrt(torch.sum(torch.square(v_pred), dim=1))

    sim = torch.div(torch.sum(torch.mul(v_true, v_pred), dim=1),
                    torch.mul(v_true_norm, v_pred_norm))

    # Floating point precision can cause sim values to be slightly outside of
    # [-1, 1] so we clip values
    sim = torch.clamp(sim, -1.0 + 1e-6, 1.0 - 1e-6)

    ang = torch.mul(radians_to_degrees, torch.acos(sim))
    return torch.mean(ang)