"""
File: src/models/rendering_tcnn.py

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.

LONER © 2023 by FCAV @ University of Michigan is licensed under CC BY-NC-SA 4.0
See the LICENSE file for details.

Authors: Seth Isaacson and Pou-Chun (Frank) Kung
"""

import torch
import torch.nn.functional as F


# ref: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    # (N_rays, N_samples_)
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    # Take uniform samples
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        # generates random numbers in the interval [0, 1)
        u = torch.rand(N_rays, N_importance, device=bins.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1]-cdf_g[..., 0]
    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    denom[denom < eps] = 1
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u-cdf_g[..., 0]) / \
        denom * (bins_g[..., 1]-bins_g[..., 0])
    return samples

def raw2outputs_adjusted(raw, z_vals, rays_o, rays_d, raw_noise_std=0, white_bkgd=False, sigma_only=False, num_colors=3, softplus=False, far=None, ret_var=True):
    """Transforms model's predictions to semantically meaningful values. Using adjusted method for unstructured environments.
    Args:
        raw: [num_rays, num_samples along ray, 4(sigma_only=False) or 1(sigma_only=True)]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_o: [num_rays, 3]. Direction of each ray.
        rays_d: [num_rays, 3]. Direction of each ray.
        sigma_only: Only sigma was predicted by network
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    if sigma_only:
        sigmas = raw[..., 0]  # (N_rays, N_samples_)
    else:
        rgbs = raw[..., :num_colors]  # (N_rays, N_samples_, 3)
        sigmas = raw[..., num_colors]  # (N_rays, N_samples_)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    # (N_rays, 1) the last delta is infinity
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(rays_d.unsqueeze(1), dim=-1)

    noise = 0.
    if raw_noise_std > 0:
        noise = torch.randn(sigmas.shape, device=sigmas.device) * raw_noise_std
    noise = 0.

    # compute alpha by the formula (3)
    if softplus == True:
        # (N_rays, N_samples_)
        alphas = 1 - torch.exp(-deltas*torch.nn.functional.softplus(sigmas+noise))
    else:
        # (N_rays, N_samples_)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise))

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1. -
                  alphas+1e-10], -1)  # [1, a1, a2, ...]
    
    T = torch.cumprod(alphas_shifted, -1)[:, :-1]
                
    # (N_rays, N_samples_)
    weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]
    
    opacity_map = torch.sum(weights, -1)

    T_shifted = torch.cat((torch.ones(T.shape[0],1).to(T.device),T[:,:-1]),1)

    # Inintilize Ray Termination at T of 50 percent
    # Treat rays with less likelyhood of termination as invalid -> mapped to zero

    T_indicices_terminal = (torch.logical_not(T > 0.5)*(T_shifted > 0.5)).nonzero(as_tuple=True)
    depths = torch.zeros(z_vals.shape[0]).to(T.device)
    depths[T_indicices_terminal[0]] = z_vals[T_indicices_terminal]

    valid = T_indicices_terminal[0]

    PROMINENCE_THRESHOLD_MAX = 0.45
    PROMINENCE_THRESHOLD_MIN = 0.35

    # Calculate Discretized Values of Termination Probabilities

    T_indicices_0_1 = (torch.logical_not(T[valid] > 0.9) * (T_shifted[valid] > 0.9)).nonzero(as_tuple=True)
    T_indicices_0_2 = (torch.logical_not(T[valid] > 0.8) * (T_shifted[valid] > 0.8)).nonzero(as_tuple=True)
    T_indicices_0_3 = (torch.logical_not(T[valid] > 0.7) * (T_shifted[valid] > 0.7)).nonzero(as_tuple=True)
    T_indicices_0_4 = (torch.logical_not(T[valid] > 0.6) * (T_shifted[valid] > 0.6)).nonzero(as_tuple=True)
    T_indicices_0_5 = (torch.logical_not(T[valid] > 0.501) * (T_shifted[valid] > 0.501)).nonzero(as_tuple=True)

    # Set up Discretized Values of Rise Deltas

    predicted_depths_0_1 = torch.zeros(z_vals[valid].shape[0]).to(T.device)
    predicted_depths_0_2 = torch.zeros(z_vals[valid].shape[0]).to(T.device)
    predicted_depths_0_3 = torch.zeros(z_vals[valid].shape[0]).to(T.device)
    predicted_depths_0_4 = torch.zeros(z_vals[valid].shape[0]).to(T.device)
    predicted_depths_0_5 = torch.zeros(z_vals[valid].shape[0]).to(T.device)

    predicted_depths_0_1[T_indicices_0_1[0]] = z_vals[valid][T_indicices_0_1]
    predicted_depths_0_2[T_indicices_0_2[0]] = z_vals[valid][T_indicices_0_2]
    predicted_depths_0_3[T_indicices_0_3[0]] = z_vals[valid][T_indicices_0_3]
    predicted_depths_0_4[T_indicices_0_4[0]] = z_vals[valid][T_indicices_0_4]
    predicted_depths_0_5[T_indicices_0_5[0]] = z_vals[valid][T_indicices_0_5]

    rise_deltas_2_1 = predicted_depths_0_2 - predicted_depths_0_1
    rise_deltas_3_2 = predicted_depths_0_3 - predicted_depths_0_2
    rise_deltas_4_3 = predicted_depths_0_4 - predicted_depths_0_3
    rise_deltas_5_4 = predicted_depths_0_5 - predicted_depths_0_4

    # rise_deltas == distance between samples where a  jump of 10 percent in Termination prob. is observed
    rise_deltas = torch.stack([rise_deltas_2_1, rise_deltas_3_2, rise_deltas_4_3, rise_deltas_5_4])
    normalized_rise_deltas = rise_deltas / predicted_depths_0_5

    T_indices_0 = T_indicices_0_1[0]
    T_indices_1 = torch.stack([T_indicices_0_1[1], T_indicices_0_2[1], T_indicices_0_3[1], T_indicices_0_4[1], T_indicices_0_5[1]])

    # maximum rise rate == stepest rise in termination prob. between two consecutive samples
    max_values, max_indices = torch.topk(normalized_rise_deltas, 2, dim =0, largest=False)

    max_indices_first = max_indices[0].unsqueeze(0)

    # Indices of T based on max rise rate
    T_indices_1_max = T_indices_1[max_indices_first, torch.arange(T_indices_1.shape[1])].squeeze(0)

    # Measurement for discrete Isolation of first and secon highest peak
    max_indices_delta = torch.abs(max_indices[1] - max_indices[0])

    # resample indices for a low isolation peaks
    resample_indices_isolation = max_indices_delta == 1

    # Evaluate the indices to drop based on the isolation of the peaks
    drop_indices_isolation = max_indices_delta == 3

    # resample indices for a prominent peak
    resample_indices_prominence = max_values[0] / torch.abs(max_values[1] + max_values[0]) > PROMINENCE_THRESHOLD_MAX

    # Evaluate the indices to drop based on the prominence of the peaks
    drop_indices_prominence= (max_values[0]/ torch.mean(normalized_rise_deltas,0)) < PROMINENCE_THRESHOLD_MIN

    depths[valid][T_indices_0[resample_indices_isolation]] = z_vals[valid][T_indices_0[resample_indices_isolation],T_indices_1_max[resample_indices_isolation]]
    depths[valid][T_indices_0[resample_indices_prominence]] = z_vals[valid][T_indices_0[resample_indices_prominence],T_indices_1_max[resample_indices_prominence]]
    depths[valid][T_indices_0[drop_indices_isolation]] = 0
    depths[valid][T_indices_0[drop_indices_prominence]] = 0

    if sigma_only:
        rgb_map = torch.tensor([-1.])
    else:
        # weights_normed = weights / (weights.sum(1, keepdim=True) + 1e-6)
        rgb_map = torch.sum(weights.unsqueeze(-1)*rgbs, -2)  # (N_rays, 3)

        if white_bkgd:
            rgb_map = rgb_map + 1-weights.sum(-1, keepdim=True)

    if ret_var:
        variance = (weights * (depths.view(-1, 1) - z_vals)**2).sum(dim=1)
        return rgb_map, depths, weights, opacity_map, variance

    return rgb_map, depths, weights, opacity_map, None


# ref: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf.py#L262

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, sigma_only=False, num_colors=3, softplus=False, far=None, ret_var=False):
    """Transforms model's predictions to semantically meaningful values. Using Default method.
    Args:
        raw: [num_rays, num_samples along ray, 4(sigma_only=False) or 1(sigma_only=True)]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        sigma_only: Only sigma was predicted by network
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """

    if sigma_only:
        sigmas = raw[..., 0]  # (N_rays, N_samples_)
    else:
        rgbs = raw[..., :num_colors]  # (N_rays, N_samples_, 3)
        sigmas = raw[..., num_colors]  # (N_rays, N_samples_)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    # (N_rays, 1) the last delta is infinity
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(rays_d.unsqueeze(1), dim=-1)

    noise = 0.
    if raw_noise_std > 0:
        noise = torch.randn(sigmas.shape, device=sigmas.device) * raw_noise_std

    # compute alpha by the formula (3)
    if softplus == True:
        # (N_rays, N_samples_)
        alphas = 1 - torch.exp(-deltas*torch.nn.functional.softplus(sigmas+noise))
    else:
        # (N_rays, N_samples_)
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise))

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1. -
                  alphas+1e-10], -1)  # [1, a1, a2, ...]
    # (N_rays, N_samples_)
    weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]
    # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    opacity_map = torch.sum(weights, -1)
    # weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # compute final weighted outputs
    if far is not None:
        z_vals_appended = torch.cat([z_vals, far], dim=-1)
        weights_appended = torch.cat(
            [weights, 1-weights.sum(dim=1, keepdim=True)], dim=1)
        depths = torch.sum(weights_appended*z_vals_appended, -1)
    else:
        depths = torch.sum(weights*z_vals, -1) 

    if sigma_only:
        rgb_map = torch.tensor([-1.])
    else:
        # weights_normed = weights / (weights.sum(1, keepdim=True) + 1e-6)
        rgb_map = torch.sum(weights.unsqueeze(-1)*rgbs, -2)  # (N_rays, 3)

        if white_bkgd:
            rgb_map = rgb_map + 1-weights.sum(-1, keepdim=True)

    if ret_var:
        variance = (weights * (depths.view(-1, 1) - z_vals)**2).sum(dim=1)
        return rgb_map, depths, weights, opacity_map, variance

    return rgb_map, depths, weights, opacity_map, None

def inference(model, xyz_, dir_, sigma_only=False, netchunk=32768, detach_sigma=True, meshing=False):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model instantiated using Tiny Cuda NN
        xyz_: (N_rays, N_samples_, 3) sampled positions
                N_samples_ is the number of sampled points in each ray;
        dir_: (N_rays, 3) ray directions
        sigma_only: do inference on sigma only or not
    Outputs:
        if sigma_only:
            raw: (N_rays, N_samples_, 1): predictions of each sample
        else:
            raw: (N_rays, N_samples_, num_colors + 1): predictions of each sample
    """
    N_rays, N_samples_ = xyz_.shape[0:2]
    xyz_ = xyz_.view(-1, 3).contiguous()  # (N_rays*N_samples_, 3)
    if sigma_only:
        dir_ = None
    else:
        # (N_rays*N_samples_, embed_dir_channels)
        dir_ = torch.repeat_interleave(
            dir_, repeats=N_samples_, dim=0).contiguous()

    # Perform model inference to get color and raw sigma
    B = xyz_.shape[0]
    if netchunk == 0:
        out = model(xyz_, dir_, sigma_only, detach_sigma)
    else:
        out_chunks = []
        for i in range(0, B, netchunk):
            out_chunks += [model(xyz_, dir_, sigma_only, detach_sigma)]
        out = torch.cat(out_chunks, 0)

    if meshing:
        return out
    else:
        return out.view(N_rays, N_samples_, -1)


# Use Fully Fused MLP from Tiny CUDA NN
# Volumetric rendering
def render_rays(rays,
                ray_sampler,
                nerf_model,
                ray_range,
                scale_factor,
                N_samples=64,
                retraw=False,
                perturb=0,
                white_bkgd=False,
                raw_noise_std=0.,
                netchunk=32768,
                num_colors=3,
                sigma_only=False,
                DEBUG=True,
                detach_sigma=True,
                return_variance = False,
                render_strategy = 'default'
                ):
    """
    Render rays by computing the output of @occ_model, sampling points based on the class probabilities, applying volumetric rendering using @tcnn_model applied on sampled points

    Inputs:
        rays: (N_rays, 3+3+3+2+?), ray origins, ray directions, unit-magnitude viewing direction, pixel coordinates, ray bin centers
        models: Occupancy Model and NeRF model instantiated using tinycudann
        N_samples: number of samples per ray
        retraw: bool. If True, include model's raw unprocessed predictions.
        lindisp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray
        white_bkgd: whether the background is white (dataset dependent)
        raw_noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final color and depth maps
            color_map: [num_rays, 3 or 1]. Estimated color of a ray.
            depth_map: [num_rays].
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray.
            raw: [num_rays, num_samples, 4 or 1]. Raw predictions from model.
    """
    
    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    viewdirs = rays[:, 6:9]  # (N_rays, 3)
    near = rays[:, -2:-1]
    far = rays[:, -1:]

    z_vals = ray_sampler.get_samples(rays, N_samples, perturb)

    xyz_samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    raw = inference(nerf_model, xyz_samples, viewdirs, netchunk=netchunk, sigma_only=sigma_only, detach_sigma=detach_sigma)

    if render_strategy == 'default':

        rgb, depth, weights, opacity, variance = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, sigma_only=sigma_only, num_colors=num_colors, far=far, ret_var=return_variance)

    elif render_strategy == 'adjusted':
        rgb, depth, weights, opacity, variance = raw2outputs_adjusted(
            raw, z_vals, rays_o, rays_d, raw_noise_std, white_bkgd, sigma_only=sigma_only, num_colors=num_colors, far=far, ret_var=return_variance)

    else:
        raise ValueError(f"Unknown render strategy: {render_strategy}")
    
    result = {'rgb_fine': rgb,
              'depth_fine': depth,
              'weights_fine': weights,
              'opacity_fine': opacity,
              }

    if return_variance:
        result["variance"] = variance

    if retraw:
        result['samples_fine'] = z_vals
        result['points_fine'] = xyz_samples

    if DEBUG:
        result['raw_fine'] = raw

        for k in result:
            if (torch.isnan(result[k]).any() or torch.isinf(result[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")
    return result