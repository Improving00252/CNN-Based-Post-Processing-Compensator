import torch
import torch.nn.functional as F
import os

def set_seed(seed):
    torch.manual_seed(seed)

def gaussian_blur_mask(channel, sigma):
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=channel.device)
    gauss_kernel = torch.exp(-0.5 * (x / sigma)**2)
    gauss_kernel = gauss_kernel / gauss_kernel.sum()

    gauss_kernel = gauss_kernel.unsqueeze(0).unsqueeze(0)
    blurred_channel = F.conv2d(channel.unsqueeze(0).unsqueeze(0), gauss_kernel.unsqueeze(1), padding='same')
    blurred_channel = F.conv2d(blurred_channel, gauss_kernel.unsqueeze(2), padding='same')
    
    return blurred_channel.squeeze(0)

def apply_gaussian_blur_blockwise_adaptive_sigma(decoded_feature, block_height=19, block_width=34):
    if len(decoded_feature.size()) == 3:
        _, height, width = decoded_feature.size()
        is_3d = True
    elif len(decoded_feature.size()) == 2:
        height, width = decoded_feature.size()
        is_3d = False
    else:
        raise ValueError("decoded_feature must be 2D or 3D tensor")
    
    blurred_feature = torch.zeros_like(decoded_feature)
    
    for i in range(0, height, block_height):
        for j in range(0, width, block_width):
            if is_3d:
                block = decoded_feature[:, i:i+block_height, j:j+block_width]
            else:
                block = decoded_feature[i:i+block_height, j:j+block_width]

            mean = torch.mean(block)
            variance = torch.var(block)
            sigma = torch.sqrt(variance)
            
            blurred_block = gaussian_blur_mask(block.squeeze(0), sigma)
            
            if is_3d:
                blurred_feature[:, i:i+block_height, j:j+block_width] = blurred_block
            else:
                blurred_feature[i:i+block_height, j:j+block_width] = blurred_block
    
    return blurred_feature

def generate_gaussian_mask(feature_maps):
    set_seed(1234)
    blurred_channels = []
    channels = feature_maps.squeeze(0)

    for i in range(channels.size(0)):
        channel = channels[i]
        blurred_channel = apply_gaussian_blur_blockwise_adaptive_sigma(channel)
        blurred_channels.append(blurred_channel)
    
    blurred_feature_maps = torch.stack(blurred_channels)
    blurred_feature_maps = blurred_feature_maps.unsqueeze(0)

    return blurred_feature_maps


previous_frame = {105: None, 90: None, 75: None}

def reset_previous_frame():
    global previous_frame
    previous_frame = {105: None, 90: None, 75: None}


def residual_mask(current_feature_maps, reference_feature_maps):
    if reference_feature_maps is None:
        return current_feature_maps

    return current_feature_maps - reference_feature_maps

def generate_residual_mask(feature_maps, key):
    global previous_frame
    channels = feature_maps.squeeze(0)
    residual_channels = []

    for i in range(channels.size(0)):
        reference_channel = None if previous_frame[key] is None else previous_frame[key][i]
        residual_channel = residual_mask(channels[i], reference_channel)
        residual_channels.append(residual_channel)

    residual_feature_maps = torch.stack(residual_channels)
    residual_feature_maps = residual_feature_maps.unsqueeze(0)

    previous_frame[key] = channels.clone()
    
    return residual_feature_maps
