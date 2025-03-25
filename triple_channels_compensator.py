import torch
from .three_channels_models import Three_Channels_RDN
from .utils import generate_residual_mask, generate_gaussian_mask

def three_channels_compensator(test_data,
                     weights_dir = "/directory/of/your/weights/file/", 
                     num_features=64, 
                     growth_rate=64, 
                     num_blocks=16, 
                     num_layers=8,
                     gaussian_num=8,
                     residual_num=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    keys = [105, 90, 75]
    #keys = [105]
    num_channels_dict = {105: 128, 90: 256, 75: 512}
    
    models = {}

    for key in keys:
        num_channels = num_channels_dict[key]
        model = Three_Channels_RDN(
            num_channels=num_channels,
            num_features=num_features,
            growth_rate=growth_rate,
            num_blocks=num_blocks,
            num_layers=num_layers,
            gaussian_num=gaussian_num,
            residual_num=residual_num).to(device)
        weights_file = f'{weights_dir}/best_key_{key}.pth'
        model.load_state_dict(torch.load(weights_file, map_location=device))
        model.eval()
        models[key] = model
    
    enhanced_data = {}
    for key in keys:
        model = models[key]
        data = test_data[key].to(device)
        #generate gaussian mask data
        gaussian_data = generate_gaussian_mask(data).to(device)
        #generate residual data
        residual_data = generate_residual_mask(data, key).to(device)

        with torch.no_grad():
            enhanced_data[key] = model(data, gaussian_data, residual_data).cpu()
    
    return enhanced_data



# Example usage:
# test_data = {
#     105: torch.randn(1, 128, 76, 136),  # Replace with your actual test data
#     90: torch.randn(1, 256, 38, 68),
#     75: torch.randn(1, 512, 19, 34)
# }
# weights_dir = './model_weights'
# enhanced_data = fctm_compensator(test_data, weights_dir)
# print(enhanced_data)
