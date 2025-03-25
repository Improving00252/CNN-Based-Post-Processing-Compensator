# CNN-Based-Post-Processing-Compensator
# Multi-Channel RDN Post-Processing Compensator

This repository contains the implementation of a **multi-channel post-processing module** based on the [RDN (Residual Dense Network)](https://github.com/yjn870/RDN-pytorch).  
It is designed to compensate for compression-induced distortions in intermediate feature maps for machine vision tasks.

## Features

- Adapted RDN model to support multi-channel input/output.
- Designed specifically for post-processing in **Feature Coding for Machine (FCM)** pipelines.
- Compatible with PyTorch.

## Usage

    from .models import RDN\
    from .fctm_compensator import fctm_compensator\
    from .two_channels_models import Two_Channels_RDN\
    from .two_channels_compensator import two_channels_compensator\
    from .three_channels_models import Three_Channels_RDN\
    from .three_channels_compensator import three_channels_compensator\
    x["data"] = fctm_compensator(x["data"])\
    x["data"] = two_channels_compensator(x["data"])\
    x["data"] = three_channels_compensator(x["data"])\
    
## 📦 Pretrained Models

| Branch         | Description                              | Download Link |
|----------------|------------------------------------------|---------------|
| `main`         | Single-channel baseline                  | [Download](https://github.com/Improving00252/CNN-Based-Post-Processing-Compensator/releases/tag/single) |
| `two_channel`  | Two-channel with Gaussian mask           | [Download](https://github.com/Improving00252/CNN-Based-Post-Processing-Compensator/releases/tag/double_channel_compensator) |
| `three_channel`| Three-channel with residual & blur mask  | [Download](https://github.com/Improving00252/CNN-Based-Post-Processing-Compensator/releases/tag/triple_channels_compensator) |


## Acknowledgements

This project is based on the open-source implementation of RDN by [@yjn870](https://github.com/yjn870):

🔗 https://github.com/yjn870/RDN-pytorch

I modified the original model to support a multi-channel structure for feature post-processing tasks.  
The original repository is licensed under the MIT License (see below).

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
