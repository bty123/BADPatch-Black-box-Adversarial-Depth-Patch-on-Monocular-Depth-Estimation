import os
import sys
import torch
import torch.nn as nn
sys.path.append('.')
from config import Config
import depth_anything_v2
import networks

file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.path.dirname(file_dir)

class liteModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(liteModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        depth = (disp - disp.min()) / (disp.max() - disp.min())

        return depth

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        depth = (disp - disp.min()) / (disp.max() - disp.min())

        return depth

class SQLdepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(SQLdepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        disp = nn.functional.interpolate(disp, input_image.shape[-2:], mode='bilinear', align_corners=True)
        disp = 1/disp
        depth = (disp - disp.min()) / (disp.max() - disp.min())
        return depth


def import_depth_model(model_type = 'monodepth2'):
    if model_type == 'monodepth2':
        model_name = 'mono+stereo_1024x320'
        code_path = os.path.join(file_dir, 'DepthNetworks', 'monodepth2')
        depth_model_dir = os.path.join(code_path, 'models')
        sys.path.append(code_path)
        model_path = os.path.join(depth_model_dir, model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        loaded_dict_enc = torch.load(encoder_path, map_location=Config.device)
        encoder = networks.ResnetEncoder(18, False)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        print("Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=Config.device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_model = DepthModelWrapper(encoder, depth_decoder)

        return depth_model

    elif model_type == 'depthhints':
        model_name = 'DH_MS_320_1024'
        code_path = os.path.join(file_dir, 'DepthNetworks', 'depth-hints')
        depth_model_dir = os.path.join(code_path, 'models')
        sys.path.append(code_path)
        model_path = os.path.join(depth_model_dir, model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        loaded_dict_enc = torch.load(encoder_path, map_location=Config.device)
        encoder = networks.ResnetEncoder(18, False)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        print("Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=Config.device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_model = DepthModelWrapper(encoder, depth_decoder)

        return depth_model
    
    elif model_type == 'lite_mono':
        encoder_path = os.path.join(file_dir, 'DepthNetworks', 'lite_mono', 'lite-mono_1024x320', "encoder.pth")
        decoder_path = os.path.join(file_dir, 'DepthNetworks', 'lite_mono', 'lite-mono_1024x320', "depth.pth")
        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)
        encoder = networks.LiteMono(model='lite-mono', height=encoder_dict['height'], width=encoder_dict['width'])
        depth_decoder = networks.DepthDecoder_lite(encoder.num_ch_enc, scales=range(3))
        model_dict = encoder.state_dict()
        depth_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})
        depth_model = liteModelWrapper(encoder, depth_decoder)

        return depth_model
    
    elif model_type == 'SQLdepth':
        model_name = 'KITTI_320x1024_models'
        # model_name = 'ConvNeXt_Large_SQLdepth'
        code_path = os.path.join(file_dir, 'DepthNetworks', 'SQLdepth')
        depth_model_dir = os.path.join(code_path, 'models')
        sys.path.append(code_path)
        model_path = os.path.join(depth_model_dir, model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        if model_name == 'KITTI_320x1024_models':
            encoder = networks.ResnetEncoderDecoder(num_layers=50, num_features=256, model_dim=32)
            depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=32, patch_size=20, dim_out=128, embedding_dim=32,
                                                        query_nums=128, num_heads=4, min_val=0.1, max_val=80) 
        elif model_name == 'ConvNeXt_Large_SQLdepth':
            encoder = networks.Unet(
                pretrained=False, 
                backbone='convnext_large', 
                in_channels=3, 
                num_classes=32, 
                decoder_channels=[1024, 512, 256, 128])
            depth_decoder = networks.Depth_Decoder_QueryTr(in_channels=32, patch_size=32, dim_out=64, embedding_dim=32,
                                                        query_nums=64, num_heads=4, min_val=0.1, max_val=80) 
        else:
            print('error model')
        loaded_dict_enc = torch.load(encoder_path, map_location=Config.device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict_enc = torch.load(depth_decoder_path, map_location=Config.device)
        depth_decoder.load_state_dict(loaded_dict_enc)
        depth_model = SQLdepthModelWrapper(encoder, depth_decoder)

        return depth_model

    elif model_type == 'MiDaS':
        from MiDaS.midas.model_loader import load_model as load_midas_model
        model, _, _, _ = load_midas_model(
            torch.device(Config.device),
            './checkpoints/midas_v21_small_256.pt',
            'midas_v21_small_256',
            False, 320, False
        ) # midas_v21_small_256.pt  dpt_beit_large_512.pt dpt_beit_base_384.pt
        depth_model = model.to(Config.device).eval()

        return depth_model

    elif model_type == 'DepthAnything':
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder_name = 'vits'
        depth_anything = depth_anything_v2.DepthAnythingV2(**model_configs[encoder_name])
        depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{encoder_name}.pth', map_location=Config.device))
        depth_model = depth_anything.to(Config.device).eval()

        return depth_model
