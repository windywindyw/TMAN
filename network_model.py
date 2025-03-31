import numpy as np
import torchvision
import torch
from torch import nn
import timm


def model_net(flag=17):
    if flag == 0:  # 权重大小：42.7M
        model = torchvision.models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(512, 8)
        model_name = '0-2015-resnet18'

    elif flag == 1:
        model = torchvision.models.convnext_tiny('DEFAULT')
        model.classifier[2] = nn.Linear(768, 8)
        model_name = '1-2020-convnext_tiny'

    elif flag == 2:  # 权重大小：46.2M
        model = torchvision.models.swin_v2_t('DEFAULT')
        model.head = nn.Linear(768, 8)
        model_name = "2-2021-swin_v2_t"

    elif flag == 3:
        model = torchvision.models.efficientnet_v2_s('DEFAULT')
        model.classifier[1] = nn.Linear(1280, 8)
        model_name = "3-2021-EfficientNet_V2_S_Weights"

    elif flag == 4:
        model = torchvision.models.maxvit_t('DEFAULT')
        model.classifier[5] = nn.Linear(512, 8)
        model_name = '4-2022-maxvit_t'

    # elif flag == 5:
    #     from safetensors.torch import load_file
    #     model = timm.create_model("hiera_small_224", pretrained=False)
    #     model.load_state_dict(load_file("Other_Models/hiera_small.safetensors"), strict=True)
    #     model.head.fc = nn.Linear(768, 8)
    #     model_name = '5-2023-hiera'

    elif flag == 6:
        from safetensors.torch import load_file
        model = timm.create_model("nextvit_small", pretrained=False)
        model.load_state_dict(load_file("Other_Models/nextvit_small.safetensors"), strict=True)
        model.head.fc = nn.Linear(1024, 8)
        model_name = '6-2023-nextvit_small'

    elif flag == 7:
        from safetensors.torch import load_file
        model = timm.create_model("hgnetv2_b3", pretrained=False)
        model.load_state_dict(load_file("Other_Models/hgnetv2.safetensors"), strict=True)
        model.head.fc = nn.Linear(2048, 8)
        model_name = "7-2023-hgnetv2"

    elif flag == 8:
        from safetensors.torch import load_file
        model = timm.create_model("xcit_tiny_12_p16_224.", pretrained=False)
        model.load_state_dict(load_file("Other_Models/xcit_tiny.safetensors"), strict=True)
        model.head = nn.Linear(192, 8)
        model_name = "8-2021-xcit_tiny"

    elif flag == 9:
        # from NewNet_B import BreastBiomarkerNet
        # model = BreastBiomarkerNet()
        # model_name = "9-convnext_tiny-Baseline"

        # from NewNet_B_EMA import BreastBiomarkerNet
        # model = BreastBiomarkerNet()
        # model_name = "9-B_EMA"

        # from NewNet_B_EMA_EM import BreastBiomarkerNet
        # model = BreastBiomarkerNet()
        # model_name = "9-B_EMA_EM"

        from NewNet_B_EMA_EM_FM import BreastBiomarkerNet
        model = BreastBiomarkerNet()
        model_name = "9-B_EMA_EM_FM"


    elif flag == 10:
        from safetensors.torch import load_file
        model = timm.create_model("timm/efficientvit_b1.r224_in1k", pretrained=False)
        model.load_state_dict(load_file("Other_Models/efficientvit.safetensors"), strict=True)
        model.head.classifier[4] = nn.Linear(1600, 8)
        model_name = "10-2023-efficientvit_b1"


    elif flag == 11:
        from Other_Models.EfficientMod.models.EfficientMod import efficientMod_s
        model = efficientMod_s()

        checkpoint = torch.load("Other_Models/EfficientMod_s_model_best.pth.tar")
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        model.head = nn.Linear(312, 8)
        model_name = "11-2024-EfficientMod_s"

    elif flag == 12:
        from Other_Models.BUSSC.BUSSC import MyModel
        model = MyModel()
        model_name = "12-2024-BUSSC"


    elif flag == 13:
        from Other_Models.DRDANet.drdanet import DRDANet
        model = DRDANet()
        model_name = "13-2022-DRDANet"

    elif flag == 14:
        from Other_Models.HATNet.mi_model_e2e import MIModel
        model = MIModel(n_classes=8)
        model_name = "14-2022-MIModel"

    elif flag == 15:
        from Other_Models.ACSNet.MyModel import MyModel
        model = MyModel()
        model_name = "15-2024-ACSNet"

    elif flag == 16:
        from Other_Models.HED.train_breastCancer_ResNet50 import Network_Wrapper
        model = Network_Wrapper()
        model_name = "16-2025-HED"

    elif flag == 17:
        from Other_Models.RMCSAM.Network_Warper import Network_Warper
        model = Network_Warper()
        model_name = "17-2022-RMCSAM"


    else:
        model = ""
        model_name = ""

    return model, model_name


if __name__ == '__main__':
    from thop import profile
    import time

    start = time.time()
    device = torch.device("cuda")

    x = torch.randn([1, 3, 224, 224]).to(device)
    model, model_name = model_net()
    print(model)
    model.to(device)
    memory_before = torch.cuda.memory_allocated(device)
    out = model(x)

    print("推理时间：" + str((time.time() - start)))
    memory_after = torch.cuda.memory_allocated(device)
    max_memory_used = torch.cuda.max_memory_allocated(device)
    # print("Memory used before:", memory_before/ 1e6)
    print("Memory used after:", memory_after / 1e6)
    # print("Max memory used:", max_memory_used/ 1e6)
    print("Output shape:", out.shape)

    flops, params = profile(model, inputs=(x,))
    print("FLOPs：" + str(round(flops / 1e9, 2)) + "G")
    print("参数量：" + str(round(params / 1e6 * 4, 2)) + "M")
