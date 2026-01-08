import torch
import torchaudio
import os
from model import EncodecModel

# ================= 配置部分 (请务必与训练时的设置保持一致) =================
# 根据你之前的命令行：model.ratios=[8,5,4,2]
RATIOS = [8, 5, 4, 2] 

# 采样率：通常 Encodec 默认为 24000，如果你训练数据是 48k 或其他，请修改
SAMPLE_RATE = 24000 
CHANNELS = 1          # 你的训练代码里看起来是单声道？如果是立体声改成 2
TARGET_BANDWIDTHS = [1.5, 3.0, 6.0, 12.0, 24.0] # 模型支持的带宽列表

# 这里的 checkpoint 路径换成你想测试的那一个（注意选较大的那个 .pt 文件，不是 disc）
CHECKPOINT_PATH = "./outputs/encodec320x_ratios8542_lt960_bs2_tc48_lr3e-4_wup5/bs2_cut48000_length500_epoch117_lr0.0003.pt"

# 输入和输出文件
INPUT_WAV = "./outputs/encodec320x_ratios8542_lt960_bs2_tc48_lr3e-4_wup5/GT.wav"
# 输出音频路径
OUTPUT_WAV = "./outputs/encodec320x_ratios8542_lt960_bs2_tc48_lr3e-4_wup5/reconstructed_GT.wav"

# =========================================================================

def load_model(checkpoint_path, sample_rate, channels, ratios, bandwidths):
    print(f"Loading model from {checkpoint_path}...")
    
    # 1. 初始化模型结构 (参数必须与训练时一致)
    # 注意：这里的参数是为了匹配 EncodecModel._get_model 的签名
    # 如果你的 config 中 causal=True, model_norm='weight_norm' 等有改动，这里也需要调整
    model = EncodecModel._get_model(
        target_bandwidths=bandwidths,
        sample_rate=sample_rate,
        channels=channels,
        causal=True,        # 默认通常为 True
        model_norm='weight_norm', 
        audio_normalize=False,
        segment=None,       # 推理时通常不需要切片
        name='my_encodec',
        ratios=ratios
    )
    
    # 2. 加载权重
    # 你的 checkpoint 包含了 'model_state_dict', 'optimizer', 'epoch' 等
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理一下 state_dict 的 key，如果是因为 DDP 训练导致有 'module.' 前缀
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") # 移除 module. 前缀
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()
    return model

@torch.no_grad()
def inference():
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_WAV):
        print(f"错误: 找不到输入文件 {INPUT_WAV}")
        return

    # 1. 加载模型
    model = load_model(CHECKPOINT_PATH, SAMPLE_RATE, CHANNELS, RATIOS, TARGET_BANDWIDTHS)
    
    # 2. 加载并预处理音频
    wav, sr = torchaudio.load(INPUT_WAV)
    
    # 重采样 (如果输入音频采样率与模型不一致)
    if sr != SAMPLE_RATE:
        print(f"Resampling from {sr} to {SAMPLE_RATE}...")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = resampler(wav)
    
    # 混合成单声道/多声道以匹配模型 (如果需要)
    if wav.shape[0] != CHANNELS:
        if CHANNELS == 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif CHANNELS == 2 and wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
            
    # 添加 Batch 维度: [C, T] -> [1, C, T]
    wav = wav.unsqueeze(0).cuda()

    # 3. 模型推理
    print("Running inference...")
    # 根据你的 train 代码: output, loss_w, _ = model(input_wav)
    # 模型返回的是元组，第一个元素是重建后的音频
    reconstructed = model(wav)
    
    # 4. 保存结果
    # 移除 Batch 维度并转回 CPU
    reconstructed = reconstructed.squeeze(0).cpu()
    
    torchaudio.save(OUTPUT_WAV, reconstructed, SAMPLE_RATE)
    print(f"Success! Reconstruction saved to: {OUTPUT_WAV}")

if __name__ == "__main__":
    inference()