import os
import glob
import soundfile as sf
import librosa
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

# ================= 配置区域 =================
INPUT_DIR = "/mnt/15T/wjc/LibriSpeech/test-clean/1089"   # 输入源目录
OUTPUT_DIR = "./demo/audios"         # 输出目录（所有文件将平铺在这里）
TARGET_SR = 24000                             # 目标采样率 24k
TARGET_SUBTYPE = 'PCM_16'                     # 目标位深 16-bit
NUM_WORKERS = os.cpu_count()                  # 并行进程数
# ===========================================

def convert_and_flatten(file_path):
    """
    读取音频，重采样，强制单声道，并写入到统一目录
    """
    try:
        # 1. 准备输出路径
        # 获取文件名 (例如: 1089-134686-0000.flac)
        filename = os.path.basename(file_path)
        # 更改后缀为 .wav
        wav_name = os.path.splitext(filename)[0] + ".wav"
        # 拼接输出路径 (不保留原目录结构)
        output_path = os.path.join(OUTPUT_DIR, wav_name)

        # 2. 如果文件已存在，跳过（可选）
        # if os.path.exists(output_path):
        #     return True

        # 3. 加载并重采样
        # librosa.load 默认会归一化数据，并转换为 float32
        # sr=TARGET_SR 会自动进行重采样 (16k -> 24k)
        # mono=True 强制混合为单声道
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # 4. 写入 WAV
        # soundfile 会自动处理 float32 -> int16 的抖动转换
        sf.write(output_path, y, sr, subtype=TARGET_SUBTYPE)
        
        return True
    except Exception as e:
        print(f"\nError converting {file_path}: {e}")
        return False

def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")

    print(f"正在扫描: {INPUT_DIR} ...")
    # 递归查找所有 .flac
    files = glob.glob(os.path.join(INPUT_DIR, "**", "*.flac"), recursive=True)
    
    if not files:
        print("未找到文件。")
        return

    print(f"找到 {len(files)} 个文件。目标格式: {TARGET_SR}Hz, {TARGET_SUBTYPE}, Flat Directory")

    # 并行处理
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(convert_and_flatten, files), total=len(files), unit="wav"))

    print(f"\n转换完成。所有文件均已输出至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()