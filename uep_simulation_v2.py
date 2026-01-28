#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端语义感知UEP传输仿真
==========================

基于RVQ语义分层的非等差错保护(UEP)传输仿真，与Encodec模型集成。

功能模块:
---------
1. 理论计算模块 - CSS SNR-BER精确计算
2. VLC信道模块 - NLOS可见光信道建模（非RF衰落）
3. Encodec集成模块 - 端到端编解码
4. 仿真引擎 - 多策略、多带宽对比
5. 图表生成模块 - 独立函数，方便调试导出

VLC信道特点（与RF的区别）:
--------------------------
- 无小尺度衰落（孔径平均效应）
- 大尺度衰落由遮挡和路径损耗主导
- 多径导致ISI而非衰落

Author: For thesis "基于语义压缩的非视距可见光通信系统的研究与实现"
"""

import argparse
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from collections import OrderedDict
import warnings

import numpy as np

# 可选导入: matplotlib用于绑图
try:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # 设置matplotlib中文支持
    try:
        rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        rcParams['axes.unicode_minus'] = False
    except:
        pass
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, plotting disabled")


###############################################################################
#                     第1部分: 理论计算模块
###############################################################################

class TheoreticalBERCalculator:
    """
    CSS/LoRa调制的理论BER计算器
    
    使用基于实测SNR阈值的经验模型，具有数值稳定性。
    原始闭式解公式对于大M值(M=2^SF, SF=7-12时M=128-4096)会产生数值溢出问题。
    
    经验模型基于:
    - 每个SF在特定SNR阈值处达到可靠通信(BER ≈ 10^-5)
    - 瀑布曲线斜率约为每2dB改善一个数量级
    
    参考: Semtech AN1200.22, LoRa物理层测试报告
    """
    
    # 各SF的实测SNR解调阈值 (dB) @ BER ≈ 10^-5
    # 数据来源: Semtech官方文档和实测数据
    SNR_THRESHOLDS = {
        7:  -7.5,
        8:  -10.0,
        9:  -12.5,
        10: -15.0,
        11: -17.5,
        12: -20.0,
    }
    
    # 瀑布曲线斜率参数: 每slope_db dB SNR改善，BER降低一个数量级
    WATERFALL_SLOPE_DB = 2.0
    
    @classmethod
    def calculate_ber(cls, snr_dB: float, SF: int) -> float:
        """
        计算给定SNR和SF下的BER (经验模型)
        
        模型逻辑:
        - SNR > threshold + 10dB: BER ≈ 1e-10 (几乎无误码)
        - SNR 在 threshold 附近: BER 呈瀑布下降
        - SNR < threshold - 10dB: BER ≈ 0.5 (完全无法通信)
        
        Args:
            snr_dB: 信噪比 (dB)
            SF: 扩频因子 (7-12)
        
        Returns:
            误比特率 BER
        """
        threshold = cls.SNR_THRESHOLDS.get(SF, -10.0)
        delta = snr_dB - threshold  # SNR相对于阈值的偏移
        
        # 在阈值处定义BER = 10^-5
        # 使用指数瀑布模型: BER = 10^(-5 - delta/slope)
        # 其中slope控制瀑布曲线的陡峭程度
        
        if delta > 10:
            # 高SNR区域: 几乎无误码
            return 1e-10
        elif delta > -10:
            # 瀑布区域: 指数衰减
            # 在delta=0时BER=10^-5, 每增加2dB降低一个数量级
            exponent = -5 - (delta / cls.WATERFALL_SLOPE_DB)
            ber = 10 ** exponent
            return np.clip(ber, 1e-10, 0.5)
        else:
            # 极低SNR区域: 无法解调
            return 0.5
    
    @classmethod
    def calculate_ser(cls, snr_dB: float, SF: int) -> float:
        """计算误符号率SER"""
        ber = cls.calculate_ber(snr_dB, SF)
        # SER到BER的近似转换: BER ≈ SER * (M/2) / (M-1) ≈ SER * 0.5 for large M
        # 反推: SER ≈ BER * 2
        return min(ber * 2.0, 1.0)
    
    @classmethod
    def generate_snr_ber_curve(cls, SF: int, 
                                snr_range: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成SNR-BER曲线数据
        
        Args:
            SF: 扩频因子
            snr_range: SNR范围，默认覆盖阈值两侧
        
        Returns:
            (snr_range, ber_values)
        """
        if snr_range is None:
            threshold = cls.SNR_THRESHOLDS.get(SF, -10)
            snr_range = np.linspace(threshold - 10, threshold + 15, 100)
        
        ber_values = np.array([cls.calculate_ber(snr, SF) for snr in snr_range])
        return snr_range, ber_values
    
    @classmethod
    def get_required_snr(cls, target_ber: float, SF: int) -> float:
        """
        计算达到目标BER所需的SNR
        
        Args:
            target_ber: 目标BER
            SF: 扩频因子
        
        Returns:
            所需SNR (dB)
        """
        threshold = cls.SNR_THRESHOLDS.get(SF, -10.0)
        
        if target_ber >= 0.5:
            return threshold - 10
        elif target_ber <= 1e-10:
            return threshold + 10
        else:
            # 反推公式: delta = (-5 - log10(BER)) * slope
            log_ber = np.log10(target_ber)
            delta = (-5 - log_ber) * cls.WATERFALL_SLOPE_DB
            return threshold + delta


class CSSOrthogonalityModel:
    """
    CSS不同SF之间的正交性模型
    
    基于Semtech正交性矩阵，用于分析并行SF传输可行性。
    """
    
    # 正交性矩阵: [victim][aggressor] = 隔离度(dB)
    ORTHOGONALITY_MATRIX = {
        7:  {7: -6,  8: -16, 9: -19, 10: -22, 11: -25, 12: -28},
        8:  {7: -24, 8: -9,  9: -19, 10: -22, 11: -25, 12: -28},
        9:  {7: -27, 8: -27, 9: -12, 10: -22, 11: -25, 12: -28},
        10: {7: -30, 8: -30, 9: -30, 10: -15, 11: -25, 12: -28},
        11: {7: -33, 8: -33, 9: -33, 10: -33, 11: -18, 12: -28},
        12: {7: -36, 8: -36, 9: -36, 10: -36, 11: -36, 12: -20},
    }
    
    @staticmethod
    def get_data_rate(sf: int, bandwidth_hz: float = 125000) -> float:
        """计算SF对应的数据速率 (bps): R = SF * BW / 2^SF"""
        return sf * bandwidth_hz / (2 ** sf)
    
    @classmethod
    def get_isolation(cls, victim_sf: int, aggressor_sf: int) -> float:
        """获取两个SF之间的隔离度 (dB)"""
        return cls.ORTHOGONALITY_MATRIX.get(victim_sf, {}).get(aggressor_sf, -30)


###############################################################################
#                     第2部分: VLC信道模块
###############################################################################

class VLCNLOSChannel:
    """
    NLOS可见光通信信道模型
    
    特点（与RF信道的本质区别）:
    1. 无小尺度衰落 - 光电探测器孔径平均效应消除相位干涉
    2. 大尺度衰落由遮挡(Blockage)主导 - 人体/物体遮挡导致10-20dB跳变
    3. 多径导致ISI而非衰落 - 时延扩展典型值2-50ns
    
    参考: NLOS VLC信道研究报告
    """
    
    def __init__(self, 
                 reflectivity: float = 0.8,        # 墙面反射率(白色石膏板0.8-0.9)
                 rms_delay_ns: float = 10.0,       # 均方根时延扩展(办公室2-10ns)
                 blockage_prob: float = 0.0,       # 遮挡事件概率
                 blockage_loss_db: float = 15.0,   # 遮挡时额外损耗
                 base_path_loss_db: float = 20.0,  # 基础路径损耗(相对LOS)
                 codebook_size: int = 1024):
        """
        初始化VLC NLOS信道
        
        Args:
            reflectivity: 墙面反射率 (0-1)
            rms_delay_ns: RMS时延扩展 (纳秒)
            blockage_prob: 每帧遮挡概率 (0-1)
            blockage_loss_db: 遮挡导致的额外损耗 (dB)
            base_path_loss_db: NLOS相对LOS的基础路径损耗 (dB)
            codebook_size: RVQ码本大小
        """
        self.reflectivity = reflectivity
        self.rms_delay_ns = rms_delay_ns
        self.blockage_prob = blockage_prob
        self.blockage_loss_db = blockage_loss_db
        self.base_path_loss_db = base_path_loss_db
        self.codebook_size = codebook_size
        self.bits_per_code = int(np.ceil(np.log2(codebook_size)))
        
        # 相干带宽估算: Bc ≈ 1/(5*τ_rms)
        self.coherence_bandwidth_mhz = 1e3 / (5 * rms_delay_ns) if rms_delay_ns > 0 else 100
        
        self.ber_calculator = TheoreticalBERCalculator()
    
    def get_effective_snr(self, channel_snr_db: float, sf: int) -> float:
        """
        计算考虑VLC信道特性后的有效SNR
        
        包括:
        - 基础路径损耗
        - 随机遮挡事件
        - ISI惩罚(当符号速率接近相干带宽时)
        """
        effective_snr = channel_snr_db - self.base_path_loss_db
        
        # 遮挡事件（大尺度衰落）
        if np.random.random() < self.blockage_prob:
            effective_snr -= self.blockage_loss_db
        
        # ISI惩罚（简化模型：当符号速率高时，性能下降）
        # 符号速率 = BW / 2^SF, 对于SF越大，符号速率越低，ISI影响越小
        symbol_rate_khz = 125 / (2 ** sf)  # 假设BW=125kHz
        if symbol_rate_khz > self.coherence_bandwidth_mhz * 1000:
            isi_penalty = 3.0  # ISI造成约3dB损耗
            effective_snr -= isi_penalty
        
        return effective_snr
    
    def get_ber(self, channel_snr_db: float, sf: int) -> float:
        """计算VLC信道下的BER"""
        effective_snr = self.get_effective_snr(channel_snr_db, sf)
        return self.ber_calculator.calculate_ber(effective_snr, sf)
    
    def corrupt_codes(self, codes: np.ndarray, ber: float) -> np.ndarray:
        """根据BER腐蚀RVQ码"""
        if ber <= 0:
            return codes.copy()
        if ber >= 0.5:
            return np.random.randint(0, self.codebook_size, codes.shape)
        
        corrupted = np.zeros_like(codes)
        for idx in np.ndindex(codes.shape):
            code = int(codes[idx])
            bits = [(code >> i) & 1 for i in range(self.bits_per_code)]
            for i in range(len(bits)):
                if np.random.random() < ber:
                    bits[i] = 1 - bits[i]
            new_code = sum(b << i for i, b in enumerate(bits))
            corrupted[idx] = min(new_code, self.codebook_size - 1)
        
        return corrupted
    
    def transmit_layer(self, codes: np.ndarray, sf: int, 
                        channel_snr_db: float) -> Tuple[np.ndarray, Dict]:
        """传输单层RVQ码"""
        effective_snr = self.get_effective_snr(channel_snr_db, sf)
        ber = self.ber_calculator.calculate_ber(effective_snr, sf)
        corrupted = self.corrupt_codes(codes, ber)
        
        stats = {
            "sf": sf,
            "channel_snr_db": channel_snr_db,
            "effective_snr_db": effective_snr,
            "ber": ber,
            "code_error_rate": np.mean(codes != corrupted),
        }
        return corrupted, stats


###############################################################################
#                     第3部分: Encodec集成模块
###############################################################################

class EncodecWrapper:
    """
    Encodec模型封装器
    
    提供统一的编解码接口，支持:
    - 预训练模型加载
    - 自定义checkpoint加载
    - 多带宽配置(1.5/3/6/12/24 kbps)
    """
    
    # 带宽与码本数量的映射
    # Encodec: 每秒75帧，每码本10bit
    # 带宽(kbps) = 75 * n_codebooks * 10 / 1000
    BANDWIDTH_TO_CODEBOOKS = {
        1.5: 2,   # 75 * 2 * 10 / 1000 = 1.5 kbps
        3.0: 4,   # 75 * 4 * 10 / 1000 = 3.0 kbps
        6.0: 8,   # 75 * 8 * 10 / 1000 = 6.0 kbps
        12.0: 16,
        24.0: 32,
    }
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.sample_rate = 24000
        self.frame_rate = 75  # 每秒潜在帧数
    
    def load_pretrained(self, bandwidth: float = 6.0):
        """加载预训练模型"""
        try:
            from model import EncodecModel
            import torch
            
            print(f"Loading pretrained Encodec model (bandwidth={bandwidth}kbps)...")
            self.model = EncodecModel.encodec_model_24khz(pretrained=True)
            self.model.set_target_bandwidth(bandwidth)
            self.model.eval()
            self.model.to(self.device)
            self.current_bandwidth = bandwidth
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str, bandwidth: float = 6.0):
        """加载自定义checkpoint"""
        try:
            from model import EncodecModel
            import torch
            
            print(f"Loading checkpoint: {checkpoint_path}")
            
            model = EncodecModel._get_model(
                target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
                sample_rate=24000,
                channels=1,
                causal=True,
                model_norm='weight_norm',
                audio_normalize=False,
                segment=None,
                name='my_encodec',
                ratios=[8, 5, 4, 2]
            )
            
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state_dict = ckpt.get('model_state_dict', ckpt)
            
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            model.set_target_bandwidth(bandwidth)
            model.eval()
            model.to(self.device)
            
            self.model = model
            self.current_bandwidth = bandwidth
            print("Checkpoint loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False
    
    def encode(self, audio_path: str) -> Tuple[np.ndarray, any, int]:
        """
        编码音频文件
        
        Returns:
            codes: RVQ码 (n_layers, n_frames)
            scale: 归一化scale
            n_samples: 原始样本数
        """
        import torch
        import torchaudio
        
        wav, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        n_samples = wav.shape[-1]
        wav = wav.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
            codes, scale = encoded_frames[0]
        
        # codes: [1, n_layers, n_frames] -> [n_layers, n_frames]
        codes_np = codes.squeeze(0).cpu().numpy()
        
        return codes_np, scale, n_samples
    
    def decode(self, codes: np.ndarray, scale) -> np.ndarray:
        """
        解码RVQ码
        
        Args:
            codes: RVQ码 (n_layers, n_frames)
            scale: 归一化scale
        
        Returns:
            audio: 重建音频波形
        """
        import torch
        
        codes_tensor = torch.from_numpy(codes).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoded_frame = (codes_tensor, scale)
            audio = self.model._decode_frame(encoded_frame)
        
        return audio.squeeze().cpu().numpy()
    
    def get_n_codebooks(self, bandwidth: float) -> int:
        """获取给定带宽下使用的码本数量"""
        return self.BANDWIDTH_TO_CODEBOOKS.get(bandwidth, 8)


###############################################################################
#                     第4部分: 传输策略（速率感知）
###############################################################################

@dataclass
class RateAwareStrategy:
    """
    速率感知的传输策略
    
    核心思想:
    - 不同SF有不同的数据速率: R = SF * BW / 2^SF
    - 并行传输可以同时使用多个SF，总速率是各SF速率之和
    - 根据总速率决定能传输多少RVQ层
    - Encodec带宽与RVQ层数对应: 1.5kbps=2层, 3kbps=4层, 6kbps=8层
    
    使用示例:
        # 创建UEP策略: 前2层用SF9保护, 其余用SF7
        s = RateAwareStrategy(
            name="UEP-SF9-2",
            mode='parallel',
            parallel_config={9: 2, 7: 6},  # SF9传2层, SF7传6层
            protected_layers=2,  # 前2层使用高SF
        )
    """
    name: str
    description: str = ""
    
    # 策略类型: 'single' (单SF) 或 'parallel' (并行SF)
    mode: str = 'single'
    
    # 单SF模式时使用的SF
    single_sf: int = 9
    
    # 并行模式时的SF配置: {SF: 层数}
    # 例如 {9: 2, 7: 6} 表示SF9传2层(更重要)，SF7传6层(次要)
    parallel_config: Dict[int, int] = field(default_factory=dict)
    
    # 受保护的层数 (使用高SF的层数)
    # 例如 protected_layers=2 表示Layer0和Layer1使用高SF
    protected_layers: int = 1
    
    # 固定的RVQ层数 (如果为None则根据速率自动计算)
    fixed_layers: int = None
    
    def get_total_rate_bps(self, bandwidth_hz: float = 125000) -> float:
        """计算策略的总数据速率 (bps)"""
        if self.mode == 'single':
            return CSSOrthogonalityModel.get_data_rate(self.single_sf, bandwidth_hz)
        else:
            # 并行模式: 各SF速率之和
            total = 0
            for sf in self.parallel_config.keys():
                total += CSSOrthogonalityModel.get_data_rate(sf, bandwidth_hz)
            return total
    
    def get_achievable_layers(self, bandwidth_hz: float = 125000, 
                               frame_rate: float = 75,
                               bits_per_code: int = 10) -> int:
        """
        计算在给定速率下能传输的RVQ层数
        
        每层所需速率 = frame_rate * bits_per_code = 750 bps
        """
        if self.fixed_layers is not None:
            return self.fixed_layers
        
        rate_per_layer = frame_rate * bits_per_code  # 750 bps
        total_rate = self.get_total_rate_bps(bandwidth_hz)
        max_layers = int(total_rate / rate_per_layer)
        
        # 限制在1-8层之间
        return int(np.clip(max_layers, 1, 8))
    
    def get_layer_sf_map(self, n_layers: int) -> Dict[int, int]:
        """
        获取每层使用的SF
        
        Args:
            n_layers: 要传输的层数
        
        Returns:
            {layer_idx: sf} 映射
        """
        if self.mode == 'single':
            return {i: self.single_sf for i in range(n_layers)}
        
        else:  # parallel mode
            # 按SF从高到低排序
            sfs = sorted(self.parallel_config.keys(), reverse=True)
            high_sf = sfs[0] if sfs else 10
            low_sf = sfs[-1] if len(sfs) > 1 else 7
            
            layer_map = {}
            # 前 protected_layers 层使用高SF
            for i in range(min(self.protected_layers, n_layers)):
                layer_map[i] = high_sf
            # 其余层使用低SF
            for i in range(self.protected_layers, n_layers):
                layer_map[i] = low_sf
            
            return layer_map
    
    def get_snr_threshold(self) -> float:
        """获取该策略的核心层SNR阈值"""
        if self.mode == 'single':
            return TheoreticalBERCalculator.SNR_THRESHOLDS.get(self.single_sf, -10)
        else:
            high_sf = max(self.parallel_config.keys())
            return TheoreticalBERCalculator.SNR_THRESHOLDS.get(high_sf, -10)


def make_strategy(name: str, 
                  high_sf: int = 10, 
                  low_sf: int = 7,
                  protected_layers: int = 1,
                  fixed_layers: int = None,
                  description: str = "") -> RateAwareStrategy:
    """
    便捷函数: 创建UEP策略
    
    Args:
        name: 策略名称
        high_sf: 核心层使用的高SF (7-12)
        low_sf: 次要层使用的低SF (7-12)  
        protected_layers: 使用高SF保护的层数
        fixed_layers: 固定层数 (None则自动计算)
        description: 描述
    
    Examples:
        # SF10保护Layer0, SF7传剩余层
        make_strategy("UEP-1", high_sf=10, low_sf=7, protected_layers=1)
        
        # SF9保护Layer0-1, SF7传剩余层
        make_strategy("UEP-2", high_sf=9, low_sf=7, protected_layers=2)
        
        # 固定4层, SF10保护前2层
        make_strategy("UEP-4L", high_sf=10, low_sf=7, protected_layers=2, fixed_layers=4)
    """
    if not description:
        description = f"UEP: 前{protected_layers}层用SF{high_sf}, 其余用SF{low_sf}"
    
    if high_sf == low_sf:
        # EEP模式
        return RateAwareStrategy(
            name=name,
            description=description,
            mode='single',
            single_sf=high_sf,
            fixed_layers=fixed_layers,
        )
    else:
        # UEP并行模式
        return RateAwareStrategy(
            name=name,
            description=description,
            mode='parallel',
            parallel_config={high_sf: protected_layers, low_sf: 8 - protected_layers},
            protected_layers=protected_layers,
            fixed_layers=fixed_layers,
        )


def create_rate_aware_strategies() -> Dict[str, RateAwareStrategy]:
    """
    创建速率感知的策略集合
    
    =============== 策略设计说明 ===============
    各SF速率 (bps) @ BW=125kHz:
      SF7:  6836 bps -> 9 layers (阈值-7.5dB)
      SF8:  3906 bps -> 5 layers (阈值-10dB)
      SF9:  2197 bps -> 2 layers (阈值-12.5dB)
      SF10: 1221 bps -> 1 layer  (阈值-15dB)
      SF11: 671 bps
      SF12: 366 bps  -> 0 layers (阈值-20dB)
    
    =============== 如何添加新策略 ===============
    使用 make_strategy() 便捷函数:
    
    # 示例1: SF9保护前2层, SF7传剩余层
    strategies["my_uep"] = make_strategy(
        name="My-UEP",
        high_sf=9,      # 核心层SF
        low_sf=7,       # 次要层SF
        protected_layers=2,  # 前2层用高SF
    )
    
    # 示例2: 固定4层, SF10保护前2层
    strategies["fixed_uep"] = make_strategy(
        name="Fixed-UEP",
        high_sf=10, low_sf=7,
        protected_layers=2,
        fixed_layers=4,  # 固定使用4层
    )
    =============================================
    """
    strategies = {}
    
    # =========== EEP基线策略 ===========
    # EEP-SF7: 高速率，低保护
    strategies["eep_sf7"] = RateAwareStrategy(
        name="EEP-SF7",
        description="等保护SF7: 高速率(6.8kbps), 阈值-7.5dB",
        mode='single',
        single_sf=7,
    )
    
    # EEP-SF9: 中等速率和保护
    strategies["eep_sf9"] = RateAwareStrategy(
        name="EEP-SF9",
        description="等保护SF9: 中速率(2.2kbps), 阈值-12.5dB",
        mode='single',
        single_sf=9,
    )
    
    # EEP-SF10: 低速率，高保护
    strategies["eep_sf10"] = RateAwareStrategy(
        name="EEP-SF10",
        description="等保护SF10: 低速率(1.2kbps), 阈值-15dB",
        mode='single',
        single_sf=10,
    )
    
    # =========== UEP策略: 1层核心保护 ===========
    # UEP SF10||SF7: Layer0用SF10, 其余用SF7 (经典配置)
    strategies["uep_10_7_1"] = make_strategy(
        name="UEP-SF10-1",
        high_sf=10, low_sf=7,
        protected_layers=1,
        description="UEP: Layer0用SF10(-15dB), 其余SF7(-7.5dB)",
    )
    
    # =========== UEP策略: 2层核心保护 ===========
    # UEP SF9||SF7: 前2层用SF9, 其余用SF7
    strategies["uep_9_7_2"] = make_strategy(
        name="UEP-SF9-2",
        high_sf=9, low_sf=7,
        protected_layers=2,
        description="UEP: Layer0-1用SF9(-12.5dB), 其余SF7(-7.5dB)",
    )
    
    return strategies


def print_strategy_rates():
    """打印各策略的速率和可传输层数"""
    print("\n" + "="*70)
    print("传输策略速率分析")
    print("="*70)
    print(f"{'策略名称':<20} {'模式':<10} {'速率(bps)':<12} {'可传层数':<10} {'对应带宽'}")
    print("-"*70)
    
    strategies = create_rate_aware_strategies()
    for name, s in strategies.items():
        rate = s.get_total_rate_bps()
        layers = s.get_achievable_layers()
        
        # 对应的Encodec带宽
        if layers <= 2:
            bw = "1.5 kbps"
        elif layers <= 4:
            bw = "3 kbps"
        else:
            bw = "6 kbps"
        
        print(f"{s.name:<20} {s.mode:<10} {rate:<12.0f} {layers:<10} {bw}")
    
    print("="*70)


###############################################################################
#                     第5部分: 端到端仿真引擎
###############################################################################

class RateAwareSimulator:
    """
    速率感知的端到端仿真引擎
    
    核心特点:
    1. 根据策略的数据速率自动决定使用多少RVQ层
    2. 支持固定带宽对比（同样层数，不同SF）
    3. 支持固定SNR对比（同样信道条件，不同策略可传不同层数）
    """
    
    # Encodec带宽与层数映射
    BANDWIDTH_TO_LAYERS = {1.5: 2, 3.0: 4, 6.0: 8, 12.0: 16, 24.0: 32}
    LAYERS_TO_BANDWIDTH = {2: 1.5, 4: 3.0, 8: 6.0, 16: 12.0, 32: 24.0}
    
    def __init__(self, 
                 encodec: EncodecWrapper,
                 channel: VLCNLOSChannel):
        self.encodec = encodec
        self.channel = channel
        self.orth_model = CSSOrthogonalityModel()
        self.strategies = create_rate_aware_strategies()
    
    def simulate_transmission(self, 
                               codes: np.ndarray,
                               strategy: RateAwareStrategy,
                               channel_snr_db: float,
                               n_layers_to_use: int = None) -> Tuple[np.ndarray, Dict]:
        """
        仿真传输过程
        
        Args:
            codes: 原始RVQ码 (total_layers, n_frames)
            strategy: 传输策略
            channel_snr_db: 信道SNR
            n_layers_to_use: 要传输的层数，如果为None则根据策略自动决定
        
        Returns:
            corrupted_codes, stats
        """
        total_layers, n_frames = codes.shape
        
        # 决定使用多少层
        if n_layers_to_use is None:
            n_layers_to_use = strategy.get_achievable_layers()
        n_layers_to_use = min(n_layers_to_use, total_layers)
        
        # 获取层-SF映射
        layer_sf_map = strategy.get_layer_sf_map(n_layers_to_use)
        
        # 初始化输出（未传输的层填充随机值）
        corrupted = np.random.randint(0, self.channel.codebook_size, codes.shape)
        layer_stats = {}
        
        # 判断是否并行传输
        is_parallel = (strategy.mode == 'parallel')
        
        for layer in range(n_layers_to_use):
            sf = layer_sf_map.get(layer, 7)
            
            # 计算有效SNR
            if is_parallel:
                # 并行传输时考虑跨SF干扰
                other_sfs = [layer_sf_map.get(l, 0) for l in range(n_layers_to_use) 
                            if l != layer]
                effective_snr = channel_snr_db
                for other_sf in other_sfs:
                    if other_sf <= 0 or other_sf == sf:
                        continue
                    isolation = abs(self.orth_model.get_isolation(sf, other_sf))
                    # 干扰带来的SNR惩罚
                    if isolation < 25:
                        effective_snr -= max(0, 25 - isolation) * 0.1
            else:
                effective_snr = channel_snr_db
            
            # 传输
            corrupted_layer, stats = self.channel.transmit_layer(
                codes[layer], sf, effective_snr
            )
            corrupted[layer] = corrupted_layer
            
            layer_stats[layer] = {
                "sf": sf,
                "transmitted": True,
                "effective_snr_db": stats["effective_snr_db"],
                "ber": stats["ber"],
                "code_error_rate": stats["code_error_rate"],
            }
        
        # 未传输的层标记
        for layer in range(n_layers_to_use, total_layers):
            layer_stats[layer] = {
                "sf": -1,
                "transmitted": False,
                "ber": 1.0,
                "code_error_rate": 1.0,
            }
        
        # 统计
        transmitted_cers = [s["code_error_rate"] for l, s in layer_stats.items() 
                          if s["transmitted"]]
        
        overall_stats = {
            "strategy": strategy.name,
            "channel_snr_db": channel_snr_db,
            "n_layers_used": n_layers_to_use,
            "layer_stats": layer_stats,
            "avg_cer": np.mean(transmitted_cers) if transmitted_cers else 1.0,
        }
        
        return corrupted, overall_stats
    
    def get_encodec_bandwidth(self, n_layers: int) -> float:
        """根据层数获取对应的Encodec带宽"""
        if n_layers <= 2:
            return 1.5
        elif n_layers <= 4:
            return 3.0
        elif n_layers <= 8:
            return 6.0
        elif n_layers <= 16:
            return 12.0
        else:
            return 24.0
    
    def run_single_experiment(self,
                               audio_path: str,
                               strategy: RateAwareStrategy,
                               channel_snr_db: float,
                               n_layers_override: int = None) -> Dict:
        """
        运行单次端到端实验
        
        重要: 策略的实际传输能力受其数据速率限制。
        例如EEP-SF10速率仅1221bps，只能传输1层；超出的层将被置零。
        
        Args:
            audio_path: 音频文件路径
            strategy: 传输策略
            channel_snr_db: 信道SNR
            n_layers_override: 编码使用的层数（用于公平对比，所有策略编码相同层数）
        
        Returns:
            实验结果字典
        """
        # 决定编码使用的层数 (所有策略相同，用于公平对比)
        if n_layers_override is not None:
            n_layers_encode = n_layers_override
        else:
            n_layers_encode = strategy.get_achievable_layers()
        
        # 策略实际能传输的层数 (受数据速率限制)
        transmittable_layers = strategy.get_achievable_layers()
        
        # 获取对应的Encodec带宽 (基于编码层数)
        encodec_bw = self.get_encodec_bandwidth(n_layers_encode)
        
        # 设置Encodec带宽并编码
        if hasattr(self.encodec, 'model') and self.encodec.model is not None:
            self.encodec.model.set_target_bandwidth(encodec_bw)
        
        codes, scale, n_samples = self.encodec.encode(audio_path)
        
        # 确保不超过实际编码的层数
        actual_layers = min(n_layers_encode, codes.shape[0])
        
        # 传输 (只传输策略能够传输的层)
        corrupted_codes, tx_stats = self.simulate_transmission(
            codes, strategy, channel_snr_db, n_layers_to_use=actual_layers
        )
        
        # 超出策略传输能力的层置为随机数 (模拟完全丢失)
        # 例如 EEP-SF10 只能传1层，第2-8层填随机值
        if transmittable_layers < actual_layers:
            # 10-bit codebook, 随机值范围 0-1023
            random_codes = np.random.randint(0, 1024, 
                size=(actual_layers - transmittable_layers, corrupted_codes.shape[1]),
                dtype=corrupted_codes.dtype)
            corrupted_codes[transmittable_layers:actual_layers, :] = random_codes
            tx_stats["layers_lost"] = actual_layers - transmittable_layers
        else:
            tx_stats["layers_lost"] = 0
        tx_stats["transmittable_layers"] = transmittable_layers
        
        # 解码
        reconstructed = self.encodec.decode(corrupted_codes, scale)
        
        # 计算质量指标
        quality = self._calculate_quality(audio_path, reconstructed, n_samples)
        
        return {
            "audio_path": audio_path,
            "strategy": strategy.name,
            "encodec_bandwidth_kbps": encodec_bw,
            "n_layers_encoded": actual_layers,
            "n_layers_transmitted": min(transmittable_layers, actual_layers),
            "n_layers_zeroed": len(tx_stats.get("layers_zeroed", [])),
            "n_frames": codes.shape[1],
            "channel_snr_db": channel_snr_db,
            "transmission": tx_stats,
            "quality": quality,
        }
    
    def _calculate_quality(self, ref_path: str, recon: np.ndarray, 
                           n_samples: int) -> Dict:
        """计算PESQ和STOI质量指标"""
        quality = {"pesq_wb": None, "stoi": None}
        
        try:
            import torchaudio
            ref_wav, sr = torchaudio.load(ref_path)
            if sr != 16000:
                ref_wav = torchaudio.transforms.Resample(sr, 16000)(ref_wav)
            ref_16k = ref_wav.squeeze().numpy()
            
            import librosa
            recon_16k = librosa.resample(recon, orig_sr=24000, target_sr=16000)
            
            min_len = min(len(ref_16k), len(recon_16k))
            ref_16k = ref_16k[:min_len]
            recon_16k = recon_16k[:min_len]
            
            try:
                from pesq import pesq
                quality["pesq_wb"] = pesq(16000, ref_16k, recon_16k, 'wb')
            except:
                pass
            
            try:
                from pystoi import stoi
                quality["stoi"] = stoi(ref_16k, recon_16k, 16000, extended=False)
            except:
                pass
                
        except Exception as e:
            print(f"Quality calculation error: {e}")
        
        return quality
    
    def run_all_strategies_comparison(self,
                                       audio_paths: List[str],
                                       snr_range: List[float] = None,
                                       strategy_keys: List[str] = None,
                                       fixed_layers: int = 8) -> Dict:
        """
        运行所有策略的端到端对比实验
        
        Args:
            audio_paths: 音频文件列表
            snr_range: SNR范围
            strategy_keys: 要测试的策略键列表，None表示全部
            fixed_layers: 固定使用的RVQ层数 (便于公平比较)
        
        Returns:
            包含所有策略结果的字典
        """
        if snr_range is None:
            snr_range = list(range(-10, 11, 1))
        
        if strategy_keys is None:
            strategy_keys = list(self.strategies.keys())
        
        print(f"\n运行所有策略对比: {len(strategy_keys)} 个策略, {len(snr_range)} 个SNR点")
        print(f"音频文件数: {len(audio_paths)}, 固定层数: {fixed_layers}")
        
        results = {
            "config": {
                "strategies": strategy_keys,
                "snr_range": snr_range,
                "fixed_layers": fixed_layers,
                "n_audio_files": len(audio_paths),
            },
            "data": {},
        }
        
        total_experiments = len(strategy_keys) * len(snr_range) * len(audio_paths)
        completed = 0
        
        for strategy_key in strategy_keys:
            if strategy_key not in self.strategies:
                print(f"  跳过未知策略: {strategy_key}")
                continue
            
            strategy = self.strategies[strategy_key]
            results["data"][strategy_key] = {}
            
            print(f"\n  测试策略: {strategy.name}")
            
            for snr in snr_range:
                trial_results = []
                
                for audio_path in audio_paths:
                    try:
                        r = self.run_single_experiment(
                            audio_path, strategy, snr, 
                            n_layers_override=fixed_layers
                        )
                        trial_results.append(r)
                        completed += 1
                    except Exception as e:
                        print(f"    Error at SNR={snr}: {e}")
                        completed += 1
                
                # 聚合结果
                if trial_results:
                    pesq_vals = [r["quality"]["pesq_wb"] for r in trial_results 
                                if r["quality"]["pesq_wb"] is not None]
                    stoi_vals = [r["quality"]["stoi"] for r in trial_results 
                                if r["quality"]["stoi"] is not None]
                    cer_vals = [r["transmission"]["avg_cer"] for r in trial_results]
                    
                    results["data"][strategy_key][snr] = {
                        "pesq_mean": np.mean(pesq_vals) if pesq_vals else None,
                        "pesq_std": np.std(pesq_vals) if pesq_vals else None,
                        "stoi_mean": np.mean(stoi_vals) if stoi_vals else None,
                        "stoi_std": np.std(stoi_vals) if stoi_vals else None,
                        "cer_mean": np.mean(cer_vals),
                        "cer_std": np.std(cer_vals),
                    }
            
            # 打印该策略的摘要
            avg_pesq = np.mean([
                results["data"][strategy_key].get(snr, {}).get("pesq_mean", 0) or 0 
                for snr in snr_range
            ])
            print(f"    完成: 平均PESQ={avg_pesq:.2f}")
        
        print(f"\n完成 {completed}/{total_experiments} 次实验")
        return results


###############################################################################
#                     第6部分: 图表生成模块（独立函数）
###############################################################################

def plot_snr_ber_curves(output_path: str = "fig1_snr_ber_curves.png",
                         sf_list: List[int] = None):
    """
    图1: CSS调制理论SNR-BER曲线
    
    展示不同SF下的瀑布曲线特性，用于论文第2章理论分析
    
    Args:
        output_path: 输出文件路径
        sf_list: 要绘制的SF列表，默认[7,8,9,10,11,12]
    """
    if sf_list is None:
        sf_list = [7, 8, 9, 10, 11, 12]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sf_list)))
    
    # 绘制每个SF的曲线
    for sf, color in zip(sf_list, colors):
        snr_range, ber_values = TheoreticalBERCalculator.generate_snr_ber_curve(sf)
        threshold = TheoreticalBERCalculator.SNR_THRESHOLDS[sf]
        
        ax.semilogy(snr_range, ber_values, 
                   color=color, linewidth=2.5,
                   label=f"SF{sf} (threshold: {threshold}dB)")
        
        # 标记阈值线
        ax.axvline(threshold, color=color, linestyle='--', alpha=0.3)
    
    # 图形设置
    ax.set_xlabel("Es/N0 (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_title("CSS Modulation: Theoretical SNR-BER Curves\n(Non-coherent Detection)", fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e-6, 1])
    ax.set_xlim([-30, 10])
    
    # 添加注释
    ax.annotate("Processing Gain: ~2.5dB per SF increase", 
               xy=(-25, 1e-5), fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sf_orthogonality(output_path: str = "fig2_sf_orthogonality.png"):
    """
    图2: SF正交性矩阵热力图
    
    展示不同SF组合的隔离度，用于论证并行传输可行性
    
    Args:
        output_path: 输出文件路径
    """
    sf_list = list(range(7, 13))
    matrix = np.zeros((6, 6))
    
    for i, sf1 in enumerate(sf_list):
        for j, sf2 in enumerate(sf_list):
            matrix[i, j] = CSSOrthogonalityModel.ORTHOGONALITY_MATRIX[sf1][sf2]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-40, vmax=0)
    
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([f"SF{sf}" for sf in sf_list])
    ax.set_yticklabels([f"SF{sf}" for sf in sf_list])
    ax.set_xlabel("Aggressor SF", fontsize=12)
    ax.set_ylabel("Victim SF", fontsize=12)
    ax.set_title("CSS SF Orthogonality Matrix (Isolation in dB)", fontsize=14)
    
    # 添加数值标注
    for i in range(6):
        for j in range(6):
            text_color = "white" if matrix[i, j] < -20 else "black"
            ax.text(j, i, f"{matrix[i, j]:.0f}",
                   ha="center", va="center", fontsize=9, color=text_color)
    
    plt.colorbar(im, ax=ax, label="Isolation (dB)")
    
    # 高亮推荐组合
    ax.add_patch(plt.Rectangle((0-0.5, 3-0.5), 1, 1, 
                                fill=False, edgecolor='red', linewidth=3))
    ax.annotate("SF10||SF7: -30/-22dB\n(Recommended)", 
               xy=(0.5, 3.5), xytext=(2, 4.5),
               fontsize=9, color='red',
               arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bandwidth_analysis(output_path: str = "fig3_bandwidth_analysis.png"):
    """
    图3: 带宽与传输容量分析
    
    对比各SF数据速率和并行传输优势
    
    Args:
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：各SF数据速率
    ax = axes[0]
    sf_list = list(range(7, 13))
    rates = [CSSOrthogonalityModel.get_data_rate(sf) for sf in sf_list]
    
    bars = ax.bar(sf_list, rates, color='steelblue', alpha=0.8, edgecolor='navy')
    ax.set_xlabel("Spreading Factor (SF)", fontsize=11)
    ax.set_ylabel("Data Rate (bps)", fontsize=11)
    ax.set_title("CSS Data Rate vs SF (BW=125kHz)", fontsize=12)
    ax.set_xticks(sf_list)
    
    for sf, r, bar in zip(sf_list, rates, bars):
        ax.text(sf, r + 100, f"{r:.0f}", ha='center', fontsize=9)
    
    # 右图：并行传输容量对比
    ax = axes[1]
    scenarios = {
        "SF9\n(single)": CSSOrthogonalityModel.get_data_rate(9),
        "SF10\n(single)": CSSOrthogonalityModel.get_data_rate(10),
        "SF10||SF7\n(parallel)": CSSOrthogonalityModel.get_data_rate(10) + CSSOrthogonalityModel.get_data_rate(7),
        "SF12||SF7\n(parallel)": CSSOrthogonalityModel.get_data_rate(12) + CSSOrthogonalityModel.get_data_rate(7),
    }
    
    names = list(scenarios.keys())
    values = list(scenarios.values())
    colors = ['gray', 'gray', '#d62728', '#9467bd']
    
    bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Total Capacity (bps)", fontsize=11)
    ax.set_title("Parallel SF Transmission Advantage", fontsize=12)
    
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 100,
               f"{v:.0f}", ha='center', fontsize=9)
    
    # 添加speedup标注
    baseline = CSSOrthogonalityModel.get_data_rate(9)
    for i, (name, v) in enumerate(scenarios.items()):
        if "parallel" in name:
            speedup = v / baseline
            ax.annotate(f"{speedup:.1f}x", xy=(i, v + 300),
                       fontsize=10, ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_uep_comparison(results: Dict, 
                        output_path: str = "fig4_uep_comparison.png",
                        bandwidth: float = 6.0):
    """
    图4: UEP策略对比图
    
    四子图：PESQ、STOI、各层CER、性能增益
    
    Args:
        results: 仿真结果字典
        output_path: 输出文件路径
        bandwidth: 要展示的带宽
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Semantic-aware UEP Strategy Comparison (Bandwidth={bandwidth}kbps)", 
                fontsize=14, fontweight='bold')
    
    data = results["data"].get(bandwidth, {})
    snr_range = results["config"]["snr_range"]
    
    # 颜色方案
    colors = {
        "eep_sf9": "#7f7f7f",
        "uep_tdm": "#1f77b4",
        "uep_parallel": "#d62728",
        "uep_aggressive": "#9467bd",
        "layer0_only": "#2ca02c",
    }
    
    # 子图1: PESQ
    ax = axes[0, 0]
    for strategy_name, strategy_data in data.items():
        pesq_vals = [strategy_data.get(snr, {}).get("pesq_mean", 0) or 0 for snr in snr_range]
        color = colors.get(strategy_name, "black")
        lw = 3 if "parallel" in strategy_name else 1.5
        ax.plot(snr_range, pesq_vals, 'o-', color=color, linewidth=lw,
               label=strategy_name, markersize=5)
    
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PESQ (WB)")
    ax.set_title("Perceptual Quality")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 子图2: STOI
    ax = axes[0, 1]
    for strategy_name, strategy_data in data.items():
        stoi_vals = [strategy_data.get(snr, {}).get("stoi_mean", 0) or 0 for snr in snr_range]
        color = colors.get(strategy_name, "black")
        lw = 3 if "parallel" in strategy_name else 1.5
        ax.plot(snr_range, stoi_vals, 's-', color=color, linewidth=lw,
               label=strategy_name, markersize=5)
    
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("STOI")
    ax.set_title("Speech Intelligibility")
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # 子图3: 各层CER (UEP-Parallel)
    ax = axes[1, 0]
    layer_colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    
    if "uep_parallel" in data:
        for layer in range(4):
            cer_vals = []
            for snr in snr_range:
                layer_cer = data["uep_parallel"].get(snr, {}).get("layer_cer", {})
                cer_vals.append(layer_cer.get(layer, 1.0))
            
            sf = "SF10" if layer == 0 else "SF7"
            ax.semilogy(snr_range, np.array(cer_vals) + 1e-10, '^-',
                       color=layer_colors[layer], linewidth=2,
                       label=f"Layer {layer} ({sf})", markersize=6)
    
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("Code Error Rate")
    ax.set_title("Per-Layer CER (UEP-Parallel)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # 子图4: 性能增益
    ax = axes[1, 1]
    ref_name = "eep_sf9"
    
    if ref_name in data:
        for strategy_name, strategy_data in data.items():
            if strategy_name == ref_name:
                continue
            gains = []
            for snr in snr_range:
                ref_pesq = data[ref_name].get(snr, {}).get("pesq_mean", 1) or 1
                cur_pesq = strategy_data.get(snr, {}).get("pesq_mean", 1) or 1
                gain = (cur_pesq - ref_pesq) / max(ref_pesq, 0.1) * 100
                gains.append(gain)
            
            color = colors.get(strategy_name, "black")
            lw = 3 if "parallel" in strategy_name else 1.5
            ax.plot(snr_range, gains, 'd-', color=color, linewidth=lw,
                   label=strategy_name, markersize=5)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PESQ Gain over EEP (%)")
    ax.set_title("UEP Performance Advantage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bandwidth_sweep(results: Dict,
                          output_path: str = "fig5_bandwidth_sweep.png",
                          strategy_name: str = "uep_parallel"):
    """
    图5: 不同带宽下的性能对比
    
    展示1.5/3/6kbps下的质量差异
    
    Args:
        results: 仿真结果字典
        output_path: 输出文件路径
        strategy_name: 要展示的策略
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Performance vs Bandwidth ({strategy_name})", fontsize=14)
    
    bandwidths = results["config"]["bandwidths"]
    snr_range = results["config"]["snr_range"]
    
    bw_colors = {1.5: '#2ca02c', 3.0: '#1f77b4', 6.0: '#d62728', 12.0: '#9467bd'}
    
    # PESQ vs SNR for different bandwidths
    ax = axes[0]
    for bw in bandwidths:
        if bw not in results["data"] or strategy_name not in results["data"][bw]:
            continue
        
        data = results["data"][bw][strategy_name]
        pesq_vals = [data.get(snr, {}).get("pesq_mean", 0) or 0 for snr in snr_range]
        
        ax.plot(snr_range, pesq_vals, 'o-', 
               color=bw_colors.get(bw, 'black'),
               linewidth=2, markersize=6,
               label=f"{bw} kbps")
    
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PESQ (WB)")
    ax.set_title("PESQ vs SNR at Different Bandwidths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # STOI vs SNR
    ax = axes[1]
    for bw in bandwidths:
        if bw not in results["data"] or strategy_name not in results["data"][bw]:
            continue
        
        data = results["data"][bw][strategy_name]
        stoi_vals = [data.get(snr, {}).get("stoi_mean", 0) or 0 for snr in snr_range]
        
        ax.plot(snr_range, stoi_vals, 's-',
               color=bw_colors.get(bw, 'black'),
               linewidth=2, markersize=6,
               label=f"{bw} kbps")
    
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("STOI")
    ax.set_title("STOI vs SNR at Different Bandwidths")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rate_bandwidth_tradeoff(output_path: str = "fig7_rate_tradeoff.png"):
    """
    图7: 速率-带宽-保护权衡分析
    
    展示各策略的速率、可传层数、以及在不同SNR下的可用性
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    strategies = create_rate_aware_strategies()
    
    # 只选择关键策略
    key_strategies = ["eep_sf7", "eep_sf9", "eep_sf10", "uep_parallel", "uep_aggressive"]
    
    # 左图: 速率对比
    ax = axes[0]
    names = []
    rates = []
    layers = []
    
    for key in key_strategies:
        if key in strategies:
            s = strategies[key]
            names.append(s.name)
            rates.append(s.get_total_rate_bps())
            layers.append(s.get_achievable_layers())
    
    colors = ['gray', 'gray', 'gray', '#d62728', '#9467bd']
    bars = ax.bar(names, rates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Data Rate (bps)", fontsize=11)
    ax.set_title("Strategy Data Rate", fontsize=12)
    ax.set_xticklabels(names, rotation=15, ha='right')
    
    for bar, r, l in zip(bars, rates, layers):
        ax.text(bar.get_x() + bar.get_width()/2, r + 100,
               f"{r:.0f}\n({l} layers)", ha='center', fontsize=8)
    
    # 中图: 可传层数
    ax = axes[1]
    bars = ax.bar(names, layers, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Max RVQ Layers", fontsize=11)
    ax.set_title("Achievable RVQ Layers", fontsize=12)
    ax.set_xticklabels(names, rotation=15, ha='right')
    
    # 添加带宽标注
    for bar, l in zip(bars, layers):
        bw = "1.5kbps" if l <= 2 else ("3kbps" if l <= 4 else "6kbps")
        ax.text(bar.get_x() + bar.get_width()/2, l + 0.2,
               bw, ha='center', fontsize=9)
    
    # 右图: SNR工作范围
    ax = axes[2]
    
    snr_thresholds = []
    for key in key_strategies:
        if key in strategies:
            s = strategies[key]
            if s.mode == 'single':
                th = TheoreticalBERCalculator.SNR_THRESHOLDS[s.single_sf]
            else:
                # 并行模式取最高SF的阈值（决定核心层可达性）
                high_sf = max(s.parallel_config.keys())
                th = TheoreticalBERCalculator.SNR_THRESHOLDS[high_sf]
            snr_thresholds.append(th)
    
    bars = ax.barh(names, [-t for t in snr_thresholds], color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel("Required SNR for BER=10^-5 (dB)", fontsize=11)
    ax.set_title("Noise Tolerance (lower is better)", fontsize=12)
    ax.invert_xaxis()
    
    for bar, th in zip(bars, snr_thresholds):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f"{th}dB", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_strategies_comparison(all_results: Dict,
                                    output_path: str = "fig8_all_strategies.png"):
    """
    图8: 所有策略统一对比图
    
    将所有策略放在同一图中对比端到端性能 (PESQ, STOI, CER)
    
    Args:
        all_results: 
            - run_all_strategies_comparison返回的结果 {"config": ..., "data": ...}
            - 或 {bandwidth: results} 结构的嵌套结果
        output_path: 输出文件路径
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("All Strategies End-to-End Comparison", fontsize=14, fontweight='bold')
    
    # 颜色和线型方案
    strategy_styles = {
        # EEP基线 - 灰色系
        "eep_sf7": {"color": "#7f7f7f", "marker": "o", "linestyle": "-", "label": "EEP-SF7"},
        "eep_sf9": {"color": "#bcbd22", "marker": "s", "linestyle": "-", "label": "EEP-SF9"},
        "eep_sf10": {"color": "#17becf", "marker": "^", "linestyle": "-", "label": "EEP-SF10"},
        
        # UEP 1层保护 
        "uep_10_7_1": {"color": "#d62728", "marker": "D", "linestyle": "-", "label": "UEP-SF10-1"},
        
        # UEP 2层保护
        "uep_9_7_2": {"color": "#ff7f0e", "marker": "*", "linestyle": "-", "label": "UEP-SF9-2", "lw": 3, "ms": 10},
        
        # # 固定带宽对比
        # "fixed_3kbps_sf9": {"color": "#8c564b", "marker": "v", "linestyle": "--", "label": "3kbps-SF9"},
        # "fixed_3kbps_uep": {"color": "#e377c2", "marker": "<", "linestyle": "--", "label": "3kbps-UEP"},
        # "fixed_6kbps_sf7": {"color": "#2ca02c", "marker": ">", "linestyle": "--", "label": "6kbps-SF7"},
        # "fixed_6kbps_uep": {"color": "#d62728", "marker": "X", "linestyle": "--", "label": "6kbps-UEP"},
    }
    
    # 解析结果格式
    all_data = {}
    snr_range = None
    
    # 格式1: 直接来自 run_all_strategies_comparison
    if "config" in all_results and "data" in all_results:
        snr_range = all_results["config"]["snr_range"]
        all_data = all_results["data"]
    else:
        # 格式2: {bandwidth: results} 嵌套结构
        for bw, bw_results in all_results.items():
            if isinstance(bw_results, dict) and "config" in bw_results and "data" in bw_results:
                if snr_range is None:
                    snr_range = bw_results["config"]["snr_range"]
                for strat_key, strat_data in bw_results["data"].items():
                    if strat_key not in all_data:
                        all_data[strat_key] = strat_data
    
    if snr_range is None or not all_data:
        print("Warning: No data to plot")
        return
    
    # Panel 1: PESQ vs SNR
    ax = axes[0]
    for strategy_key, strategy_data in all_data.items():
        style = strategy_styles.get(strategy_key, {
            "color": "black", "marker": "o", "linestyle": "-", "label": strategy_key
        })
        
        pesq_vals = [strategy_data.get(snr, {}).get("pesq_mean", 0) or 0 for snr in snr_range]
        
        lw = style.get("lw", 2)
        ms = style.get("ms", 6)
        ax.plot(snr_range, pesq_vals, 
               marker=style["marker"],
               color=style["color"],
               linestyle=style["linestyle"],
               linewidth=lw,
               markersize=ms,
               label=style["label"])
    
    ax.set_xlabel("Channel SNR (dB)", fontsize=11)
    ax.set_ylabel("PESQ (WB)", fontsize=11)
    ax.set_title("Perceptual Quality - PESQ (Higher is Better)")
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(snr_range), max(snr_range)])
    
    # Panel 2: STOI vs SNR
    ax = axes[1]
    for strategy_key, strategy_data in all_data.items():
        style = strategy_styles.get(strategy_key, {
            "color": "black", "marker": "o", "linestyle": "-", "label": strategy_key
        })
        
        stoi_vals = [strategy_data.get(snr, {}).get("stoi_mean", 0) or 0 for snr in snr_range]
        
        lw = style.get("lw", 2)
        ms = style.get("ms", 6)
        ax.plot(snr_range, stoi_vals, 
               marker=style["marker"],
               color=style["color"],
               linestyle=style["linestyle"],
               linewidth=lw,
               markersize=ms,
               label=style["label"])
    
    ax.set_xlabel("Channel SNR (dB)", fontsize=11)
    ax.set_ylabel("STOI", fontsize=11)
    ax.set_title("Intelligibility - STOI (Higher is Better)")
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(snr_range), max(snr_range)])
    ax.set_ylim([0, 1.05])
    
    # Panel 3: CER vs SNR (log scale)
    ax = axes[2]
    for strategy_key, strategy_data in all_data.items():
        style = strategy_styles.get(strategy_key, {
            "color": "black", "marker": "o", "linestyle": "-", "label": strategy_key
        })
        
        cer_vals = [strategy_data.get(snr, {}).get("cer_mean", 1) for snr in snr_range]
        
        lw = style.get("lw", 2)
        ms = style.get("ms", 6)
        ax.semilogy(snr_range, np.array(cer_vals) + 1e-10,
                   marker=style["marker"],
                   color=style["color"],
                   linestyle=style["linestyle"],
                   linewidth=lw,
                   markersize=ms,
                   label=style["label"])
    
    ax.set_xlabel("Channel SNR (dB)", fontsize=11)
    ax.set_ylabel("Code Error Rate (CER)", fontsize=11)
    ax.set_title("Transmission Reliability (Lower is Better)")
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([min(snr_range), max(snr_range)])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


###############################################################################
#                     第7部分: 主程序
###############################################################################

def run_full_simulation(args):
    """运行完整仿真流程"""
    print("="*70)
    print("端到端语义感知UEP传输仿真")
    print("="*70)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成理论曲线（无需模型）
    print("\n[1/5] 生成理论SNR-BER曲线...")
    plot_snr_ber_curves(str(output_dir / "fig1_snr_ber_curves.png"))
    
    print("[2/5] 生成SF正交性矩阵...")
    plot_sf_orthogonality(str(output_dir / "fig2_sf_orthogonality.png"))
    
    print("[3/5] 生成带宽分析图...")
    plot_bandwidth_analysis(str(output_dir / "fig3_bandwidth_analysis.png"))
    
    # 2. 端到端仿真（需要模型和音频）
    if args.input and not args.theory_only:
        print("\n[4/5] 加载Encodec模型...")
        encodec = EncodecWrapper(device=args.device)
        
        if args.pretrained:
            success = encodec.load_pretrained(args.bandwidth)
        elif args.checkpoint:
            success = encodec.load_checkpoint(args.checkpoint, args.bandwidth)
        else:
            print("警告: 未指定模型，跳过端到端仿真")
            success = False
        
        if success:
            # 创建信道
            channel = VLCNLOSChannel(
                reflectivity=0.8,
                rms_delay_ns=10.0,
                blockage_prob=0.0,
                base_path_loss_db=args.path_loss,
            )
            
            # 收集音频文件
            input_path = Path(args.input)
            if input_path.is_file():
                audio_paths = [str(input_path)]
            else:
                audio_paths = list(map(str, input_path.glob("*.wav")))
            
            print(f"Found {len(audio_paths)} audio file(s)")
            
            # 创建速率感知仿真器
            simulator = RateAwareSimulator(encodec, channel)
            
            # 打印策略速率分析
            print_strategy_rates()
            
            # 选择要测试的策略 (排除固定带宽策略以专注于原始UEP对比)
            main_strategies = [
                "eep_sf7", "eep_sf9", "eep_sf10",
                "uep_10_7_1", "uep_9_7_2",
            ]
            
            # 运行所有策略对比实验
            print("\n[5/5] 运行所有策略端到端仿真...")
            snr_range = list(range(-10, 11, 1))
            
            all_results = simulator.run_all_strategies_comparison(
                audio_paths=audio_paths,
                snr_range=snr_range,
                strategy_keys=main_strategies,
                fixed_layers=8,  # 8层=6kbps，公平对比
            )
            
            # 绘制速率权衡分析图
            plot_rate_bandwidth_tradeoff(str(output_dir / "fig7_rate_tradeoff.png"))
            
            # 绘制所有策略统一对比图 (PESQ, STOI, CER)
            plot_all_strategies_comparison(
                all_results,
                output_path=str(output_dir / "fig8_all_strategies.png")
            )
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"results_{timestamp}.json"
            
            def convert_numpy(obj):
                if isinstance(obj, (np.floating, np.integer)):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(k): convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj
            
            with open(results_file, 'w') as f:
                json.dump(convert_numpy(all_results), f, indent=2)
            print(f"\nResults saved: {results_file}")
    else:
        print("\n跳过端到端仿真 (使用 --theory-only 或未指定输入)")
    
    print("\n" + "="*70)
    print("仿真完成!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="端到端语义感知UEP传输仿真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 仅生成理论曲线
  python uep_simulation_v2.py --theory-only -o ./figures/
  
  # 使用预训练模型进行端到端仿真
  python uep_simulation_v2.py --pretrained -i ./audio/ -o ./results/
  
  # 使用自定义checkpoint，多带宽对比
  python uep_simulation_v2.py -c checkpoint.pt -i ./audio/ --multi-bandwidth
        """
    )
    
    # 模型参数
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="自定义checkpoint路径")
    parser.add_argument("--pretrained", action="store_true",
                        help="使用预训练模型")
    parser.add_argument("-b", "--bandwidth", type=float, default=6.0,
                        help="目标带宽 (kbps)")
    
    # 输入输出
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="输入音频文件或目录")
    parser.add_argument("-o", "--output", type=str, default="./uep_results/",
                        help="输出目录")
    
    # 信道参数
    parser.add_argument("--path-loss", type=float, default=20.0,
                        help="NLOS基础路径损耗 (dB)")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (cpu/cuda)")
    parser.add_argument("--theory-only", action="store_true",
                        help="仅生成理论曲线")
    
    args = parser.parse_args()
    run_full_simulation(args)


if __name__ == "__main__":
    main()
