#!/usr/bin/env python3
"""
RVQ Layer Sensitivity Analysis Experiment
==========================================
This script analyzes the semantic importance of each RVQ layer by performing 
ablation experiments. It helps validate the hypothesis that Layer 0 contains 
the majority of semantic information.

Experiment Design:
------------------
1. Encode audio samples using the trained Encodec model
2. For each layer i (0 to N-1):
   - Zero out layer i's codes (or replace with random codes)
   - Decode and measure PESQ/STOI degradation
3. Generate sensitivity curves and contribution analysis

Author: For thesis "基于语义压缩的非视距可见光通信系统的研究与实现"
"""

import argparse
import os
import json
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("Warning: pesq not installed. PESQ metrics will be skipped.")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    print("Warning: pystoi not installed. STOI metrics will be skipped.")

from model import EncodecModel


# ======================== Configuration ========================
DEFAULT_CONFIG = {
    "sample_rate": 24000,
    "channels": 1,
    "ratios": [8, 5, 4, 2],
    "target_bandwidths": [1.5, 3.0, 6.0, 12.0, 24.0],
    "bandwidth": 6.0,  # Test bandwidth (determines number of active layers)
}


# ======================== Model Loading ========================
def load_model(checkpoint_path: str, config: dict, device: str = "cuda"):
    """Load trained Encodec model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = EncodecModel._get_model(
        target_bandwidths=config["target_bandwidths"],
        sample_rate=config["sample_rate"],
        channels=config["channels"],
        causal=True,
        model_norm='weight_norm',
        audio_normalize=False,
        segment=None,
        name='my_encodec',
        ratios=config["ratios"]
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if trained with DDP
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    
    # Set target bandwidth
    model.set_target_bandwidth(config["bandwidth"])
    
    return model


# ======================== Core Analysis Functions ========================
def encode_audio(model: EncodecModel, wav: torch.Tensor) -> tuple:
    """
    Encode audio to RVQ codes.
    
    Returns:
        codes: Tensor of shape [B, K, T] where K is number of codebooks
        scale: Normalization scale (if normalize=True)
    """
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        # For single segment, we get one frame
        codes, scale = encoded_frames[0]
    return codes, scale


def decode_with_modified_codes(model: EncodecModel, codes: torch.Tensor, 
                                scale: torch.Tensor, 
                                layer_to_modify: int = None,
                                modification: str = "zero") -> torch.Tensor:
    """
    Decode RVQ codes with optional layer modification.
    
    Args:
        model: Encodec model
        codes: Original codes [B, K, T]
        scale: Scale factor
        layer_to_modify: Which layer to modify (0-indexed). None means no modification.
        modification: Type of modification - "zero", "random", or "drop"
    
    Returns:
        Reconstructed audio waveform
    """
    modified_codes = codes.clone()
    
    if layer_to_modify is not None:
        B, K, T = codes.shape
        if modification == "zero":
            # Replace layer codes with zeros
            modified_codes[:, layer_to_modify, :] = 0
        elif modification == "random":
            # Replace with random valid codebook indices
            codebook_size = model.quantizer.bins  # Usually 1024
            modified_codes[:, layer_to_modify, :] = torch.randint(
                0, codebook_size, (B, T), device=codes.device
            )
        elif modification == "drop":
            # Only use layers before layer_to_modify (cumulative drop)
            modified_codes[:, layer_to_modify:, :] = 0
    
    with torch.no_grad():
        # Decode
        encoded_frame = (modified_codes, scale)
        reconstructed = model._decode_frame(encoded_frame)
    
    return reconstructed


def compute_layer_contribution(model: EncodecModel, codes: torch.Tensor,
                                scale: torch.Tensor) -> dict:
    """
    Compute the contribution of each layer by measuring the quantization error.
    
    This directly measures how much each layer contributes to the final embedding.
    
    Returns:
        Dictionary with layer-wise contribution statistics
    """
    # Get the encoder output (before quantization)
    # We need to access the internal RVQ structure
    codes_transposed = codes.transpose(0, 1)  # [K, B, T]
    
    contributions = {}
    total_energy = 0.0
    layer_energies = []
    
    with torch.no_grad():
        quantizer = model.quantizer.vq
        n_layers = codes_transposed.shape[0]
        
        for i in range(n_layers):
            # Decode single layer
            layer_codes = codes_transposed[i:i+1]  # [1, B, T]
            layer_quantized = quantizer.layers[i].decode(layer_codes[0])  # [B, D, T]
            
            # Compute energy (L2 norm)
            layer_energy = torch.norm(layer_quantized).item() ** 2
            layer_energies.append(layer_energy)
            total_energy += layer_energy
    
    # Normalize to get contribution ratios
    for i, energy in enumerate(layer_energies):
        contributions[f"layer_{i}"] = {
            "energy": energy,
            "contribution_ratio": energy / total_energy if total_energy > 0 else 0
        }
    
    return contributions


# ======================== Metrics Calculation ========================
def calculate_metrics(ref_wav: np.ndarray, deg_wav: np.ndarray, 
                      sample_rate: int = 16000) -> dict:
    """Calculate PESQ and STOI metrics between reference and degraded audio."""
    metrics = {}
    
    # Ensure same length
    min_len = min(len(ref_wav), len(deg_wav))
    ref_wav = ref_wav[:min_len]
    deg_wav = deg_wav[:min_len]
    
    # PESQ (requires 16kHz)
    if PESQ_AVAILABLE:
        try:
            metrics["pesq_nb"] = pesq(sample_rate, ref_wav, deg_wav, 'nb')
            metrics["pesq_wb"] = pesq(sample_rate, ref_wav, deg_wav, 'wb')
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            metrics["pesq_nb"] = None
            metrics["pesq_wb"] = None
    
    # STOI
    if STOI_AVAILABLE:
        try:
            metrics["stoi"] = stoi(ref_wav, deg_wav, sample_rate, extended=False)
        except Exception as e:
            print(f"STOI calculation failed: {e}")
            metrics["stoi"] = None
    
    # SNR
    noise = ref_wav - deg_wav
    signal_power = np.mean(ref_wav ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power > 0:
        metrics["snr"] = 10 * np.log10(signal_power / noise_power)
    else:
        metrics["snr"] = float('inf')
    
    return metrics


# ======================== Main Experiment ========================
class RVQSensitivityAnalyzer:
    """
    RVQ Layer Sensitivity Analyzer
    
    Performs ablation experiments to measure the semantic importance of each RVQ layer.
    """
    
    def __init__(self, model: EncodecModel, device: str = "cuda"):
        self.model = model
        self.device = device
        self.results = []
        
    def analyze_single_audio(self, wav_path: str, save_reconstructions: bool = False,
                              output_dir: str = None) -> dict:
        """
        Analyze RVQ sensitivity for a single audio file.
        
        Args:
            wav_path: Path to input audio file
            save_reconstructions: Whether to save reconstructed audio files
            output_dir: Directory to save reconstructed audio
        
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess audio
        wav, sr = torchaudio.load(wav_path)
        
        # Resample if needed
        target_sr = self.model.sample_rate
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
        
        # Ensure mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Add batch dimension and move to device
        wav = wav.unsqueeze(0).to(self.device)  # [1, C, T]
        
        # Encode
        codes, scale = encode_audio(self.model, wav)
        n_layers = codes.shape[1]
        
        # Get baseline reconstruction (no modification)
        baseline_recon = decode_with_modified_codes(self.model, codes, scale, None)

        # ==================== 【新增代码开始】 ====================
        # 保存对照组（Baseline）：即“满血复活”的重构音频
        # 注意：要在进入 layer 循环之前保存，避免重复保存
        if save_reconstructions and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(wav_path).stem
            
            # 命名为 _baseline.wav，方便区分
            save_path = f"{output_dir}/{filename}_baseline.wav"
            
            # 注意需要把 Tensor 从 GPU 挪到 CPU，并去掉 Batch 维度 [1, C, T] -> [C, T]
            torchaudio.save(
                save_path,
                baseline_recon.squeeze(0).cpu(), 
                target_sr
            )
            print(f"Saved baseline: {save_path}")
        # ==================== 【新增代码结束】 ====================
        
        # Convert to numpy for metrics (resample to 16kHz for PESQ)
        wav_np = wav.squeeze().cpu().numpy()
        baseline_np = baseline_recon.squeeze().cpu().numpy()
        
        # Resample to 16kHz for metrics
        if target_sr != 16000:
            import librosa
            wav_16k = librosa.resample(wav_np, orig_sr=target_sr, target_sr=16000)
            baseline_16k = librosa.resample(baseline_np, orig_sr=target_sr, target_sr=16000)
        else:
            wav_16k = wav_np
            baseline_16k = baseline_np
        
        # Baseline metrics
        baseline_metrics = calculate_metrics(wav_16k, baseline_16k, 16000)
        
        # Analyze each layer
        layer_results = []
        for layer_idx in range(n_layers):
            # Zero ablation
            zero_recon = decode_with_modified_codes(
                self.model, codes, scale, layer_idx, "zero"
            )
            zero_np = zero_recon.squeeze().cpu().numpy()
            if target_sr != 16000:
                zero_16k = librosa.resample(zero_np, orig_sr=target_sr, target_sr=16000)
            else:
                zero_16k = zero_np
            
            zero_metrics = calculate_metrics(wav_16k, zero_16k, 16000)
            
            # Random ablation
            random_recon = decode_with_modified_codes(
                self.model, codes, scale, layer_idx, "random"
            )
            random_np = random_recon.squeeze().cpu().numpy()
            if target_sr != 16000:
                random_16k = librosa.resample(random_np, orig_sr=target_sr, target_sr=16000)
            else:
                random_16k = random_np
            
            random_metrics = calculate_metrics(wav_16k, random_16k, 16000)
            
            # Calculate degradation (delta from baseline)
            layer_result = {
                "layer_index": layer_idx,
                "zero_ablation": {
                    "metrics": zero_metrics,
                    "pesq_drop": baseline_metrics.get("pesq_wb", 0) - zero_metrics.get("pesq_wb", 0) if zero_metrics.get("pesq_wb") else None,
                    "stoi_drop": baseline_metrics.get("stoi", 0) - zero_metrics.get("stoi", 0) if zero_metrics.get("stoi") else None,
                },
                "random_ablation": {
                    "metrics": random_metrics,
                    "pesq_drop": baseline_metrics.get("pesq_wb", 0) - random_metrics.get("pesq_wb", 0) if random_metrics.get("pesq_wb") else None,
                    "stoi_drop": baseline_metrics.get("stoi", 0) - random_metrics.get("stoi", 0) if random_metrics.get("stoi") else None,
                }
            }
            layer_results.append(layer_result)
            
            # Save reconstructions if requested
            if save_reconstructions and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = Path(wav_path).stem
                torchaudio.save(
                    f"{output_dir}/{filename}_layer{layer_idx}_zero.wav",
                    zero_recon.squeeze(0).cpu(), target_sr
                )
                torchaudio.save(
                    f"{output_dir}/{filename}_layer{layer_idx}_random.wav",
                    random_recon.squeeze(0).cpu(), target_sr
                )
        
        # Compute layer contributions (energy-based)
        contributions = compute_layer_contribution(self.model, codes, scale)
        
        # Cumulative drop analysis (keep only first k layers)
        cumulative_results = []
        for k in range(1, n_layers + 1):
            # Keep only first k layers
            partial_codes = codes.clone()
            partial_codes[:, k:, :] = 0
            
            partial_recon = decode_with_modified_codes(self.model, partial_codes, scale, None)
            partial_np = partial_recon.squeeze().cpu().numpy()
            if target_sr != 16000:
                partial_16k = librosa.resample(partial_np, orig_sr=target_sr, target_sr=16000)
            else:
                partial_16k = partial_np
            
            partial_metrics = calculate_metrics(wav_16k, partial_16k, 16000)
            cumulative_results.append({
                "num_layers": k,
                "metrics": partial_metrics
            })
        
        return {
            "audio_file": wav_path,
            "n_layers": n_layers,
            "baseline_metrics": baseline_metrics,
            "layer_ablation_results": layer_results,
            "layer_contributions": contributions,
            "cumulative_results": cumulative_results
        }
    
    def analyze_dataset(self, audio_paths: list, save_reconstructions: bool = False,
                         output_dir: str = None) -> dict:
        """Analyze multiple audio files and aggregate results."""
        all_results = []
        
        for path in tqdm(audio_paths, desc="Analyzing audio files"):
            try:
                result = self.analyze_single_audio(path, save_reconstructions, output_dir)
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        return {
            "individual_results": all_results,
            "aggregated": aggregated
        }
    
    def _aggregate_results(self, results: list) -> dict:
        """Aggregate results across multiple audio files."""
        if not results:
            return {}
        
        n_layers = results[0]["n_layers"]
        
        # Initialize aggregation
        agg = {
            "n_samples": len(results),
            "n_layers": n_layers,
            "baseline": {"pesq_wb": [], "stoi": [], "snr": []},
            "layer_ablation": {i: {"pesq_drop": [], "stoi_drop": []} for i in range(n_layers)},
            "cumulative": {k: {"pesq_wb": [], "stoi": []} for k in range(1, n_layers + 1)},
            "contributions": {i: [] for i in range(n_layers)}
        }
        
        for r in results:
            # Baseline
            if r["baseline_metrics"].get("pesq_wb"):
                agg["baseline"]["pesq_wb"].append(r["baseline_metrics"]["pesq_wb"])
            if r["baseline_metrics"].get("stoi"):
                agg["baseline"]["stoi"].append(r["baseline_metrics"]["stoi"])
            if r["baseline_metrics"].get("snr"):
                agg["baseline"]["snr"].append(r["baseline_metrics"]["snr"])
            
            # Layer ablation
            for layer_result in r["layer_ablation_results"]:
                idx = layer_result["layer_index"]
                if layer_result["zero_ablation"]["pesq_drop"]:
                    agg["layer_ablation"][idx]["pesq_drop"].append(
                        layer_result["zero_ablation"]["pesq_drop"]
                    )
                if layer_result["zero_ablation"]["stoi_drop"]:
                    agg["layer_ablation"][idx]["stoi_drop"].append(
                        layer_result["zero_ablation"]["stoi_drop"]
                    )
            
            # Cumulative
            for cum_result in r["cumulative_results"]:
                k = cum_result["num_layers"]
                if cum_result["metrics"].get("pesq_wb"):
                    agg["cumulative"][k]["pesq_wb"].append(cum_result["metrics"]["pesq_wb"])
                if cum_result["metrics"].get("stoi"):
                    agg["cumulative"][k]["stoi"].append(cum_result["metrics"]["stoi"])
            
            # Contributions
            for key, val in r["layer_contributions"].items():
                idx = int(key.split("_")[1])
                agg["contributions"][idx].append(val["contribution_ratio"])
        
        # Compute means and stds
        summary = {
            "baseline": {
                "pesq_wb_mean": np.mean(agg["baseline"]["pesq_wb"]) if agg["baseline"]["pesq_wb"] else None,
                "pesq_wb_std": np.std(agg["baseline"]["pesq_wb"]) if agg["baseline"]["pesq_wb"] else None,
                "stoi_mean": np.mean(agg["baseline"]["stoi"]) if agg["baseline"]["stoi"] else None,
                "stoi_std": np.std(agg["baseline"]["stoi"]) if agg["baseline"]["stoi"] else None,
            },
            "layer_sensitivity": {},
            "cumulative_quality": {},
            "layer_contributions": {}
        }
        
        for i in range(n_layers):
            summary["layer_sensitivity"][f"layer_{i}"] = {
                "pesq_drop_mean": np.mean(agg["layer_ablation"][i]["pesq_drop"]) if agg["layer_ablation"][i]["pesq_drop"] else None,
                "pesq_drop_std": np.std(agg["layer_ablation"][i]["pesq_drop"]) if agg["layer_ablation"][i]["pesq_drop"] else None,
                "stoi_drop_mean": np.mean(agg["layer_ablation"][i]["stoi_drop"]) if agg["layer_ablation"][i]["stoi_drop"] else None,
                "stoi_drop_std": np.std(agg["layer_ablation"][i]["stoi_drop"]) if agg["layer_ablation"][i]["stoi_drop"] else None,
            }
            summary["layer_contributions"][f"layer_{i}"] = {
                "mean": np.mean(agg["contributions"][i]) if agg["contributions"][i] else None,
                "std": np.std(agg["contributions"][i]) if agg["contributions"][i] else None,
            }
        
        for k in range(1, n_layers + 1):
            summary["cumulative_quality"][f"first_{k}_layers"] = {
                "pesq_wb_mean": np.mean(agg["cumulative"][k]["pesq_wb"]) if agg["cumulative"][k]["pesq_wb"] else None,
                "stoi_mean": np.mean(agg["cumulative"][k]["stoi"]) if agg["cumulative"][k]["stoi"] else None,
            }
        
        return summary


# ======================== Visualization ========================
def plot_sensitivity_results(results: dict, output_path: str = "sensitivity_analysis.png"):
    """Generate visualization plots for sensitivity analysis results."""
    summary = results["aggregated"]
    n_layers = summary.get("n_layers", len(summary["layer_sensitivity"]))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RVQ Layer Sensitivity Analysis", fontsize=14, fontweight='bold')
    
    layers = list(range(n_layers))
    
    # Plot 1: PESQ Drop by Layer (Zero Ablation)
    ax1 = axes[0, 0]
    pesq_drops = [summary["layer_sensitivity"][f"layer_{i}"]["pesq_drop_mean"] or 0 for i in layers]
    pesq_stds = [summary["layer_sensitivity"][f"layer_{i}"]["pesq_drop_std"] or 0 for i in layers]
    bars = ax1.bar(layers, pesq_drops, yerr=pesq_stds, capsize=3, color='steelblue', alpha=0.8)
    ax1.set_xlabel("RVQ Layer Index")
    ax1.set_ylabel("PESQ Drop (Higher = More Important)")
    ax1.set_title("Layer Sensitivity: PESQ Degradation when Layer is Zeroed")
    ax1.set_xticks(layers)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Highlight Layer 0
    if pesq_drops[0] > max(pesq_drops[1:]) if len(pesq_drops) > 1 else True:
        bars[0].set_color('coral')
    
    # Plot 2: STOI Drop by Layer
    ax2 = axes[0, 1]
    stoi_drops = [summary["layer_sensitivity"][f"layer_{i}"]["stoi_drop_mean"] or 0 for i in layers]
    stoi_stds = [summary["layer_sensitivity"][f"layer_{i}"]["stoi_drop_std"] or 0 for i in layers]
    bars2 = ax2.bar(layers, stoi_drops, yerr=stoi_stds, capsize=3, color='seagreen', alpha=0.8)
    ax2.set_xlabel("RVQ Layer Index")
    ax2.set_ylabel("STOI Drop (Higher = More Important)")
    ax2.set_title("Layer Sensitivity: STOI Degradation when Layer is Zeroed")
    ax2.set_xticks(layers)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    if stoi_drops[0] > max(stoi_drops[1:]) if len(stoi_drops) > 1 else True:
        bars2[0].set_color('coral')
    
    # Plot 3: Cumulative Quality (First K Layers)
    ax3 = axes[1, 0]
    cum_layers = list(range(1, n_layers + 1))
    cum_pesq = [summary["cumulative_quality"][f"first_{k}_layers"]["pesq_wb_mean"] or 0 for k in cum_layers]
    cum_stoi = [summary["cumulative_quality"][f"first_{k}_layers"]["stoi_mean"] or 0 for k in cum_layers]
    
    ax3.plot(cum_layers, cum_pesq, 'o-', color='steelblue', linewidth=2, markersize=8, label='PESQ')
    ax3.set_xlabel("Number of Layers Used (First K)")
    ax3.set_ylabel("PESQ Score", color='steelblue')
    ax3.tick_params(axis='y', labelcolor='steelblue')
    ax3.set_title("Cumulative Quality: Using First K Layers Only")
    ax3.set_xticks(cum_layers)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(cum_layers, cum_stoi, 's--', color='seagreen', linewidth=2, markersize=8, label='STOI')
    ax3_twin.set_ylabel("STOI Score", color='seagreen')
    ax3_twin.tick_params(axis='y', labelcolor='seagreen')
    
    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    # Plot 4: Layer Contribution (Energy-based)
    ax4 = axes[1, 1]
    contributions = [summary["layer_contributions"][f"layer_{i}"]["mean"] or 0 for i in layers]
    contribution_stds = [summary["layer_contributions"][f"layer_{i}"]["std"] or 0 for i in layers]
    
    colors = ['coral' if i == 0 else 'lightsteelblue' for i in layers]
    ax4.bar(layers, contributions, yerr=contribution_stds, capsize=3, color=colors, alpha=0.8)
    ax4.set_xlabel("RVQ Layer Index")
    ax4.set_ylabel("Energy Contribution Ratio")
    ax4.set_title("Layer Contribution: Proportion of Total Embedding Energy")
    ax4.set_xticks(layers)
    
    # Add percentage labels on bars
    for i, (c, h) in enumerate(zip(layers, contributions)):
        ax4.text(c, h + 0.01, f'{h*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity analysis plot saved to: {output_path}")


def print_summary_table(results: dict):
    """Print a formatted summary table of results."""
    summary = results["aggregated"]
    
    print("\n" + "="*70)
    print("RVQ LAYER SENSITIVITY ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTotal samples analyzed: {summary.get('n_samples', 'N/A')}")
    print(f"Number of RVQ layers: {summary.get('n_layers', 'N/A')}")
    
    print("\n--- Baseline Reconstruction Quality ---")
    baseline = summary.get("baseline", {})
    print(f"  PESQ (WB): {baseline.get('pesq_wb_mean', 'N/A'):.3f} ± {baseline.get('pesq_wb_std', 'N/A'):.3f}")
    print(f"  STOI:      {baseline.get('stoi_mean', 'N/A'):.3f} ± {baseline.get('stoi_std', 'N/A'):.3f}")
    
    print("\n--- Layer Sensitivity (Zero Ablation) ---")
    print(f"{'Layer':<8} {'PESQ Drop':<20} {'STOI Drop':<20} {'Contribution':<15}")
    print("-"*63)
    
    sensitivity = summary.get("layer_sensitivity", {})
    contributions = summary.get("layer_contributions", {})
    
    total_pesq_drop = sum(
        sensitivity[f"layer_{i}"]["pesq_drop_mean"] or 0 
        for i in range(summary.get("n_layers", 0))
    )
    
    for i in range(summary.get("n_layers", 0)):
        layer_key = f"layer_{i}"
        pesq_drop = sensitivity.get(layer_key, {}).get("pesq_drop_mean", 0) or 0
        pesq_std = sensitivity.get(layer_key, {}).get("pesq_drop_std", 0) or 0
        stoi_drop = sensitivity.get(layer_key, {}).get("stoi_drop_mean", 0) or 0
        stoi_std = sensitivity.get(layer_key, {}).get("stoi_drop_std", 0) or 0
        contrib = contributions.get(layer_key, {}).get("mean", 0) or 0
        
        pesq_pct = (pesq_drop / total_pesq_drop * 100) if total_pesq_drop > 0 else 0
        
        print(f"Layer {i:<3} {pesq_drop:>6.3f} ± {pesq_std:<6.3f}   {stoi_drop:>6.4f} ± {stoi_std:<6.4f}   {contrib*100:>6.1f}%")
    
    print("\n--- Key Findings ---")
    if sensitivity:
        layer_0_pesq = sensitivity.get("layer_0", {}).get("pesq_drop_mean", 0) or 0
        other_pesq_total = sum(
            sensitivity.get(f"layer_{i}", {}).get("pesq_drop_mean", 0) or 0
            for i in range(1, summary.get("n_layers", 0))
        )
        if total_pesq_drop > 0:
            layer_0_importance = (layer_0_pesq / total_pesq_drop) * 100
            print(f"  • Layer 0 accounts for {layer_0_importance:.1f}% of total PESQ sensitivity")
            print(f"  • Layer 0 PESQ drop: {layer_0_pesq:.3f} vs Others total: {other_pesq_total:.3f}")
    
    print("\n" + "="*70)


# python rvq_sensitivity_analysis.py -c ./demo/selected/bs4_cut72000_length500_epoch218_lr0.0003__20260108.pt -i ./demo/audios/GT.wav --save-audio
# python rvq_sensitivity_analysis.py --pretrained -i ./demo/audios/GT.wav --save-audio
# python rvq_sensitivity_analysis.py --pretrained -i ./demo/audios/GT.wav --save-audio -b 24
# ======================== Main Entry Point ========================
def main():
    parser = argparse.ArgumentParser(
        description="RVQ Layer Sensitivity Analysis for Encodec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Analyze a single audio file
  python rvq_sensitivity_analysis.py -c checkpoint.pt -i input.wav

  # Analyze a directory of audio files
  python rvq_sensitivity_analysis.py -c checkpoint.pt -i ./test_audio/ -o ./results/

  # Use official pretrained model
  python rvq_sensitivity_analysis.py --pretrained -i input.wav
        """
    )
    
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use official pretrained 24kHz model")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input audio file or directory")
    parser.add_argument("-o", "--output", type=str, default="./sensitivity_results/",
                        help="Output directory for results")
    parser.add_argument("-b", "--bandwidth", type=float, default=6.0,
                        help="Target bandwidth in kbps")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="Model sample rate")
    parser.add_argument("--save-audio", action="store_true",
                        help="Save reconstructed audio files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup config
    config = DEFAULT_CONFIG.copy()
    config["sample_rate"] = args.sample_rate
    config["bandwidth"] = args.bandwidth
    
    # Load model
    if args.pretrained:
        print("Loading official pretrained 24kHz model...")
        model = EncodecModel.encodec_model_24khz(pretrained=True)
        model.set_target_bandwidth(args.bandwidth)
        model.eval()
        model.to(args.device)
    elif args.checkpoint:
        model = load_model(args.checkpoint, config, args.device)
    else:
        parser.error("Either --checkpoint or --pretrained must be specified")
    
    # Collect audio files
    input_path = Path(args.input)
    if input_path.is_file():
        audio_paths = [str(input_path)]
    elif input_path.is_dir():
        audio_paths = [str(p) for p in input_path.glob("*.wav")]
        if not audio_paths:
            audio_paths = [str(p) for p in input_path.glob("**/*.wav")]
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    print(f"Found {len(audio_paths)} audio file(s) to analyze")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run analysis
    analyzer = RVQSensitivityAnalyzer(model, args.device)
    
    if len(audio_paths) == 1:
        results = analyzer.analyze_single_audio(
            audio_paths[0],
            save_reconstructions=args.save_audio,
            output_dir=str(output_dir / f"reconstructions_{timestamp}")
        )
        results = {"individual_results": [results], "aggregated": analyzer._aggregate_results([results])}
    else:
        results = analyzer.analyze_dataset(
            audio_paths,
            save_reconstructions=args.save_audio,
            output_dir=str(output_dir / f"reconstructions_{timestamp}")
        )
    
    # Save results
    results_file = output_dir / f"sensitivity_results_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    print(f"Results saved to: {results_file}")
    
    # Generate plots
    plot_path = output_dir / f"sensitivity_plot_{timestamp}.png"
    plot_sensitivity_results(results, str(plot_path))
    
    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
