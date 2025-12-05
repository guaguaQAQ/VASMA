import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import dino.utils as utils
import itertools
import json
import numpy as np
from PIL import Image
import os.path as osp
from torch.nn import Module
from torchvision.transforms.functional import to_pil_image
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import scipy.linalg as la

# VAE模块定义
# 原始VAE模型，使用BatchNorm
class VAE(Module):
    def __init__(self, input_dim=1024, hidden_dim=512, latent_dim=256):
        super(VAE, self).__init__()
        
        # 编码器 - 确保与预训练模型结构一致
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),  # 第一层从input_dim维度到512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # 均值和方差
        self.fc_mu = nn.Linear(512, 256)
        self.fc_var = nn.Linear(512, 256)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)  # 确保输出维度与输入维度相同
        )
    
    # VAE损失函数
def vae_loss(recon_x, x, mean, log_var, target=None, clip_weights=None):
    REC = (recon_x - x).pow(2).sum(1).mean()
    KLD = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
    return (REC + 1 * KLD)

# 权重初始化函数（修复版，避免零元素张量警告）
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # 检查权重是否存在且不为空
        if hasattr(m, 'weight') and m.weight is not None and m.weight.numel() > 0:
            m.weight.data.normal_(0.0, 0.02)
        # 检查偏置是否存在且不为空
        if hasattr(m, 'bias') and m.bias is not None and m.bias.numel() > 0:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        # 检查权重是否存在且不为空
        if hasattr(m, 'weight') and m.weight is not None and m.weight.numel() > 0:
            m.weight.data.normal_(1.0, 0.02)
        # 检查偏置是否存在且不为空
        if hasattr(m, 'bias') and m.bias is not None and m.bias.numel() > 0:
            m.bias.data.fill_(0)

# 编码器
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096),  # 修改输入维度为1024，与CLIP特征维度匹配
            nn.ReLU(),
        )
        self.mean = nn.Linear(4096, 512)
        self.log_var = nn.Linear(4096, 512)
        self.apply(weights_init)
        
    def forward(self, x, a=None):
        # a 参数是可选的，在这里我们只使用 x
        x = self.net(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 4096),  # 输入是512维的潜在空间向量
            nn.LeakyReLU(0.2),
            nn.Linear(4096, 1024)  # 输出改为1024维，与CLIP特征维度匹配
        )
        self.apply(weights_init)
    
    def forward(self, x):
        # 确保输入数据类型与模型权重一致
        x = x.to(self.net[0].weight.dtype)
        out = self.net(x)
        return out
        
# 数据流形和切空间投影相关类和函数
class ManifoldProjector:
    """
    数据流形学习和切空间投影类
    """
    def __init__(self, manifold_dim=64, n_neighbors=20):
        self.manifold_dim = manifold_dim
        self.n_neighbors = n_neighbors
        self.pca = None
        self.tangent_basis = None
        self.mean_feature = None
        self.fitted = False
        
    def fit_manifold(self, features):
        """
        使用PCA和局部线性嵌入来学习数据流形
        
        Args:
            features: 输入特征张量，形状为 [N, feature_dim]
        """
        print(f"正在拟合数据流形，输入特征形状: {features.shape}")
        
        # 转换为numpy数组进行处理
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
            
        # 计算均值特征
        self.mean_feature = np.mean(features_np, axis=0)
        
        # 使用PCA进行初步降维和找到主要变化方向
        try:
            # 确保manifold_dim不超过特征维度和样本数量
            n_samples, feature_dim = features_np.shape
            effective_manifold_dim = min(self.manifold_dim, feature_dim, n_samples - 1)
            
            if effective_manifold_dim <= 0:
                print(f"❌ 流形学习失败 - 有效流形维度小于等于0")
                print(f"   - 样本数量: {n_samples}")
                print(f"   - 特征维度: {feature_dim}")
                print(f"   - 请求的流形维度: {self.manifold_dim}")
                self.fitted = False
                return
                
            print(f"使用有效流形维度: {effective_manifold_dim}")
            
            self.pca = PCA(n_components=effective_manifold_dim)
            self.pca.fit(features_np)
            
            # 获取切空间的基向量（PCA的主成分）
            self.tangent_basis = self.pca.components_  # shape: [effective_manifold_dim, feature_dim]
            self.manifold_dim = effective_manifold_dim  # 更新实际使用的维度
            
            print(f"✅ 流形学习完成，切空间维度: {self.tangent_basis.shape}")
            print(f"   - 前5个主成分的解释方差比: {self.pca.explained_variance_ratio_[:5]}")
            print(f"   - 累计解释方差比: {np.sum(self.pca.explained_variance_ratio_):.4f}")
            self.fitted = True
            
        except Exception as e:
            print(f"❌ 流形学习失败 - 详细错误信息:")
            print(f"   - 错误类型: {type(e).__name__}")
            print(f"   - 错误描述: {str(e)}")
            print(f"   - 输入特征形状: {features_np.shape}")
            print(f"   - 尝试的流形维度: {self.manifold_dim}")
            print(f"   - 系统将回退到标准高斯噪声模式")
            self.fitted = False
        
    def project_noise_to_tangent_space(self, noise_features, dalle_features=None, blend_factor=0.7):
        """
        将高斯噪声投影到数据流形的切空间
        
        Args:
            noise_features: 高斯噪声特征，形状为 [N, feature_dim]
            dalle_features: DALL-E生成的特征（可选），用于引导投影方向
            blend_factor: 噪声和DALL-E特征的混合比例
            
        Returns:
            projected_features: 投影到切空间的特征
        """
        if not self.fitted:
            print("❌ 流形投影失败 - 流形尚未拟合")
            print("   - 请先调用fit_manifold()来学习数据流形")
            print("   - 返回原始噪声特征作为回退")
            return noise_features
            
        try:
            # 确保输入是numpy数组
            if isinstance(noise_features, torch.Tensor):
                noise_np = noise_features.detach().cpu().numpy()
            else:
                noise_np = noise_features.copy()
                
            if dalle_features is not None:
                if isinstance(dalle_features, torch.Tensor):
                    dalle_np = dalle_features.detach().cpu().numpy()
                else:
                    dalle_np = dalle_features.copy()
                
                # 混合噪声和DALL-E特征
                mixed_features = blend_factor * noise_np + (1 - blend_factor) * dalle_np
            else:
                mixed_features = noise_np
                
            # 中心化特征
            centered_features = mixed_features - self.mean_feature
            
            # 投影到切空间：先投影到主成分空间，再重构
            # 步骤1: 投影到切空间（降维）
            tangent_coords = np.dot(centered_features, self.tangent_basis.T)  # [N, manifold_dim]
            
            # 步骤2: 重构回原始空间（但限制在切空间内）
            projected_centered = np.dot(tangent_coords, self.tangent_basis)  # [N, feature_dim]
            
            # 步骤3: 加回均值
            projected_features = projected_centered + self.mean_feature
            
            # 转换回pytorch张量
            if isinstance(noise_features, torch.Tensor):
                projected_features = torch.tensor(projected_features, 
                                                dtype=noise_features.dtype, 
                                                device=noise_features.device)
                
            return projected_features
            
        except Exception as e:
            print(f"❌ 流形投影过程失败 - 详细错误信息:")
            print(f"   - 错误类型: {type(e).__name__}")
            print(f"   - 错误描述: {str(e)}")
            print(f"   - 噪声特征形状: {noise_features.shape}")
            if dalle_features is not None:
                print(f"   - DALL-E特征形状: {dalle_features.shape}")
            print(f"   - 混合比例: {blend_factor}")
            print(f"   - 切空间维度: {self.tangent_basis.shape if hasattr(self, 'tangent_basis') else 'N/A'}")
            print(f"   - 返回原始噪声特征作为回退")
            return noise_features
    
    def generate_manifold_noise(self, n_samples, feature_dim, device='cuda', noise_scale=0.1):
        """
        在流形切空间中生成结构化噪声
        
        Args:
            n_samples: 要生成的样本数量
            feature_dim: 特征维度
            device: 设备
            noise_scale: 噪声缩放因子
            
        Returns:
            structured_noise: 结构化噪声特征
        """
        # 检查输入参数的有效性
        if n_samples <= 0 or feature_dim <= 0:
            print(f"❌ 流形噪声生成失败 - 参数无效:")
            print(f"   - 样本数量: {n_samples}")
            print(f"   - 特征维度: {feature_dim}")
            print(f"   - 回退到最小有效参数")
            n_samples = max(1, n_samples)
            feature_dim = max(1, feature_dim)
            
        if not self.fitted:
            print("⚠️  流形尚未拟合，使用标准高斯噪声")
            return torch.randn(n_samples, feature_dim, device=device) * noise_scale
            
        try:
            # 在切空间坐标中生成噪声
            tangent_noise = np.random.randn(n_samples, self.manifold_dim) * noise_scale
            
            # 将切空间噪声映射到原始特征空间
            structured_noise = np.dot(tangent_noise, self.tangent_basis)
            
            # 添加均值特征（如果存在）
            if self.mean_feature is not None:
                structured_noise += self.mean_feature
            
            # 转换为pytorch张量
            structured_noise = torch.tensor(structured_noise, 
                                          dtype=torch.float32, 
                                          device=device)
            
            return structured_noise
            
        except Exception as e:
            print(f"❌ 生成流形噪声失败 - 详细错误信息:")
            print(f"   - 错误类型: {type(e).__name__}")
            print(f"   - 错误描述: {str(e)}")
            print(f"   - 请求样本数: {n_samples}")
            print(f"   - 特征维度: {feature_dim}")
            print(f"   - 噪声缩放: {noise_scale}")
            print(f"   - 流形维度: {self.manifold_dim if hasattr(self, 'manifold_dim') else 'N/A'}")
            print(f"   - 切空间形状: {self.tangent_basis.shape if hasattr(self, 'tangent_basis') else 'N/A'}")
            print(f"   - 回退到标准高斯噪声")
            return torch.randn(n_samples, feature_dim, device=device) * noise_scale

def create_dalle_noise_features(dalle_features, noise_ratio=0.3, manifold_projector=None):
    """
    将DALL-E特征转换为带有结构化噪声的特征
    
    Args:
        dalle_features: DALL-E生成的特征
        noise_ratio: 噪声比例
        manifold_projector: 流形投影器
        
    Returns:
        noisy_features: 带噪声的特征
    """
    # 检查输入特征是否为空
    if dalle_features.numel() == 0:
        print("❌ DALL-E特征为空，无法生成噪声特征")
        return dalle_features
        
    device = dalle_features.device
    dtype = dalle_features.dtype
    
    # 生成高斯噪声
    gaussian_noise = torch.randn_like(dalle_features)
    
    if manifold_projector is not None and manifold_projector.fitted:
        try:
            print(f"🔄 使用流形投影生成DALL-E噪声特征...")
            # 使用流形投影器生成结构化噪声
            structured_noise = manifold_projector.project_noise_to_tangent_space(
                gaussian_noise, 
                dalle_features, 
                blend_factor=noise_ratio
            )
            print(f"✅ 流形投影成功")
            return structured_noise.to(dtype)
        except Exception as e:
            print(f"❌ DALL-E特征流形投影失败 - 详细错误信息:")
            print(f"   - 错误类型: {type(e).__name__}")
            print(f"   - 错误描述: {str(e)}")
            print(f"   - DALL-E特征形状: {dalle_features.shape}")
            print(f"   - 噪声比例: {noise_ratio}")
            print(f"   - 回退到简单加性噪声")
            # 回退到简单噪声
            noisy_features = dalle_features + gaussian_noise * noise_ratio
            return noisy_features
    else:
        print(f"⚠️  流形投影器不可用，使用简单加性噪声")
        # 简单的加性噪声
        noisy_features = dalle_features + gaussian_noise * noise_ratio
        return noisy_features

# 旧的VAE方法，不再使用
def old_vae_methods():
    pass

# 训练VAE模型函数（增强版，包含流形学习）
def train_vae(cfg, clip_model, gpt3_prompt, classnames, template, dalle_features=None, train_loader=None):
    print("\n开始训练增强版VAE模型（含流形学习）...")
    
    vae_cache_dir = os.path.join(cfg['cache_dir'], 'vae_cache')
    os.makedirs(vae_cache_dir, exist_ok=True)
    
    # 创建流形投影器
    manifold_projector = ManifoldProjector(
        manifold_dim=cfg.get('manifold_dim', 64),
        n_neighbors=cfg.get('n_neighbors', 20)
    )
    
    # 创建编码器和生成器
    netE = Encoder().cuda()
    netG = Generator().cuda()
    
    # 初始化优化器
    optimizerE = torch.optim.AdamW(netE.parameters(), lr=cfg.get('vae_lr', 0.001))
    optimizerG = torch.optim.AdamW(netG.parameters(), lr=cfg.get('vae_lr', 0.001))
    
    # 获取CLIP文本特征
    text_features_list = []
    for classname in classnames:
        # 从gpt3_prompt中获取该类别的提示词
        prompt = gpt3_prompt.get(classname, classname)
        # 确保提示词不会太长
        if isinstance(prompt, list) and len(prompt) > 0:
            # 如果是列表，只取第一个元素
            prompt = prompt[0]
        elif isinstance(prompt, str):
            # 如果是字符串，确保长度适中
            prompt = prompt.split('.')[0] if '.' in prompt else prompt
            
        # 应用模板并确保不超过CLIP上下文长度
        texts = []
        for t in template:
            formatted_text = t.format(prompt)
            # 如果文本太长，截断它
            if len(formatted_text.split()) > 60:  # 留出一些余量，CLIP限制是77个token
                formatted_text = ' '.join(formatted_text.split()[:60]) + '.'
            texts.append(formatted_text)
            
        try:
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_features_list.append(text_feature)
        except RuntimeError as e:
            print(f"处理类别'{classname}'时出错: {e}")
            # 使用更简单的提示词重试
            simple_texts = [f"a photo of a {classname}."]
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(simple_texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_features_list.append(text_feature)
            print(f"已使用简化提示词处理类别'{classname}'")
    
    # 将特征列表合并成一个张量
    text_features = torch.cat(text_features_list, dim=0)
    
    # 学习数据流形（使用文本特征、真实图片特征和DALL-E特征）
    print("学习数据流形...")
    manifold_features = text_features.clone()
    
    # 1. 提取真实训练图片的CLIP特征（改进版：确保只使用实际训练样本）
    if train_loader is not None:
        print("提取真实训练图片特征用于流形学习...")
        
        # ===== 关键改进：自适应采样策略 =====
        shots = cfg.get('shots', 0)
        
        if shots == 0:
            print("   0-shot配置：跳过真实样本提取")
            real_image_features = []
        elif shots <= 16:
            # Few-shot场景：只使用实际训练集，不重复采样
            print(f"   Few-shot模式 ({shots}-shot)：仅使用实际训练样本，避免分布偏移")
            real_image_features = []
            
            with torch.no_grad():
                for i, (images, _) in enumerate(train_loader):
                    images = images.cuda()
                    batch_features = clip_model.encode_image(images)
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)
                    real_image_features.append(batch_features)
                    # 只遍历一次训练集，不重复采样
            
            if real_image_features:
                real_features_tensor = torch.cat(real_image_features, dim=0)
                actual_samples = len(real_features_tensor)
                print(f"   ✅ 获取到 {actual_samples} 个真实训练样本特征（精确匹配训练集大小）")
                
                # 数据质量检查
                expected_samples = shots * len(classnames)
                if actual_samples != expected_samples:
                    print(f"   ⚠️  样本数量提示：实际 {actual_samples} vs 预期 {expected_samples}")
                
                # 将真实图片特征加入流形学习
                manifold_features = torch.cat([manifold_features, real_features_tensor], dim=0)
        else:
            # Many-shot场景：可以适当扩充，但要控制上限
            max_real_samples = min(cfg.get('real_image_samples', 1000), shots * len(classnames) * 3)
            print(f"   Many-shot模式 ({shots}-shot)：限制真实样本数为 {max_real_samples}")
            
            real_image_features = []
            sample_count = 0
            
            with torch.no_grad():
                for i, (images, _) in enumerate(train_loader):
                    if sample_count >= max_real_samples:
                        break
                    images = images.cuda()
                    batch_features = clip_model.encode_image(images)
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)
                    real_image_features.append(batch_features)
                    sample_count += len(batch_features)
            
            if real_image_features:
                real_features_tensor = torch.cat(real_image_features, dim=0)[:max_real_samples]
                print(f"   获取到 {len(real_features_tensor)} 个真实图片特征用于流形学习")
                manifold_features = torch.cat([manifold_features, real_features_tensor], dim=0)
    
    # 2. 如果提供了DALL-E特征，也加入流形学习
    if dalle_features is not None:
        print(f"整合DALL-E特征到流形学习中，DALL-E特征形状: {dalle_features.shape}")
        # 确保DALL-E特征与文本特征维度一致
        if dalle_features.shape[-1] == text_features.shape[-1]:
            manifold_features = torch.cat([manifold_features, dalle_features], dim=0)
        else:
            print("DALL-E特征维度不匹配，仅使用文本特征和真实图片特征进行流形学习")
    
    # ===== 详细诊断信息 =====
    print(f"\n{'='*60}")
    print(f"📊 流形学习数据源统计 (Shots: {cfg.get('shots', 0)})")
    print(f"{'='*60}")
    print(f"总特征数: {len(manifold_features)}")
    print(f"  ├─ 文本特征 (类别原型): {len(text_features)}")
    
    if train_loader is not None and 'real_features_tensor' in locals():
        real_ratio = len(real_features_tensor) / len(manifold_features) * 100
        print(f"  ├─ 真实训练样本: {len(real_features_tensor)} ({real_ratio:.1f}%)")
        if shots > 0:
            print(f"  │   └─ 期望样本数: {shots * len(classnames)} ({shots} shots × {len(classnames)} 类)")
    else:
        print(f"  ├─ 真实训练样本: 0 (未使用)")
    
    if dalle_features is not None and dalle_features.shape[-1] == text_features.shape[-1]:
        dalle_ratio = len(dalle_features) / len(manifold_features) * 100
        print(f"  └─ DALL-E特征: {len(dalle_features)} ({dalle_ratio:.1f}%)")
    else:
        print(f"  └─ DALL-E特征: 0 (未使用)")
    
    print(f"{'='*60}\n")
    
    # 拟合流形
    manifold_projector.fit_manifold(manifold_features)
    
    # ===== 改进：VAE训练使用与流形学习一致的特征 =====
    shots = cfg.get('shots', 0)
    
    if shots == 0:
        # 0-shot: 只使用文本特征训练VAE，保持纯净
        print("VAE训练策略：0-shot模式，仅使用文本特征")
        vae_training_features = text_features
    elif shots <= 16:
        # Few-shot: 混合文本和少量真实样本
        print(f"VAE训练策略：Few-shot模式 ({shots}-shot)，混合文本+真实样本")
        if 'real_features_tensor' in locals() and len(real_image_features) > 0:
            # 平衡文本和真实样本的比例
            vae_training_features = torch.cat([text_features, real_features_tensor], dim=0)
            print(f"   训练特征: {len(text_features)} 文本 + {len(real_features_tensor)} 真实样本")
        else:
            vae_training_features = text_features
            print(f"   训练特征: {len(text_features)} 文本（无真实样本）")
    else:
        # Many-shot: 使用更多样化的特征
        print(f"VAE训练策略：Many-shot模式 ({shots}-shot)，使用流形特征子集")
        # 随机采样流形特征的子集用于训练
        max_vae_samples = min(len(manifold_features), len(classnames) * 10)
        indices = torch.randperm(len(manifold_features))[:max_vae_samples]
        vae_training_features = manifold_features[indices]
        print(f"   从 {len(manifold_features)} 个流形特征中采样 {max_vae_samples} 个")
    
    # 创建标签和数据集
    labels = torch.arange(len(vae_training_features)).cuda()
    batch_size = min(16, len(vae_training_features))
    
    # 将特征张量转换为数据加载器
    vae_dataset = torch.utils.data.TensorDataset(vae_training_features, labels)
    vae_dataloader = torch.utils.data.DataLoader(
        vae_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    print(f"VAE训练数据加载器已创建: {len(vae_training_features)} 样本, batch_size={batch_size}")
    
    # 训练VAE模型
    best_loss = float('inf')
    best_state_dict_E = None
    best_state_dict_G = None
    model_path_E = os.path.join(cfg['cache_dir'], f"best_vae_encoder_{cfg['shots']}shots.pt")
    model_path_G = os.path.join(cfg['cache_dir'], f"best_vae_generator_{cfg['shots']}shots.pt")
    
    # 训练循环
    for epoch in range(cfg.get('vae_epochs', 100)):
        total_loss = 0
        batch_count = 0
        
        # 设置为训练模式
        netE.train()
        netG.train()
        
        for feat_batch, target in vae_dataloader:
            batch_count += 1
            
            # 确保输入数据类型正确
            feat_batch = feat_batch.float()
            
            # 前向传播
            optimizerE.zero_grad()
            optimizerG.zero_grad()
            
            # 编码
            mean, log_var = netE(feat_batch)
            
            # 重参数化
            std = torch.exp(0.5 * log_var)
            z = torch.randn_like(std).cuda()
            z = mean + std * z
            
            # 生成
            bias = netG(z)
            
            # 损失计算
            recon_features = bias
            loss = vae_loss(recon_features, feat_batch, mean, log_var)
            
            # 反向传播和优化
            loss.backward()
            optimizerE.step()
            optimizerG.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{cfg.get('vae_epochs', 100)}, Loss: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state_dict_E = netE.state_dict()
            best_state_dict_G = netG.state_dict()
    
    # 保存最佳模型
    if best_state_dict_E is not None:
        torch.save(best_state_dict_E, model_path_E)
        torch.save(best_state_dict_G, model_path_G)
    
    # 加载最佳模型
    netE.load_state_dict(torch.load(model_path_E))
    netG.load_state_dict(torch.load(model_path_G))
    
    # 设置为评估模式
    netE.eval()
    netG.eval()
    
    print("VAE模型训练完成!")
    return netE, netG, manifold_projector

# 使用VAE生成图像特征（增强版，包含流形投影）
def generate_vae_features(cfg, netE, netG, clip_model, gpt3_prompt, classnames, template, 
                         manifold_projector=None, n_samples=10, use_manifold_noise=True):
    print("\n使用增强版VAE生成图像特征（含流形投影）...")
    
    vae_cache_dir = os.path.join(cfg['cache_dir'], 'vae_generated')
    os.makedirs(vae_cache_dir, exist_ok=True)
    
    # 检查是否已有生成的特征
    features_path = os.path.join(vae_cache_dir, f"vae_features_{cfg['shots']}shots.pt")
    if os.path.exists(features_path) and not cfg.get('regenerate_vae', False):
        print(f"加载已有VAE生成特征: {features_path}")
        return torch.load(features_path)
    
    # 确保VAE模型处于评估模式
    netE.eval()
    netG.eval()
    
    # 设置默认的dtype以确保一致性
    default_dtype = clip_model.dtype
    
    all_features = []
    all_labels = []
    
    # 为每个类别生成n_samples个样本
    for class_idx, classname in enumerate(classnames):
        # 从gpt3_prompt中获取该类别的提示词
        prompt = gpt3_prompt.get(classname, classname)
        # 确保提示词不会太长
        if isinstance(prompt, list) and len(prompt) > 0:
            # 如果是列表，只取第一个元素
            prompt = prompt[0]
        elif isinstance(prompt, str):
            # 如果是字符串，确保长度适中
            prompt = prompt.split('.')[0] if '.' in prompt else prompt
            
        # 应用模板并确保不超过CLIP上下文长度
        texts = []
        for t in template:
            formatted_text = t.format(prompt)
            # 如果文本太长，截断它
            if len(formatted_text.split()) > 60:  # 留出一些余量，CLIP限制是77个token
                formatted_text = ' '.join(formatted_text.split()[:60]) + '.'
            texts.append(formatted_text)
            
        try:
            with torch.no_grad():
                # 获取文本特征
                text_feature = clip_model.encode_text(clip.tokenize(texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
        except RuntimeError as e:
            print(f"处理类别'{classname}'时出错: {e}")
            # 使用更简单的提示词重试
            simple_texts = [f"a photo of a {classname}."]
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(simple_texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        # 通过增强版VAE生成特征（含流形投影）
        with torch.no_grad():
            for i in range(n_samples):
                # 编码
                mean, log_var = netE(text_feature.float())
                
                # 重参数化 - 使用流形结构化噪声
                std = torch.exp(0.5 * log_var)
                
                # 在潜在空间使用标准重参数化
                standard_noise = torch.randn_like(std)
                z = mean + std * standard_noise
                
                # 生成特征
                gen_feature = netG(z)
                
                # 如果启用流形噪声且有流形投影器，对生成的特征进行后处理
                if use_manifold_noise and manifold_projector is not None and manifold_projector.fitted:
                    try:
                        # 在特征空间中生成流形结构化噪声
                        feature_noise = manifold_projector.generate_manifold_noise(
                            n_samples=1,
                            feature_dim=gen_feature.shape[-1],  # 使用特征空间维度（1024）
                            device=gen_feature.device,
                            noise_scale=cfg.get('manifold_noise_scale', 0.1)
                        )
                        
                        # 将生成的特征与流形噪声结合
                        noise_ratio = cfg.get('feature_blend_factor', 0.8)
                        enhanced_feature = noise_ratio * gen_feature + (1 - noise_ratio) * feature_noise
                        
                        # 通过流形投影进一步优化
                        final_feature = manifold_projector.project_noise_to_tangent_space(
                            enhanced_feature,
                            text_feature,
                            blend_factor=0.9  # 主要保持增强特征，少量混合原始文本特征
                        )
                        gen_feature = final_feature
                        
                    except Exception as e:
                        print(f"❌ VAE流形增强失败 - 详细错误信息:")
                        print(f"   - 错误类型: {type(e).__name__}")
                        print(f"   - 错误描述: {str(e)}")
                        print(f"   - 生成特征形状: {gen_feature.shape}")
                        print(f"   - 文本特征形状: {text_feature.shape}")
                        print(f"   - 类别: {classnames[class_idx]}")
                        print(f"   - 样本索引: {i+1}/{n_samples}")
                        print(f"   - 使用原始生成特征")
                        # 如果流形增强失败，使用原始生成的特征
                
                # 归一化
                gen_feature /= gen_feature.norm(dim=-1, keepdim=True)
                
                all_features.append(gen_feature)
                all_labels.append(torch.tensor([class_idx], device='cuda'))
    
    # 将特征和标签组合成数据集
    vae_features = torch.cat(all_features, dim=0)
    vae_labels = torch.cat(all_labels, dim=0)
    
    # 保存特征到文件
    torch.save((vae_features, vae_labels), features_path)
    
    print(f"VAE生成了 {len(vae_features)} 个特征向量，已保存到 {features_path}")
    return vae_features, vae_labels

# 构建VAE缓存模型
def build_vae_cache_model(cfg, clip_model, vae_features, vae_labels):
    print("\n构建VAE缓存模型...")
    
    # 使用one-hot编码标签，但确保维度与clip/dino缓存一致
    num_classes = 1000
    vae_cache_values = torch.zeros(len(vae_labels), num_classes).cuda().to(clip_model.dtype)
    for i, label in enumerate(vae_labels):
        label_idx = label.item() if hasattr(label, 'item') else label
        vae_cache_values[i, label_idx] = 1
    
    # 将特征转置，使其形状与clip/dino缓存一致
    # 标准的缓存键形状应该是 [num_classes, feature_dim]，而不是 [feature_dim, num_classes]
    vae_cache_keys = vae_features.to(clip_model.dtype)
    
    print(f"VAE缓存模型构建完成: 键形状 {vae_cache_keys.shape}, 值形状 {vae_cache_values.shape}")
    return vae_cache_keys, vae_cache_values


def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def run_ensemble_tip_dalle_adapter_F(cfg, 
                            clip_cache_keys, 
                            clip_cache_values, 
                            clip_test_features, 
                            dino_cache_keys, 
                            dino_cache_values, 
                            dino_test_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F,
                            vae_adapter=None,  # 使用适配器对象而不是缓存键
                            vae_cache_values=None):
    
    # 打印各缓存的形状信息，便于调试
    print(f"CLIP缓存键形状: {clip_cache_keys.shape}, 值形状: {clip_cache_values.shape}")
    print(f"DINO缓存键形状: {dino_cache_keys.shape}, 值形状: {dino_cache_values.shape}")
    if vae_adapter is not None and vae_cache_values is not None:
        print(f"VAE适配器权重形状: {vae_adapter.weight.shape}, 值形状: {vae_cache_values.shape}")
    
    # Enable the cached keys to be learnable
    clip_adapter = nn.Linear(clip_cache_keys.shape[0], clip_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    clip_adapter.weight = nn.Parameter(clip_cache_keys.t())
    dino_adapter = nn.Linear(dino_cache_keys.shape[0], dino_cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    dino_adapter.weight = nn.Parameter(dino_cache_keys.t())
    
    optimizer = torch.optim.AdamW(
        itertools.chain(dino_adapter.parameters(), clip_adapter.parameters()),
        lr=cfg['lr'], 
        eps=1e-4)
    
    # 计算总训练步数（考虑0-shot情况）
    total_steps = cfg['train_epoch'] * (
        (len(train_loader_F) if train_loader_F is not None else 0) + 
        len(dalle_train_loader_F)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        clip_adapter.train()
        dino_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        # origin image (跳过0-shot情况)
        if train_loader_F is not None:
            for i, (images, target) in enumerate(tqdm(train_loader_F)):
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    clip_image_features = clip_model.encode_image(images)
                    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                    dino_image_features = dino_model(images)
                    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)

                clip_affinity = clip_adapter(clip_image_features)
                clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                dino_affinity = dino_adapter(dino_image_features).to(dino_cache_values.dtype)
                dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                clip_logits = 100. * clip_image_features @ clip_weights

                # 融合CLIP和DINO特征
                cache_logits_list = [clip_cache_logits, dino_cache_logits]
                
                # 如果提供了VAE适配器和缓存值，也添加到融合中
                if vae_adapter is not None and vae_cache_values is not None:
                    vae_affinity = vae_adapter(clip_image_features)  # 使用专用的VAE适配器
                    vae_cache_logits = ((-1) * (beta - beta * vae_affinity)).exp() @ vae_cache_values
                    cache_logits_list.append(vae_cache_logits)
                
                cache_logits = logits_fuse(clip_logits, cache_logits_list)
                tip_logits = clip_logits + cache_logits * alpha
                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        # dalle image
        for i, (images, target) in enumerate(tqdm(dalle_train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)

            clip_affinity = clip_adapter(clip_image_features)
            clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
            dino_affinity = dino_adapter(dino_image_features).to(dino_cache_values.dtype)
            dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
            clip_logits = 100. * clip_image_features @ clip_weights

            # 融合CLIP和DINO特征
            cache_logits_list = [clip_cache_logits, dino_cache_logits]
            
            # 如果提供了VAE适配器和缓存值，也添加到融合中
            if vae_adapter is not None and vae_cache_values is not None:
                vae_affinity = vae_adapter(clip_image_features)  # 使用专用的VAE适配器
                vae_cache_logits = ((-1) * (beta - beta * vae_affinity)).exp() @ vae_cache_values
                cache_logits_list.append(vae_cache_logits)
            
            cache_logits = logits_fuse(clip_logits, cache_logits_list)
            tip_logits = clip_logits + cache_logits * alpha
            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        clip_adapter.eval()
        dino_adapter.eval()

        clip_affinity = clip_adapter(clip_test_features)
        dino_affinity = dino_adapter(dino_test_features).to(dino_cache_values.dtype)
        clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
        dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
        clip_logits = 100. * clip_test_features @ clip_weights
        
        # 融合CLIP和DINO特征
        cache_logits_list = [clip_cache_logits, dino_cache_logits]
        
        # 如果提供了VAE适配器和缓存值，也添加到测试评估中
        if vae_adapter is not None and vae_cache_values is not None:
            vae_affinity = vae_adapter(clip_test_features)  # 使用专用的VAE适配器
            vae_cache_logits = ((-1) * (beta - beta * vae_affinity)).exp() @ vae_cache_values
            cache_logits_list.append(vae_cache_logits)
        
        cache_logits = logits_fuse(clip_logits, cache_logits_list)
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** CaFo's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(clip_adapter.weight, cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
            torch.save(dino_adapter.weight, cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
    
    clip_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
    dino_adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, CaFo's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    del clip_logits, tip_logits, cache_logits, clip_cache_logits, dino_cache_logits, clip_affinity, dino_affinity 
    # Search Hyperparameters
    # _ = search_hp(cfg, affinity, clip_cache_values, clip_test_features, test_labels, clip_weights, clip_adapter=adapter)
    best_beta, best_alpha = search_ensemble_hp(cfg, clip_cache_keys, clip_cache_values, clip_test_features, dino_cache_keys, dino_cache_values, dino_test_features, test_labels, clip_weights, clip_adapter=clip_adapter, dino_adapter=dino_adapter)
    clip_affinity = clip_adapter(clip_test_features)
    dino_affinity = dino_adapter(dino_test_features).to(dino_cache_values.dtype)
    clip_cache_logits = ((-1) * (best_beta - best_beta * clip_affinity)).exp() @ clip_cache_values
    dino_cache_logits = ((-1) * (best_beta - best_beta * dino_affinity)).exp() @ dino_cache_values
    clip_logits = 100. * clip_test_features @ clip_weights
    
    # 融合CLIP和DINO特征
    cache_logits_list = [clip_cache_logits, dino_cache_logits]
    
    # 如果提供了VAE适配器和缓存值，也添加到最终评估中
    if vae_adapter is not None and vae_cache_values is not None:
        vae_affinity = vae_adapter(clip_test_features)  # 使用专用的VAE适配器
        vae_cache_logits = ((-1) * (best_beta - best_beta * vae_affinity)).exp() @ vae_cache_values
        cache_logits_list.append(vae_cache_logits)
    
    cache_logits = logits_fuse(clip_logits, cache_logits_list)
    tip_logits = clip_logits + cache_logits * best_alpha
    print("save logits!!!!!!!!!!!!!")
    torch.save(tip_logits, cfg['cache_dir'] + "/best_tip_dino_dalle_logits_" + str(cfg['shots']) + "shots.pt")

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['clip_backbone'])
    clip_model.eval()

    # DINO
    dino_model = torchvision_models.__dict__[cfg['dino_backbone']](num_classes=0)
    dino_model.fc = nn.Identity()
    dino_model.cuda()
    utils.load_pretrained_weights(dino_model, "dino/dino_resnet50_pretrain.pth", "teacher", "vit_small'", 16)
    dino_model.eval()

    # ImageNet dataset
    random.seed(1)  #####原始是2
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    # 0-shot情况下训练集为空，需要特殊处理
    if cfg['shots'] == 0:
        print("⚠️  0-shot配置：训练集为空，跳过训练数据加载器创建")
        train_loader_cache = None
        train_loader_F = None
    else:
        train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
        train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg['root_path'], cfg['dalle_shots'])
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    dalle_train_loader_cache = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    dalle_train_loader_F = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    
    with open(cfg['gpt3_prompt_file']) as f:
        gpt3_prompt = json.load(f)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, gpt3_prompt, clip_model, imagenet.template)
    
    # 获取DALL-E图像特征用于流形学习
    print("提取DALL-E图像特征用于流形学习...")
    dalle_clip_features = []
    sample_count = 0
    max_samples = cfg.get('manifold_samples', 500)  # 限制样本数量以减少计算开销
    
    for i, (images, _) in enumerate(tqdm(dalle_train_loader_cache)):
        if sample_count >= max_samples:
            break
        images = images.cuda()
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            dalle_clip_features.append(batch_features)
            sample_count += len(batch_features)
    
    if dalle_clip_features:
        dalle_features_tensor = torch.cat(dalle_clip_features, dim=0)[:max_samples]
        print(f"获取到 {len(dalle_features_tensor)} 个DALL-E特征用于流形学习")
    else:
        dalle_features_tensor = None
        print("未获取到DALL-E特征，将仅使用文本特征进行流形学习")
    
    # 训练增强版VAE模型 - 编码器、生成器和流形投影器
    netE, netG, manifold_projector = train_vae(
        cfg, clip_model, gpt3_prompt, imagenet.classnames, imagenet.template, 
        dalle_features_tensor, train_loader_cache
    )
    
    # 使用增强版VAE生成特征
    vae_features, vae_labels = generate_vae_features(cfg, netE, netG, clip_model, gpt3_prompt, 
                                                   imagenet.classnames, imagenet.template, 
                                                   manifold_projector, 
                                                   n_samples=cfg.get('vae_samples', 10),
                                                   use_manifold_noise=cfg.get('use_manifold_noise', True))
    
    # 构建VAE缓存模型
    vae_cache_keys, vae_cache_values = build_vae_cache_model(cfg, clip_model, vae_features, vae_labels)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    
    # ===== 0-Shot特殊处理：不使用真实样本缓存 =====
    if cfg['shots'] == 0:
        print("\n⚠️  检测到0-shot配置，将不使用真实样本缓存")
        # 获取类别数量
        num_classes = len(imagenet.classnames)
        
        # 创建空的缓存张量
        # CLIP RN50特征维度: 1024, DINO ResNet50特征维度: 2048
        clip_cache_keys = torch.zeros(1024, 0, dtype=torch.float16).cuda()
        clip_cache_values = torch.zeros(0, num_classes, dtype=torch.float16).cuda()
        dino_cache_keys = torch.zeros(2048, 0, dtype=torch.float16).cuda()
        dino_cache_values = torch.zeros(0, num_classes, dtype=torch.float16).cuda()
        
        print(f"   创建空缓存: CLIP keys {clip_cache_keys.shape}, values {clip_cache_values.shape}")
        print(f"              DINO keys {dino_cache_keys.shape}, values {dino_cache_values.shape}")
    else:
        # 正常的缓存加载流程（非0-shot）
        print("\nConstructing CLIP cache model.")
        clip_cache_keys, clip_cache_values = build_clip_cache_model(cfg, clip_model, train_loader_cache)
        print("\nConstructing DINO cache model.")
        dino_cache_keys, dino_cache_values = build_dino_cache_model(cfg, dino_model, train_loader_cache)

    print("\nConstructing cache model by dalle image.")
    print("\nConstructing CLIP cache model.")
    clip_dalle_cache_keys, clip_dalle_cache_values = build_clip_dalle_cache_model(cfg, clip_model, dalle_train_loader_cache)
    print("\nConstructing DINO cache model.")
    dino_dalle_cache_keys, dino_dalle_cache_values = build_dino_dalle_cache_model(cfg, dino_model, dalle_train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    print("\nLoading CLIP feature.")
    test_clip_features, test_labels = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)
    print("\nLoading DINO feature.")
    test_dino_features, test_labels = pre_DINO_load_features(cfg, "test", dino_model, test_loader)
    
    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
   
    # 创建专用VAE适配器，避免使用CLIP适配器
    # 为此，我们创建一个单独的适配器，注意输入和输出维度
    print("创建专用VAE适配器...")
    print(f"VAE缓存键原始形状: {vae_cache_keys.shape}")
    # 适配器输入维度为特征维度(1024)，输出维度为样本数量
    vae_adapter = nn.Linear(1024, vae_cache_values.shape[0], bias=False).cuda().to(clip_model.dtype)
    # 初始化权重，不需要转置，因为Linear层会在内部进行转置
    vae_adapter.weight.data.copy_(vae_cache_keys)
    run_ensemble_tip_dalle_adapter_F(cfg, 
                            torch.cat((clip_cache_keys, clip_dalle_cache_keys), dim=1),
                            torch.cat((clip_cache_values, clip_dalle_cache_values), dim=0), 
                            test_clip_features, 
                            torch.cat((dino_cache_keys, dino_dalle_cache_keys), dim=1), 
                            torch.cat((dino_cache_values, dino_dalle_cache_values), dim=0), 
                            test_dino_features, 
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F,
                            vae_adapter,  # 传递适配器对象而不是缓存键
                            vae_cache_values)

if __name__ == '__main__':
    main()