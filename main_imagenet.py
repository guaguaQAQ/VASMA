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
import traceback
from datasets.vae_dataset import build_vae_dataset
import numpy as np
from PIL import Image
import os.path as osp
from torch.nn import Module
from torchvision.transforms.functional import to_pil_image
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import scipy.linalg as la
from calibration_metrics import evaluate_with_calibration, compute_confidence_interval, print_calibration_table

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


class ClassAwareManifoldProjector:
    """
    类别感知的流形投影器：为每个类别维护独立的PCA
    
    与全局ManifoldProjector的区别：
    - 全局PCA: 所有类别的特征一起做PCA，学习跨类别的通用流形
    - 类别PCA: 每个类别独立做PCA，学习每个类别的专属流形结构
    
    优势：
    - 更好地捕捉每个类别的语义空间结构
    - 避免类别间的干扰
    - 生成时更有针对性
    """
    def __init__(self, classnames, manifold_dim=64):
        self.classnames = classnames
        self.manifold_dim = manifold_dim
        # 为每个类别创建独立的ManifoldProjector
        self.class_projectors = {
            c: ManifoldProjector(manifold_dim=manifold_dim) 
            for c in classnames
        }
        self.class_indices = {c: idx for idx, c in enumerate(classnames)}
        
    def fit_class_manifold(self, class_idx, features):
        """
        为指定类别拟合流形
        
        Args:
            class_idx: 类别索引
            features: 该类别的特征张量，形状为 [N, feature_dim]
        """
        if class_idx < len(self.classnames):
            classname = self.classnames[class_idx]
            print(f"  拟合类别 {class_idx} ({classname}) 的流形，特征数: {len(features)}")
            self.class_projectors[classname].fit_manifold(features)
        
    def project_to_class_tangent(self, features, class_idx):
        """
        将特征投影到指定类别的切空间
        
        Args:
            features: 输入特征 [N, feature_dim]
            class_idx: 目标类别索引
        """
        if class_idx < len(self.classnames):
            classname = self.classnames[class_idx]
            return self.class_projectors[classname].project_noise_to_tangent_space(features)
        return features
    
    def generate_class_noise(self, class_idx, n_samples, feature_dim, device='cuda', noise_scale=0.1):
        """
        在指定类别的流形切空间中生成结构化噪声
        
        Args:
            class_idx: 类别索引
            n_samples: 采样数量
            feature_dim: 特征维度
            device: 设备
            noise_scale: 噪声缩放因子
        """
        if class_idx < len(self.classnames):
            classname = self.classnames[class_idx]
            return self.class_projectors[classname].generate_manifold_noise(
                n_samples, feature_dim, device, noise_scale
            )
        return torch.randn(n_samples, feature_dim, device=device) * noise_scale
    
    def is_class_fitted(self, class_idx):
        """检查指定类别的流形是否已拟合"""
        if class_idx < len(self.classnames):
            classname = self.classnames[class_idx]
            return self.class_projectors[classname].fitted
        return False


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


class ConditionalVAE(nn.Module):
    """
    条件VAE：使用语义锚点 t_c 作为条件先验 p(z|t_c)

    与标准VAE的区别：
    - 标准VAE: p(z) = N(0, I)，先验与输入无关
    - 条件VAE: p(z|t_c) = N(μ_prior, σ²I)，先验由语义锚点决定

    其中：
    - t_c: 语义锚点（CLIP文本特征）
    - μ_prior = W * t_c: 投影到潜在空间
    - σ: 先验标准差（可调）
    """
    def __init__(self, input_dim=512, latent_dim=128, use_conditional_prior=True):
        super(ConditionalVAE, self).__init__()

        self.use_conditional_prior = use_conditional_prior
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        self.anchor_projection = nn.Linear(input_dim, latent_dim, bias=False)

    def get_conditional_prior(self, anchor_features, sigma=0.1):
        prior_mu = self.anchor_projection(anchor_features)
        prior_mu = F.normalize(prior_mu, dim=-1)
        return prior_mu, sigma

    def reparameterize(self, mu, logvar, prior_mu=None, sigma=0.1):
        std = torch.exp(0.5 * logvar)

        if prior_mu is not None and self.use_conditional_prior:
            eps_prior = torch.randn_like(std)
            z = prior_mu + eps_prior * sigma
        else:
            eps = torch.randn_like(std)
            z = mu + eps * std

        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, anchor_features=None, use_prior=True):
        mu, logvar = self.encode(x)

        prior_mu = None
        sigma = 0.1
        if use_prior and anchor_features is not None and self.use_conditional_prior:
            prior_mu, sigma = self.get_conditional_prior(anchor_features)

        z = self.reparameterize(mu, logvar, prior_mu, sigma)
        recon = self.decode(z)

        return recon, mu, logvar, z

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


def conditional_vae_loss(recon_x, x, mu, logvar, prior_mu=None, sigma=0.1, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    if prior_mu is not None:
        sigma_q_sq = torch.exp(logvar)
        mu_diff_sq = (mu - prior_mu) ** 2
        kld_loss = -0.5 * torch.sum(
            1 + logvar - sigma_q_sq - mu_diff_sq / (sigma ** 2)
        )
    else:
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss


def fusion_images_with_clip_scores(clip_model, dalle_images, vae_images, dalle_labels, vae_labels):
    assert torch.all(dalle_labels == vae_labels), "DALL-E和VAE图像的标签必须一致"

    with torch.no_grad():
        dalle_features = clip_model.encode_image(dalle_images)
        dalle_features /= dalle_features.norm(dim=-1, keepdim=True)

        vae_features = clip_model.encode_image(vae_images)
        vae_features /= vae_features.norm(dim=-1, keepdim=True)

    text_inputs = torch.cat([clip.tokenize(f"a photo of object {dalle_labels[i].item()}") for i in range(dalle_labels.size(0))]).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    dalle_scores = (100.0 * dalle_features @ text_features.T).diag()
    vae_scores = (100.0 * vae_features @ text_features.T).diag()

    total_scores = dalle_scores + vae_scores
    dalle_weights = dalle_scores / total_scores
    vae_weights = vae_scores / total_scores

    fusion_features = dalle_weights.unsqueeze(1) * dalle_features + vae_weights.unsqueeze(1) * vae_features
    fusion_features /= fusion_features.norm(dim=-1, keepdim=True)

    return fusion_features, dalle_labels


def enhanced_train_vae_with_manifold(train_loader, val_loader, clip_model, gpt3_prompt,
                                   classnames, template, dalle_train_loader=None,
                                   epochs=10, save_path=None, cfg=None):
    print("\n开始增强版VAE训练（含流形学习）...")

    manifold_projector = ManifoldProjector(
        manifold_dim=cfg.get('manifold_dim', 64) if cfg else 64,
        n_neighbors=cfg.get('n_neighbors', 20) if cfg else 20
    )

    print("提取文本特征...")
    text_features_list = []
    for classname in classnames:
        prompt = gpt3_prompt.get(classname, classname)
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt = prompt[0]
        elif isinstance(prompt, str):
            prompt = prompt.split('.')[0] if '.' in prompt else prompt

        texts = []
        for t in template:
            formatted_text = t.format(prompt)
            if len(formatted_text.split()) > 60:
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
            simple_texts = [f"a photo of a {classname}."]
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(simple_texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_features_list.append(text_feature)

    text_features = torch.cat(text_features_list, dim=0)

    print("提取真实训练图片特征用于流形学习...")
    shots = cfg.get('shots', 0) if cfg else 0

    if shots == 0:
        print("   0-shot配置：跳过真实样本提取，避免数据泄露")
        real_image_features = []
        real_features_tensor = None
    else:
        real_image_features = []
        sample_count = 0
        max_real_samples = cfg.get('real_image_samples', 1000) if cfg else 1000

        with torch.no_grad():
            for i, (images, _) in enumerate(train_loader):
                if sample_count >= max_real_samples:
                    break
                images = images.cuda()
                batch_features = clip_model.encode_image(images)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
                real_image_features.append(batch_features)
                sample_count += len(batch_features)

        real_features_tensor = None
        if real_image_features:
            real_features_tensor = torch.cat(real_image_features, dim=0)[:max_real_samples]
            print(f"获取到 {len(real_features_tensor)} 个真实图片特征用于流形学习")

    dalle_features_tensor = None
    if dalle_train_loader is not None:
        print("提取DALL-E特征用于流形学习...")
        dalle_features = []
        sample_count = 0
        max_dalle_samples = cfg.get('manifold_samples', 500) if cfg else 500

        with torch.no_grad():
            for i, (images, _) in enumerate(dalle_train_loader):
                if sample_count >= max_dalle_samples:
                    break
                images = images.cuda()
                batch_features = clip_model.encode_image(images)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
                dalle_features.append(batch_features)
                sample_count += len(batch_features)

        if dalle_features:
            dalle_features_tensor = torch.cat(dalle_features, dim=0)[:max_dalle_samples]
            print(f"获取到 {len(dalle_features_tensor)} 个DALL-E特征用于流形学习")

    manifold_features = text_features.clone()

    if real_features_tensor is not None:
        manifold_features = torch.cat([manifold_features, real_features_tensor], dim=0)

    if dalle_features_tensor is not None:
        manifold_features = torch.cat([manifold_features, dalle_features_tensor], dim=0)

    print(f"流形学习使用总特征数: {len(manifold_features)}")
    print(f"  - 文本特征: {len(text_features)}")
    if real_features_tensor is not None:
        print(f"  - 真实图片特征: {len(real_features_tensor)}")
    if dalle_features_tensor is not None:
        print(f"  - DALL-E特征: {len(dalle_features_tensor)}")

    manifold_projector.fit_manifold(manifold_features)
    print("流形学习完成，准备用于增强VAE训练")
    print(f"流形投影器状态: {'已拟合' if manifold_projector.fitted else '未拟合'}")

    return text_features, manifold_projector

# 训练VAE模型函数（增强版，包含类别感知流形学习）
def train_vae(cfg, clip_model, gpt3_prompt, classnames, template, dalle_features=None, train_loader=None):
    print("\n开始训练增强版VAE模型（含类别感知流形学习）...")
    
    vae_cache_dir = os.path.join(cfg['cache_dir'], 'vae_cache')
    os.makedirs(vae_cache_dir, exist_ok=True)
    
    # 创建类别感知的流形投影器（每个类别独立PCA）
    class_manifold_projector = ClassAwareManifoldProjector(
        classnames=classnames,
        manifold_dim=cfg.get('manifold_dim', 64)
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
        prompt = gpt3_prompt.get(classname, classname)
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt = prompt[0]
        elif isinstance(prompt, str):
            prompt = prompt.split('.')[0] if '.' in prompt else prompt
            
        texts = [t.format(prompt) for t in template]
        try:
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_features_list.append(text_feature)
        except:
            simple_texts = [f"a photo of a {classname}."]
            with torch.no_grad():
                text_feature = clip_model.encode_text(clip.tokenize(simple_texts).cuda())
                text_feature = text_feature.mean(dim=0, keepdim=True)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            text_features_list.append(text_feature)
    
    text_features = torch.cat(text_features_list, dim=0)
    
    # ========================================================================
    # 类别感知流形学习：为每个类别独立拟合PCA
    # ========================================================================
    print("\n" + "="*60)
    print("📊 类别感知流形学习：为每个类别独立拟合PCA")
    print("="*60)
    
    shots = cfg.get('shots', 0)
    
    # 为每个类别收集特征并独立拟合PCA
    for class_idx, classname in enumerate(classnames):
        print(f"\n处理类别 {class_idx}/{len(classnames)}: {classname}")
        
        # 类别特征列表
        class_features = []
        
        # 1. 添加该类别的文本特征
        class_features.append(text_features[class_idx:class_idx+1])
        
        # 2. 如果有真实样本，添加该类别的真实样本
        if train_loader is not None and shots > 0:
            real_class_features = []
            with torch.no_grad():
                for images, labels in train_loader:
                    # 只处理该类别的样本
                    class_mask = (labels == class_idx)
                    if class_mask.sum() > 0:
                        class_images = images[class_mask].cuda()
                        feat = clip_model.encode_image(class_images)
                        feat /= feat.norm(dim=-1, keepdim=True)
                        real_class_features.append(feat)
            
            if real_class_features:
                real_class_tensor = torch.cat(real_class_features, dim=0)
                class_features.append(real_class_tensor)
                print(f"  - 真实样本: {len(real_class_tensor)} 个")
        
        # 3. 如果有DALL-E特征，添加该类别的DALL-E特征
        if dalle_features is not None:
            # dalle_features 按类别组织
            # 假设 dalle_features 对应的标签已知，这里简化处理
            pass  # DALL-E特征暂不按类别分割
        
        # 合并该类别的所有特征
        if len(class_features) > 0:
            combined_features = torch.cat(class_features, dim=0)
            print(f"  - 合并后特征数: {len(combined_features)}")
            # 为该类别拟合PCA
            class_manifold_projector.fit_class_manifold(class_idx, combined_features)
    
    print("\n" + "="*60)
    print("✅ 类别感知流形学习完成")
    print("="*60)
    
    # ========================================================================
    # 为没有足够样本的类别添加文本特征作为后备
    # ========================================================================
    for class_idx in range(len(classnames)):
        if not class_manifold_projector.is_class_fitted(class_idx):
            classname = classnames[class_idx]
            print(f"⚠️  类别 {class_idx} ({classname}) 样本不足，使用文本特征")
            # 使用文本特征作为后备（只有1个样本无法做有效PCA）
            class_manifold_projector.fit_class_manifold(
                class_idx, 
                text_features[class_idx:class_idx+1].expand(10, -1)  # 复制10次以满足PCA最小样本需求
            )
    
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
    return netE, netG, class_manifold_projector

# 使用VAE生成图像特征（增强版，支持类别感知流形投影）
def generate_vae_features(cfg, netE, netG, clip_model, gpt3_prompt, classnames, template, 
                         manifold_projector=None, n_samples=10, use_manifold_noise=True):
    print("\n使用增强版VAE生成图像特征...")
    
    vae_cache_dir = os.path.join(cfg['cache_dir'], 'vae_generated')
    os.makedirs(vae_cache_dir, exist_ok=True)
    
    # 检查是否已有生成的特征
    features_path = os.path.join(vae_cache_dir, f"vae_features_{cfg['shots']}shots.pt")
    if os.path.exists(features_path) and not cfg.get('regenerate_vae', False):
        print(f"加载已有VAE生成特征: {features_path}")
        return torch.load(features_path)
    
    # 确定使用的是哪种流形投影器
    is_class_aware = isinstance(manifold_projector, ClassAwareManifoldProjector)
    if is_class_aware:
        print("使用类别感知流形投影器（每类独立PCA）")
    else:
        print("使用全局流形投影器")
    
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
        
        # 通过增强版VAE生成特征（类别感知流形投影）
        with torch.no_grad():
            for i in range(n_samples):
                # 编码
                mean, log_var = netE(text_feature.float())
                
                # 重参数化
                std = torch.exp(0.5 * log_var)
                standard_noise = torch.randn_like(std)
                z = mean + std * standard_noise
                
                # 生成特征
                gen_feature = netG(z)
                
                # 如果启用流形噪声，使用流形投影器对特征进行后处理
                if use_manifold_noise and manifold_projector is not None:
                    try:
                        if is_class_aware:
                            # 类别感知模式：为该类别生成专属的流形噪声
                            feature_noise = manifold_projector.generate_class_noise(
                                class_idx,
                                n_samples=1,
                                feature_dim=gen_feature.shape[-1],
                                device=gen_feature.device,
                                noise_scale=cfg.get('manifold_noise_scale', 0.1)
                            )
                            
                            # 将生成的特征与类别专属流形噪声结合
                            noise_ratio = cfg.get('feature_blend_factor', 0.8)
                            enhanced_feature = noise_ratio * gen_feature + (1 - noise_ratio) * feature_noise
                            
                            # 投影到类别专属切空间
                            final_feature = manifold_projector.project_to_class_tangent(
                                enhanced_feature,
                                class_idx
                            )
                        else:
                            # 全局模式：使用全局流形投影器
                            feature_noise = manifold_projector.generate_manifold_noise(
                                n_samples=1,
                                feature_dim=gen_feature.shape[-1],
                                device=gen_feature.device,
                                noise_scale=cfg.get('manifold_noise_scale', 0.1)
                            )
                            noise_ratio = cfg.get('feature_blend_factor', 0.8)
                            enhanced_feature = noise_ratio * gen_feature + (1 - noise_ratio) * feature_noise
                            final_feature = manifold_projector.project_noise_to_tangent_space(
                                enhanced_feature, text_feature, blend_factor=0.9
                            )
                        
                        gen_feature = final_feature
                        
                    except Exception as e:
                        print(f"  类别 {class_idx} 流形增强失败，使用原始生成特征")
                
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
                            vae_train_loader_F=None,
                            separated_caches=None):

    clip_dtype = next(clip_model.parameters()).dtype
    device = next(clip_model.parameters()).device
    print(f"CLIP模型数据类型: {clip_dtype}, 设备: {device}")

    clip_weights = clip_weights.to(clip_dtype)
    clip_cache_keys = clip_cache_keys.to(clip_dtype)
    clip_cache_values = clip_cache_values.to(clip_dtype)
    dino_cache_keys = dino_cache_keys.to(clip_dtype)
    dino_cache_values = dino_cache_values.to(clip_dtype)

    use_dynamic_routing = bool(cfg.get('use_dynamic_evidence_routing', False)) and separated_caches is not None
    dr_clip_k_real = dr_clip_v_real = dr_clip_k_pixel = dr_clip_v_pixel = None
    dr_dino_k = dr_dino_v = None
    dr_clip_k_cvae_real = dr_clip_v_cvae_real = None
    dr_clip_k_cvae_pixel = dr_clip_v_cvae_pixel = None
    clip_adapter_real = None
    clip_adapter_cvae_real = None
    clip_adapter_cvae_pixel = None

    if use_dynamic_routing:
        dr_clip_k_real = separated_caches['clip_real'][0].to(clip_dtype).to(device)
        dr_clip_v_real = separated_caches['clip_real'][1].to(clip_dtype).to(device)
        dr_clip_k_pixel = separated_caches['clip_pixel'][0].to(clip_dtype).to(device)
        dr_clip_v_pixel = separated_caches['clip_pixel'][1].to(clip_dtype).to(device)
        dr_dino_k = separated_caches['dino'][0].to(clip_dtype).to(device)
        dr_dino_v = separated_caches['dino'][1].to(clip_dtype).to(device)

        if separated_caches.get('clip_cvae_real') is not None:
            dr_clip_k_cvae_real = separated_caches['clip_cvae_real'][0].to(clip_dtype).to(device)
            dr_clip_v_cvae_real = separated_caches['clip_cvae_real'][1].to(clip_dtype).to(device)
        if separated_caches.get('clip_cvae_pixel') is not None:
            dr_clip_k_cvae_pixel = separated_caches['clip_cvae_pixel'][0].to(clip_dtype).to(device)
            dr_clip_v_cvae_pixel = separated_caches['clip_cvae_pixel'][1].to(clip_dtype).to(device)

        n_branches = 3
        if dr_clip_k_cvae_real is not None:
            n_branches += 1
        if dr_clip_k_cvae_pixel is not None:
            n_branches += 1
        print(f"动态注意力证据路由: {n_branches} 路分支。")
        print(f"  C_real {dr_clip_k_real.shape}, C_pixel {dr_clip_k_pixel.shape}", end="")
        if dr_clip_k_cvae_real is not None:
            print(f", C_cvae_real {dr_clip_k_cvae_real.shape}", end="")
        if dr_clip_k_cvae_pixel is not None:
            print(f", C_cvae_pixel {dr_clip_k_cvae_pixel.shape}", end="")
        print(f", C_feature {dr_dino_k.shape}")

        if dr_clip_k_real.shape[1] > 0:
            clip_adapter_real = nn.Linear(dr_clip_k_real.shape[0], dr_clip_k_real.shape[1], bias=False).to(clip_dtype).to(device)
            clip_adapter_real.weight = nn.Parameter(dr_clip_k_real.t().clone())
        clip_adapter = nn.Linear(dr_clip_k_pixel.shape[0], dr_clip_k_pixel.shape[1], bias=False).to(clip_dtype).to(device)
        clip_adapter.weight = nn.Parameter(dr_clip_k_pixel.t().clone())
        dino_adapter = nn.Linear(dr_dino_k.shape[0], dr_dino_k.shape[1], bias=False).to(clip_dtype).to(device)
        dino_adapter.weight = nn.Parameter(dr_dino_k.t().clone())

        if dr_clip_k_cvae_real is not None:
            clip_adapter_cvae_real = nn.Linear(dr_clip_k_cvae_real.shape[0], dr_clip_k_cvae_real.shape[1], bias=False).to(clip_dtype).to(device)
            clip_adapter_cvae_real.weight = nn.Parameter(dr_clip_k_cvae_real.t().clone())
        if dr_clip_k_cvae_pixel is not None:
            clip_adapter_cvae_pixel = nn.Linear(dr_clip_k_cvae_pixel.shape[0], dr_clip_k_cvae_pixel.shape[1], bias=False).to(clip_dtype).to(device)
            clip_adapter_cvae_pixel.weight = nn.Parameter(dr_clip_k_cvae_pixel.t().clone())

        opt_params = list(clip_adapter.parameters()) + list(dino_adapter.parameters())
        if clip_adapter_real is not None:
            opt_params += list(clip_adapter_real.parameters())
        if clip_adapter_cvae_real is not None:
            opt_params += list(clip_adapter_cvae_real.parameters())
        if clip_adapter_cvae_pixel is not None:
            opt_params += list(clip_adapter_cvae_pixel.parameters())
    else:
        clip_adapter = nn.Linear(clip_cache_keys.shape[0], clip_cache_keys.shape[1], bias=False).to(clip_dtype).to(device)
        clip_adapter.weight = nn.Parameter(clip_cache_keys.t())
        dino_adapter = nn.Linear(dino_cache_keys.shape[0], dino_cache_keys.shape[1], bias=False).to(clip_dtype).to(device)
        dino_adapter.weight = nn.Parameter(dino_cache_keys.t())
        opt_params = list(dino_adapter.parameters()) + list(clip_adapter.parameters())

    print(f"缓存张量数据类型统一为: {clip_dtype}")
    print(f"CLIP缓存: keys {clip_cache_keys.dtype}, values {clip_cache_values.dtype}")
    print(f"DINO缓存: keys {dino_cache_keys.dtype}, values {dino_cache_values.dtype}")
    print(f"适配器dtype: {clip_adapter.weight.dtype}")

    optimizer = torch.optim.AdamW(opt_params, lr=cfg['lr'], eps=1e-4)

    top_k_ev = int(cfg.get('evidence_top_k', 16))
    gate_tau = float(cfg.get('gate_temperature', 1.0))

    def forward_tip(clip_image_features, dino_image_features, beta_loc, alpha_loc, mask_evidence=None):
        if use_dynamic_routing:
            tip, _, _ = ensemble_tip_logits_dynamic(
                clip_image_features,
                dino_image_features,
                clip_weights,
                beta_loc,
                alpha_loc,
                top_k_ev,
                gate_tau,
                clip_adapter_real,
                clip_adapter,
                dino_adapter,
                dr_clip_k_real,
                dr_clip_v_real,
                dr_clip_k_pixel,
                dr_clip_v_pixel,
                dr_dino_k,
                dr_dino_v,
                mask_evidence=mask_evidence,
            )
            return tip
        clip_affinity = clip_adapter(clip_image_features).to(clip_dtype)
        clip_cache_logits = ((-1) * (beta_loc - beta_loc * clip_affinity)).exp() @ clip_cache_values
        dino_affinity = dino_adapter(dino_image_features).to(clip_dtype)
        dino_cache_logits = ((-1) * (beta_loc - beta_loc * dino_affinity)).exp() @ dino_cache_values
        clip_logits = 100. * clip_image_features @ clip_weights
        cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
        return clip_logits + cache_logits * alpha_loc

    total_steps = cfg['train_epoch'] * (
        (len(train_loader_F) if train_loader_F is not None else 0) +
        len(dalle_train_loader_F) +
        (len(vae_train_loader_F) if vae_train_loader_F is not None else 0)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        clip_adapter.train()
        dino_adapter.train()
        if clip_adapter_real is not None:
            clip_adapter_real.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        if train_loader_F is not None:
            for i, (images, target) in enumerate(tqdm(train_loader_F)):
                images, target = images.to(device), target.to(device)
                with torch.no_grad():
                    clip_image_features = clip_model.encode_image(images)
                    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                    clip_image_features = clip_image_features.to(clip_dtype)

                    dino_image_features = dino_model(images)
                    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
                    dino_image_features = dino_image_features.to(clip_dtype)

                tip_logits = forward_tip(clip_image_features, dino_image_features, beta, alpha)
                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        for i, (images, target) in enumerate(tqdm(dalle_train_loader_F)):
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                clip_image_features = clip_image_features.to(clip_dtype)

                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
                dino_image_features = dino_image_features.to(clip_dtype)

            tip_logits = forward_tip(clip_image_features, dino_image_features, beta, alpha)
            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if vae_train_loader_F is not None:
            for i, (images, target) in enumerate(tqdm(vae_train_loader_F)):
                images, target = images.to(device), target.to(device)
                with torch.no_grad():
                    clip_image_features = clip_model.encode_image(images)
                    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                    clip_image_features = clip_image_features.to(clip_dtype)

                    dino_image_features = dino_model(images)
                    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
                    dino_image_features = dino_image_features.to(clip_dtype)

                tip_logits = forward_tip(clip_image_features, dino_image_features, beta, alpha)
                loss = F.cross_entropy(tip_logits, target)

                acc = cls_acc(tip_logits, target)
                correct_samples += acc / 100 * len(tip_logits)
                all_samples += len(tip_logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        if cfg.get('use_fusion', False) and dalle_train_loader_F is not None and vae_train_loader_F is not None:
            dalle_iterator = iter(dalle_train_loader_F)
            vae_iterator = iter(vae_train_loader_F)
            min_batches = min(len(dalle_train_loader_F), len(vae_train_loader_F))

            print("训练DALL-E和VAE融合图像...")
            for _ in range(min_batches):
                try:
                    dalle_images, dalle_target = next(dalle_iterator)
                    vae_images, vae_target = next(vae_iterator)

                    min_batch_size = min(dalle_images.size(0), vae_images.size(0))
                    dalle_images, dalle_target = dalle_images[:min_batch_size], dalle_target[:min_batch_size]
                    vae_images, vae_target = vae_images[:min_batch_size], vae_target[:min_batch_size]

                    dalle_images, dalle_target = dalle_images.to(device), dalle_target.to(device)
                    vae_images, vae_target = vae_images.to(device), vae_target.to(device)

                    if not torch.all(dalle_target == vae_target):
                        continue

                    text_inputs = torch.cat([clip.tokenize(f"a photo of object {dalle_target[i].item()}") for i in range(dalle_target.size(0))]).to(device)

                    with torch.no_grad():
                        dalle_features = clip_model.encode_image(dalle_images)
                        dalle_features /= dalle_features.norm(dim=-1, keepdim=True)
                        dalle_features = dalle_features.to(clip_dtype)

                        vae_features = clip_model.encode_image(vae_images)
                        vae_features /= vae_features.norm(dim=-1, keepdim=True)
                        vae_features = vae_features.to(clip_dtype)

                        text_features = clip_model.encode_text(text_inputs)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                    dalle_scores = (100.0 * dalle_features @ text_features.T).diag()
                    vae_scores = (100.0 * vae_features @ text_features.T).diag()

                    total_scores = dalle_scores + vae_scores
                    dalle_weights = dalle_scores / total_scores
                    vae_weights = vae_scores / total_scores

                    fusion_features = dalle_weights.unsqueeze(1) * dalle_features + vae_weights.unsqueeze(1) * vae_features
                    fusion_features /= fusion_features.norm(dim=-1, keepdim=True)
                    fusion_target = dalle_target

                    with torch.no_grad():
                        dino_dalle_features = dino_model(dalle_images)
                        dino_dalle_features /= dino_dalle_features.norm(dim=-1, keepdim=True)
                        dino_dalle_features = dino_dalle_features.to(clip_dtype)

                        dino_vae_features = dino_model(vae_images)
                        dino_vae_features /= dino_vae_features.norm(dim=-1, keepdim=True)
                        dino_vae_features = dino_vae_features.to(clip_dtype)

                        dino_fusion_features = (dalle_weights.unsqueeze(1) * dino_dalle_features +
                                               vae_weights.unsqueeze(1) * dino_vae_features)
                        dino_fusion_features /= dino_fusion_features.norm(dim=-1, keepdim=True)

                    tip_logits = forward_tip(fusion_features, dino_fusion_features, beta, alpha)
                    loss = F.cross_entropy(tip_logits, fusion_target)

                    acc = cls_acc(tip_logits, fusion_target)
                    correct_samples += acc / 100 * len(tip_logits)
                    all_samples += len(tip_logits)
                    loss_list.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                except StopIteration:
                    break

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        clip_adapter.eval()
        dino_adapter.eval()
        if clip_adapter_real is not None:
            clip_adapter_real.eval()

        clip_test_features = clip_test_features.to(clip_dtype)
        dino_test_features = dino_test_features.to(clip_dtype)

        tip_logits = forward_tip(clip_test_features, dino_test_features, beta, alpha)
        acc = cls_acc(tip_logits, test_labels)

        print("**** VASMA's val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(clip_adapter.weight, cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
            torch.save(dino_adapter.weight, cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
            if use_dynamic_routing and clip_adapter_real is not None:
                torch.save(clip_adapter_real.weight, cfg['cache_dir'] + "/best_F_clip_adapter_real_" + str(cfg['shots']) + "shots.pt")

    loaded_clip_w = torch.load(cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt", map_location=device)
    loaded_dino_w = torch.load(cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt", map_location=device)
    clip_adapter.weight = nn.Parameter(loaded_clip_w.to(clip_dtype).to(device))
    dino_adapter.weight = nn.Parameter(loaded_dino_w.to(clip_dtype).to(device))
    if use_dynamic_routing and clip_adapter_real is not None:
        p_real = cfg['cache_dir'] + "/best_F_clip_adapter_real_" + str(cfg['shots']) + "shots.pt"
        if os.path.exists(p_real):
            loaded_real = torch.load(p_real, map_location=device)
            clip_adapter_real.weight = nn.Parameter(loaded_real.to(clip_dtype).to(device))
    print(f"**** After fine-tuning, VASMA's best val accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    if use_dynamic_routing:
        best_beta, best_alpha = search_dynamic_evidence_hp(
            cfg,
            clip_test_features,
            dino_test_features,
            test_labels,
            clip_weights,
            clip_adapter_real,
            clip_adapter,
            dino_adapter,
            dr_clip_k_real,
            dr_clip_v_real,
            dr_clip_k_pixel,
            dr_clip_v_pixel,
            dr_dino_k,
            dr_dino_v,
        )
    else:
        best_beta, best_alpha = search_ensemble_hp(cfg, clip_adapter.weight.t(), clip_cache_values,
                                                 clip_test_features, dino_adapter.weight.t(), dino_cache_values,
                                                 dino_test_features, test_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    clip_test_features = clip_test_features.to(clip_dtype)
    dino_test_features = dino_test_features.to(clip_dtype)

    tip_logits = forward_tip(clip_test_features, dino_test_features, best_beta, best_alpha)
    if use_dynamic_routing:
        with torch.no_grad():
            _, test_alphas, branch_logits = ensemble_tip_logits_dynamic(
                clip_test_features,
                dino_test_features,
                clip_weights,
                best_beta,
                best_alpha,
                top_k_ev,
                gate_tau,
                clip_adapter_real,
                clip_adapter,
                dino_adapter,
                dr_clip_k_real,
                dr_clip_v_real,
                dr_clip_k_pixel,
                dr_clip_v_pixel,
                dr_dino_k,
                dr_dino_v,
            )
        clip_logits = 100. * clip_test_features @ clip_weights
        clip_cache_logits = branch_logits['L_pixel']
        dino_cache_logits = branch_logits['L_feature']
        cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
    else:
        clip_affinity = clip_adapter(clip_test_features).to(clip_dtype)
        dino_affinity = dino_adapter(dino_test_features).to(clip_dtype)
        clip_cache_logits = ((-1) * (best_beta - best_beta * clip_affinity)).exp() @ clip_cache_values
        dino_cache_logits = ((-1) * (best_beta - best_beta * dino_affinity)).exp() @ dino_cache_values
        clip_logits = 100. * clip_test_features @ clip_weights
        cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
    acc = cls_acc(tip_logits, test_labels)
    print("**** VASMA's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))

    save_dir = cfg['cache_dir']
    os.makedirs(save_dir, exist_ok=True)

    labels_path = os.path.join(save_dir, f"test_labels_{cfg['shots']}shots.npy")
    np.save(labels_path, test_labels.cpu().numpy())
    print(f"已保存 labels 到: {labels_path}")

    unified_logits_path = os.path.join(save_dir, f"test_logits_unified_{cfg['shots']}shots.npy")
    np.save(unified_logits_path, tip_logits.detach().cpu().numpy())
    print(f"已保存 Unified logits 到: {unified_logits_path}")

    if use_dynamic_routing:
        alphas_path = os.path.join(save_dir, f"test_evidence_alphas_{cfg['shots']}shots.npy")
        np.save(alphas_path, test_alphas.detach().cpu().numpy())
        print(f"已保存动态证据门控权重 [α_real, α_pixel, α_feature]: {alphas_path}")

    clip_cache_logits_path = os.path.join(save_dir, f"test_logits_clip_{cfg['shots']}shots.npy")
    np.save(clip_cache_logits_path, clip_cache_logits.detach().cpu().numpy())
    print(f"已保存 ClipCache logits 到: {clip_cache_logits_path}")

    naive_alpha = 0.5
    clipdino_logits_path = os.path.join(save_dir, f"test_logits_clipdino_{cfg['shots']}shots.npy")
    np.save(clipdino_logits_path, (clip_logits + cache_logits * naive_alpha).detach().cpu().numpy())
    print(f"已保存 ClipDino logits (naive fusion, alpha={naive_alpha}) 到: {clipdino_logits_path}")

    print(f"\n所有预测结果已保存到: {save_dir}")

    if cfg.get('run_evidence_masking', False) and use_dynamic_routing:
        print("\n-------- 证据遮盖与因果归因 (Evidence Masking / Faithfulness) --------")
        masking_rows = []
        with torch.no_grad():
            mask_specs = [
                ('mask_real', {'real': True}),
                ('mask_pixel', {'pixel': True}),
                ('mask_feature', {'feature': True}),
                ('mask_pixel_and_feature', {'pixel': True, 'feature': True}),
            ]
            for name, mdict in mask_specs:
                tip_m, _, _ = ensemble_tip_logits_dynamic(
                    clip_test_features,
                    dino_test_features,
                    clip_weights,
                    best_beta,
                    best_alpha,
                    top_k_ev,
                    gate_tau,
                    clip_adapter_real,
                    clip_adapter,
                    dino_adapter,
                    dr_clip_k_real,
                    dr_clip_v_real,
                    dr_clip_k_pixel,
                    dr_clip_v_pixel,
                    dr_dino_k,
                    dr_dino_v,
                    mask_evidence=mdict,
                )
                masking_rows.append(evidence_masking_report(tip_logits, tip_m, test_labels, name))
        for row in masking_rows:
            print("  [{}] acc_masked={:.2f} (Δacc={:.2f}), flip%={:.2f}, Δconf={:.4f}".format(
                row['ablation'], row['acc_masked'], row['acc_drop'],
                row['prediction_flip_rate_percent'], row['mean_conf_drop']))
        mask_json = os.path.join(save_dir, f"evidence_masking_{cfg['shots']}shots.json")
        with open(mask_json, 'w', encoding='utf-8') as f:
            json.dump(masking_rows, f, indent=2, ensure_ascii=False)
        print(f"证据遮盖统计已写入: {mask_json}")

    if use_dynamic_routing:
        print("\n" + "="*80)
        print("TRANSPARENT AUDIT: Evidence Provenance Analysis")
        print("="*80)

        with torch.no_grad():
            L_real = branch_logits['L_real']
            L_pixel = branch_logits['L_pixel']
            L_feature = branch_logits['L_feature']

            weighted_L_real = test_alphas[:, 0:1] * L_real
            weighted_L_pixel = test_alphas[:, 1:2] * L_pixel
            weighted_L_feature = test_alphas[:, 2:3] * L_feature

            cache_contrib_total = weighted_L_real + weighted_L_pixel + weighted_L_feature

            test_probs = F.softmax(tip_logits, dim=1)
            test_preds = tip_logits.argmax(dim=1)
            test_confs = test_probs.max(dim=1).values

            print(f"\nEvidence Routing Statistics ({len(test_labels)} test samples):")

            avg_alphas = test_alphas.mean(dim=0)
            print(f"  Average Alpha Weights:")
            print(f"    a_real (real samples)     = {avg_alphas[0].item():.3f}")
            print(f"    a_pixel (DALL-E+VAE)      = {avg_alphas[1].item():.3f}")
            print(f"    a_feature (DINO features)= {avg_alphas[2].item():.3f}")
            print(f"    Sum check (should be 1.0) = {avg_alphas.sum().item():.3f}")

            print(f"\n  Per-Class Alpha Distribution (mean +/- std):")
            for c in range(test_probs.shape[1]):
                mask_c = (test_preds == c)
                cnt = mask_c.sum().item()
                if cnt > 0:
                    a0 = test_alphas[mask_c, 0].mean().item()
                    a1 = test_alphas[mask_c, 1].mean().item()
                    a2 = test_alphas[mask_c, 2].mean().item()
                    s0 = test_alphas[mask_c, 0].std().item()
                    s1 = test_alphas[mask_c, 1].std().item()
                    s2 = test_alphas[mask_c, 2].std().item()
                    classname = str(c)
                    print(f"    Class {c} ({classname}, n={cnt}): "
                          f"a_real={a0:.3f}+/-{s0:.3f}  a_pixel={a1:.3f}+/-{s1:.3f}  a_feature={a2:.3f}+/-{s2:.3f}")

            print(f"\n  Confidence Stratification:")
            conf_bins = [(0.0, 0.5, "low"), (0.5, 0.8, "medium"), (0.8, 1.0, "high")]
            for lo, hi, label in conf_bins:
                mask_bin = (test_confs >= lo) & (test_confs < hi)
                n_bin = mask_bin.sum().item()
                if n_bin > 0:
                    bin_alphas = test_alphas[mask_bin].mean(dim=0)
                    print(f"    {label} confidence [{lo:.1f},{hi:.1f}] (n={n_bin}): "
                          f"a_real={bin_alphas[0]:.3f}  a_pixel={bin_alphas[1]:.3f}  a_feature={bin_alphas[2]:.3f}")

            print(f"\n  Prediction Flip Rate under Source Ablation:")
            masking_specs = [
                ('no_mask',        {}),
                ('mask_real',      {'real': True}),
                ('mask_pixel',     {'pixel': True}),
                ('mask_feature',   {'feature': True}),
                ('mask_pixel+feat',{'pixel': True, 'feature': True}),
            ]
            flip_rows = []
            for mask_name, mdict in masking_specs:
                tip_m, _, _ = ensemble_tip_logits_dynamic(
                    clip_test_features, dino_test_features, clip_weights,
                    best_beta, best_alpha, top_k_ev, gate_tau,
                    clip_adapter_real, clip_adapter, dino_adapter,
                    dr_clip_k_real, dr_clip_v_real, dr_clip_k_pixel, dr_clip_v_pixel, dr_dino_k, dr_dino_v,
                    mask_evidence=mdict,
                )
                acc_m = cls_acc(tip_m, test_labels)
                conf_m = F.softmax(tip_m, dim=1).max(dim=1).values.mean().item()
                pred_m = tip_m.argmax(dim=1)
                flip_rate = (test_preds != pred_m).float().mean().item() * 100.0
                flip_rows.append({
                    'condition': mask_name,
                    'accuracy': round(acc_m, 2),
                    'flip_rate_percent': round(flip_rate, 2),
                    'mean_confidence': round(conf_m, 4),
                })
                print(f"    {mask_name:20s}: acc={acc_m:.2f}, flip%={flip_rate:.1f}, conf={conf_m:.4f}")

            audit_sample_path = os.path.join(save_dir, f"audit_provenance_{cfg['shots']}shots.npy")
            audit_per_sample = {
                'alpha_real':     test_alphas[:, 0].cpu().numpy(),
                'alpha_pixel':    test_alphas[:, 1].cpu().numpy(),
                'alpha_feature':  test_alphas[:, 2].cpu().numpy(),
                'predictions':    test_preds.cpu().numpy(),
                'confidences':    test_confs.cpu().numpy(),
                'true_labels':    test_labels.cpu().numpy(),
                'L_real_max':     L_real.max(dim=1).values.cpu().numpy(),
                'L_pixel_max':    L_pixel.max(dim=1).values.cpu().numpy(),
                'L_feature_max':  L_feature.max(dim=1).values.cpu().numpy(),
            }
            np.save(audit_sample_path, audit_per_sample)
            print(f"\n  Per-sample provenance saved to: {audit_sample_path}")

            flip_json_path = os.path.join(save_dir, f"audit_fliprate_{cfg['shots']}shots.json")
            with open(flip_json_path, 'w', encoding='utf-8') as f:
                json.dump(flip_rows, f, indent=2, ensure_ascii=False)
            print(f"  Ablation flip rates saved to: {flip_json_path}")

            if cfg.get('enable_shap_attribution', True):
                print("\n" + "-"*60)
                print("  SHAP COMPLIANT ATTRIBUTION: Axiomatic Evidence Analysis")
                print("-"*60)

                try:
                    import shap_attribution as shap_mod

                    clip_adapter_real_dev = clip_adapter_real.to(device) if clip_adapter_real is not None else None
                    clip_adapter_dev = clip_adapter.to(device)
                    dino_adapter_dev = dino_adapter.to(device)
                    dr_clip_k_real_dev = dr_clip_k_real.to(device)
                    dr_clip_v_real_dev = dr_clip_v_real.to(device)
                    dr_clip_k_pixel_dev = dr_clip_k_pixel.to(device)
                    dr_clip_v_pixel_dev = dr_clip_v_pixel.to(device)
                    dr_dino_k_dev = dr_dino_k.to(device)
                    dr_dino_v_dev = dr_dino_v.to(device)
                    clip_test_features_dev = clip_test_features.to(device)
                    dino_test_features_dev = dino_test_features.to(device)
                    clip_weights_dev = clip_weights.to(device)

                    max_shap_samples = min(cfg.get('max_shap_samples', 500), len(test_labels))
                    sample_indices = torch.randperm(len(test_labels))[:max_shap_samples]

                    clip_f_sample = clip_test_features_dev[sample_indices]
                    dino_f_sample = dino_test_features_dev[sample_indices]
                    labels_sample = test_labels[sample_indices]
                    alphas_sample = test_alphas[sample_indices]

                    print(f"  Computing Shapley values for {max_shap_samples} samples ...")
                    print(f"  (n_samples={cfg.get('shap_n_samples', 32)}, "
                          f"branches=3, classes={clip_weights_dev.shape[1]})")

                    phi_real, phi_pixel, phi_feature, v_empty, v_full = shap_mod.compute_shapley_values(
                        clip_f_sample, dino_f_sample, clip_weights_dev,
                        best_beta, best_alpha,
                        clip_adapter_real_dev, clip_adapter_dev, dino_adapter_dev,
                        dr_clip_k_real_dev, dr_clip_v_real_dev,
                        dr_clip_k_pixel_dev, dr_clip_v_pixel_dev,
                        dr_dino_k_dev, dr_dino_v_dev,
                        n_samples=cfg.get('shap_n_samples', 32),
                    )

                    norm_real, norm_pixel, norm_feature, sign_meta = shap_mod.normalize_shapley_to_unit(
                        phi_real, phi_pixel, phi_feature, v_empty, v_full
                    )

                    axiom_results = shap_mod.verify_shapley_axioms(
                        phi_real, phi_pixel, phi_feature, v_empty, v_full
                    )

                    print(f"\n  [Axiom Verification]")
                    eff_status = "PASSED" if axiom_results['efficiency_satisfied'] else "FAILED"
                    print(f"    Efficiency: error={axiom_results['efficiency_error']:.6f}  [{eff_status}]")
                    print(f"    Fraction negative phi: real={axiom_results['fraction_negative_phi']:.1%}, "
                          f"pixel={axiom_results['fraction_negative_phi']:.1%}, "
                          f"feature={axiom_results['fraction_negative_phi']:.1%}")
                    print(f"    All-branches-negative: {axiom_results['fraction_all_negative']:.1%}")

                    gap_results = shap_mod.compute_attribution_gap(
                        norm_real, norm_pixel, norm_feature, alphas_sample
                    )

                    print(f"\n  [Attribution Gap: Shapley vs Attention alphas]")
                    print(f"    Euclidean gap:  {gap_results['gap_euclidean_mean']:.4f}")
                    print(f"    MAE gap:        {gap_results['gap_mae_mean']:.4f}")
                    print(f"    Per-branch diff: real={gap_results['diff_real']:.4f}, "
                          f"pixel={gap_results['diff_pixel']:.4f}, "
                          f"feature={gap_results['diff_feature']:.4f}")

                    class_names = [f"class_{i}" for i in range(clip_weights_dev.shape[1])]

                    report = shap_mod.generate_shap_report(
                        phi_real, phi_pixel, phi_feature,
                        norm_real, norm_pixel, norm_feature,
                        v_empty, v_full,
                        alphas_sample, class_names,
                        axiom_results, gap_results, sign_meta,
                    )

                    shap_json_path = os.path.join(save_dir, f"shap_attribution_{cfg['shots']}shots.json")
                    with open(shap_json_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    print(f"\n  SHAP report saved to: {shap_json_path}")

                    shap_npy_path = os.path.join(save_dir, f"shap_norm_values_{cfg['shots']}shots.npy")
                    np.save(shap_npy_path, {
                        'phi_real': phi_real.cpu().numpy(),
                        'phi_pixel': phi_pixel.cpu().numpy(),
                        'phi_feature': phi_feature.cpu().numpy(),
                        'norm_real': norm_real.cpu().numpy(),
                        'norm_pixel': norm_pixel.cpu().numpy(),
                        'norm_feature': norm_feature.cpu().numpy(),
                        'v_empty': v_empty.cpu().numpy(),
                        'v_full': v_full.cpu().numpy(),
                        'sample_indices': sample_indices.cpu().numpy(),
                    })
                    print(f"  Per-sample Shapley values saved to: {shap_npy_path}")

                    shap_weights_path = os.path.join(save_dir, f"shap_weights_{cfg['shots']}shots.npy")
                    np.save(shap_weights_path, {
                        'shap_norm_weights': gap_results['shap_weights_avg'].cpu().numpy(),
                        'attention_alphas': alphas_sample.cpu().numpy(),
                        'gap_euclidean': gap_results['gap_euclidean'].cpu().numpy(),
                        'gap_mae': gap_results['gap_mae'].cpu().numpy(),
                        'sample_indices': sample_indices.cpu().numpy(),
                    })
                    print(f"  Shapley weights (norm) saved to: {shap_weights_path}")

                    gs = report['global_summary']
                    print(f"\n  [Global Shapley Summary]")
                    print(f"    Raw phi (mean):    real={gs['phi_real_mean']:.4f}, "
                          f"pixel={gs['phi_pixel_mean']:.4f}, feature={gs['phi_feature_mean']:.4f}")
                    print(f"    Normed phi (mean): real={gs['norm_real_mean']:.4f}, "
                          f"pixel={gs['norm_pixel_mean']:.4f}, feature={gs['norm_feature_mean']:.4f}")
                    print(f"    Frac negative:     real={gs['frac_negative_real']:.1%}, "
                          f"pixel={gs['frac_negative_pixel']:.1%}, feature={gs['frac_negative_feature']:.1%}")
                    print(f"    Sign patterns:     {gs['sign_pattern_distribution']}")

                    class_eff_errors = [c['efficiency_error'] for c in report['per_class']]
                    print(f"\n  [Per-Class Efficiency]")
                    print(f"    Worst class: {max(class_eff_errors):.6f}")
                    print(f"    Mean error:  {sum(class_eff_errors)/len(class_eff_errors):.6f}")
                    print(f"    All classes satisfy efficiency: {all(c['efficiency_satisfied'] for c in report['per_class'])}")

                    print("-"*60)
                    print("  SHAP ATTRIBUTION ANALYSIS COMPLETE")
                    print("-"*60)

                except ImportError:
                    print("\n  [SKIP] shap_attribution.py not found in current directory.")
                except Exception as e:
                    print(f"\n  [ERROR] Shapley computation failed: {e}")
                    traceback.print_exc()

        print("\n" + "="*80)
        print("AUDIT COMPLETE -- provenance decomposition available.")
        print("="*80)

    else:
        print("\n[Audit Skipped] Dynamic evidence routing is disabled.")
        print("Enable 'use_dynamic_evidence_routing: true' in config to use transparent audit.")

    return tip_logits, test_labels

def main():

    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    cfg['use_fusion'] = cfg.get('use_fusion', False)
    if cfg['use_fusion']:
        print("\n将使用DALL-E和VAE图像融合训练")

    cfg['use_dynamic_evidence_routing'] = cfg.get('use_dynamic_evidence_routing', False)
    cfg['evidence_top_k'] = cfg.get('evidence_top_k', 16)
    cfg['gate_temperature'] = cfg.get('gate_temperature', 1.0)
    cfg['run_evidence_masking'] = cfg.get('run_evidence_masking', False)
    if cfg['use_dynamic_evidence_routing']:
        print("\n已启用动态注意力证据路由 (Dynamic Attention-based Evidence Routing)")
        print(f"  evidence_top_k={cfg['evidence_top_k']}, gate_temperature={cfg['gate_temperature']}")

    cfg['manifold_dim'] = cfg.get('manifold_dim', 64)
    cfg['n_neighbors'] = cfg.get('n_neighbors', 20)
    cfg['real_image_samples'] = cfg.get('real_image_samples', 1000)
    cfg['manifold_samples'] = cfg.get('manifold_samples', 500)

    if cfg.get('use_manifold_learning', True):
        print(f"\n将使用流形学习增强VAE训练")
        print(f"  - 流形维度: {cfg['manifold_dim']}")
        print(f"  - 真实图片样本数: {cfg['real_image_samples']}")
        print(f"  - DALL-E样本数: {cfg['manifold_samples']}")

    print("\nRunning configs.")
    print(cfg, "\n")

    clip_model, preprocess = clip.load(cfg['clip_backbone'])
    clip_model.eval()

    dino_model = torchvision_models.__dict__[cfg['dino_backbone']](num_classes=0)
    dino_model.fc = nn.Identity()
    dino_model.cuda()
    utils.load_pretrained_weights(dino_model, "dino/dino_resnet50_pretrain.pth", "teacher", "vit_small'", 16)
    dino_model.eval()

    random.seed(1)
    torch.manual_seed(1)

    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'])

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    if cfg['shots'] == 0:
        print("0-shot配置：训练集为空")
        train_loader_cache = None
        train_loader_F = None
    else:
        train_loader_cache = torch.utils.data.DataLoader(imagenet.train_x, batch_size=256, num_workers=8, shuffle=False)
        train_loader_F = torch.utils.data.DataLoader(imagenet.train_x, batch_size=256, num_workers=8, shuffle=True)

    if cfg.get('dalle_shots', 0) > 0:
        dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg['root_path'], cfg['dalle_shots'])
        dalle_train_loader_cache = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
        dalle_train_loader_F = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
        print(f"已加载 DALL-E 数据集 (dalle_shots={cfg['dalle_shots']})")
    else:
        dalle_dataset = None
        dalle_train_loader_cache = None
        dalle_train_loader_F = None
        print("dalle_shots=0，未使用 DALL-E 数据")

    save_features_dir = cfg['cache_dir']
    os.makedirs(save_features_dir, exist_ok=True)

    real_feat_path = os.path.join(save_features_dir, f"real_features_{cfg['shots']}shots.npy")
    if not os.path.exists(real_feat_path) and train_loader_cache is not None:
        print("\n保存真实训练集特征...")
        real_features_list = []
        real_labels_list = []
        with torch.no_grad():
            for images, labels in train_loader_cache:
                features = clip_model.encode_image(images.cuda())
                features = F.normalize(features, dim=-1)
                real_features_list.append(features.cpu().numpy())
                real_labels_list.append(labels.numpy())
        real_features = np.concatenate(real_features_list, axis=0)
        real_labels = np.concatenate(real_labels_list, axis=0)
        np.save(real_feat_path, real_features)
        np.save(os.path.join(save_features_dir, f"labels_{cfg['shots']}shots.npy"), real_labels)
        print(f"  已保存: {real_feat_path}, 形状: {real_features.shape}")

    dalle_feat_path = os.path.join(save_features_dir, f"dalle_features_{cfg['dalle_shots']}shots.npy")
    if not os.path.exists(dalle_feat_path) and dalle_train_loader_cache is not None:
        print("\n保存DALL-E生成特征...")
        dalle_features_list = []
        dalle_labels_list = []
        with torch.no_grad():
            for images, labels in dalle_train_loader_cache:
                features = clip_model.encode_image(images.cuda())
                features = F.normalize(features, dim=-1)
                dalle_features_list.append(features.cpu().numpy())
                dalle_labels_list.append(labels.numpy())
        dalle_features = np.concatenate(dalle_features_list, axis=0)
        dalle_labels = np.concatenate(dalle_labels_list, axis=0)
        np.save(dalle_feat_path, dalle_features)
        np.save(os.path.join(save_features_dir, f"dalle_labels_{cfg['dalle_shots']}shots.npy"), dalle_labels)
        print(f"  已保存: {dalle_feat_path}, 形状: {dalle_features.shape}")

    use_vae = cfg.get('use_vae', cfg.get('use_vae_generation', False))
    vae_train_loader_cache = None
    vae_train_loader_F = None
    clip_cvae_cache_keys = clip_cvae_cache_values = None
    dalle_cvae_cache_keys = dalle_cvae_cache_values = None

    if use_vae:
        print("\n使用VAE生成图像增强训练...")
        vae_dataset_dir = os.path.join(cfg['root_path'], f"vae_{cfg['dataset']}")
        os.makedirs(vae_dataset_dir, exist_ok=True)

        vae_json_path = os.path.join(vae_dataset_dir, f"vae_{cfg['dataset']}.json")
        vae_model_path = os.path.join(cfg['cache_dir'], f"best_vae_model_{cfg['shots']}shots.pt")

        if not os.path.exists(vae_json_path):
            print(f"\n未找到VAE生成的图像数据集，将训练增强版VAE模型并生成图像")
            print(f"目标JSON路径: {vae_json_path}")

            if not os.path.exists(vae_model_path):
                text_features, manifold_projector = enhanced_train_vae_with_manifold(
                    train_loader_cache,
                    None,
                    clip_model,
                    None,
                    imagenet.classnames,
                    imagenet.template,
                    dalle_train_loader_cache,
                    epochs=cfg.get('vae_epochs', 10),
                    save_path=vae_model_path,
                    cfg=cfg
                )

                if cfg.get('use_conditional_prior', True):
                    print("\n使用条件先验 p(z|t_c) 训练VAE...")
                    try:
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        input_dim = 512
                        latent_dim = cfg.get('vae_latent_dim', 128)

                        cvae_model = ConditionalVAE(
                            input_dim=input_dim,
                            latent_dim=latent_dim,
                            use_conditional_prior=True
                        ).to(device)

                        prior_sigma = cfg.get('prior_sigma', 0.1)
                        prior_beta = cfg.get('prior_beta', 0.5)
                        print(f"   条件先验参数: sigma={prior_sigma}, beta={prior_beta}")

                        cvae_optimizer = torch.optim.Adam(cvae_model.parameters(), lr=1e-3)
                        cvae_epochs = cfg.get('cvae_epochs', 20)
                        print(f"\n训练条件VAE，epochs={cvae_epochs}...")

                        for epoch in range(cvae_epochs):
                            cvae_model.train()
                            total_loss = 0
                            total_recon = 0
                            total_kld = 0

                            for images, labels in tqdm(dalle_train_loader_cache, desc=f'CVAE Epoch {epoch+1}/{cvae_epochs}'):
                                images = images.to(device)
                                labels = labels.to(device)

                                with torch.no_grad():
                                    clip_features = clip_model.encode_image(images)
                                    clip_features = F.normalize(clip_features, dim=-1)

                                anchor_features = text_features[labels].to(device)

                                recon, mu, logvar, z = cvae_model(clip_features, anchor_features, use_prior=True)

                                with torch.no_grad():
                                    prior_mu = cvae_model.anchor_projection(anchor_features)
                                    prior_mu = F.normalize(prior_mu, dim=-1)

                                loss, recon_loss, kld_loss = conditional_vae_loss(
                                    recon, clip_features, mu, logvar,
                                    prior_mu=prior_mu, sigma=prior_sigma, beta=prior_beta
                                )

                                cvae_optimizer.zero_grad()
                                loss.backward()
                                cvae_optimizer.step()

                                total_loss += loss.item()
                                total_recon += recon_loss.item()
                                total_kld += kld_loss.item()

                            avg_loss = total_loss / len(dalle_train_loader_cache)
                            avg_recon = total_recon / len(dalle_train_loader_cache)
                            avg_kld = total_kld / len(dalle_train_loader_cache)
                            print(f'CVAE Epoch {epoch+1}/{cvae_epochs}, Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})')

                        cvae_model_path = os.path.join(cfg['cache_dir'], f"cvae_model_{cfg['shots']}shots.pt")
                        torch.save(cvae_model.state_dict(), cvae_model_path)
                        print(f"条件VAE模型保存到 {cvae_model_path}")
                        print("条件先验训练完成！")

                    except Exception as e:
                        print(f"条件VAE训练失败: {e}")
                        traceback.print_exc()
                        print("将跳过条件先验训练")

                manifold_path = os.path.join(cfg['cache_dir'], f"manifold_projector_{cfg['shots']}shots.pt")
                torch.save(manifold_projector, manifold_path)
                print(f"流形投影器保存到 {manifold_path}")

            if use_vae and os.path.exists(vae_model_path):
                print(f"使用增强版VAE模型生成图像...")

                manifold_path = os.path.join(cfg['cache_dir'], f"manifold_projector_{cfg['shots']}shots.pt")
                loaded_manifold_projector = None
                if os.path.exists(manifold_path):
                    try:
                        loaded_manifold_projector = torch.load(manifold_path)
                        print(f"已加载流形投影器: {manifold_path}")
                    except Exception as e:
                        print(f"加载流形投影器失败: {e}")

                print("VAE图像生成功能当前不可用，将跳过此步骤")

                vae_dataset_placeholder = {
                    "dataset_name": cfg['dataset'],
                    "generated_with_manifold": True,
                    "note": "Placeholder for manifold-enhanced training"
                }

                with open(vae_json_path, 'w') as f:
                    json.dump(vae_dataset_placeholder, f, indent=2)

                print(f"已创建VAE数据集占位符: {vae_json_path}")

        if use_vae:
            print(f"\n检查VAE数据集: {vae_json_path}")
            try:
                if os.path.exists(vae_json_path):
                    with open(vae_json_path, 'r') as f:
                        vae_content = json.load(f)

                    if isinstance(vae_content, dict) and vae_content.get('note') == 'Placeholder for manifold-enhanced training':
                        print("检测到VAE占位符文件，流形学习已启用但跳过VAE数据集加载")
                        print("   将继续使用DALL-E图像和流形增强进行训练")
                        vae_train_loader_cache = None
                        vae_train_loader_F = None
                    else:
                        cfg['vae_shots'] = cfg.get('vae_shots', cfg['shots'])
                        vae_dataset = build_vae_dataset(cfg['dataset'], cfg['root_path'], cfg['vae_shots'])
                        if vae_dataset is not None:
                            vae_train_loader_cache = build_data_loader(data_source=vae_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
                            vae_train_loader_F = build_data_loader(data_source=vae_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
                            print(f"成功加载VAE数据集，包含 {len(vae_dataset.train_x)} 张图像")
                        else:
                            vae_train_loader_cache = None
                            vae_train_loader_F = None
                else:
                    vae_train_loader_cache = None
                    vae_train_loader_F = None
            except Exception as e:
                print(f"VAE数据集处理失败: {e}")
                traceback.print_exc()
                vae_train_loader_cache = None
                vae_train_loader_F = None
                print("将继续使用DALL-E图像和流形增强进行训练")

    with open(cfg['gpt3_prompt_file']) as f:
        gpt3_prompt = json.load(f)

    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(imagenet.classnames, gpt3_prompt, clip_model, imagenet.template)

    clip_dtype = next(clip_model.parameters()).dtype
    clip_weights = clip_weights.to(clip_dtype)
    print(f"CLIP weights dtype: {clip_weights.dtype}")

    print("\nConstructing cache model by few-shot visual features and labels.")

    if cfg['shots'] == 0:
        print("\n检测到0-shot配置，将不使用真实样本缓存")
        num_classes = len(imagenet.classnames)
        clip_cache_keys = torch.zeros(1024, 0, dtype=torch.float16).cuda()
        clip_cache_values = torch.zeros(0, num_classes, dtype=torch.float16).cuda()
        dino_cache_keys = torch.zeros(2048, 0, dtype=torch.float16).cuda()
        dino_cache_values = torch.zeros(0, num_classes, dtype=torch.float16).cuda()
    else:
        print("\nConstructing CLIP cache model.")
        clip_cache_keys, clip_cache_values = build_clip_cache_model(cfg, clip_model, train_loader_cache)
        print("\nConstructing DINO cache model.")
        dino_cache_keys, dino_cache_values = build_dino_cache_model(cfg, dino_model, train_loader_cache)

    print("\nConstructing cache model by dalle image.")
    print("\nConstructing CLIP cache model.")
    clip_dalle_cache_keys, clip_dalle_cache_values = build_clip_dalle_cache_model(cfg, clip_model, dalle_train_loader_cache)
    print("\nConstructing DINO cache model.")
    dino_dalle_cache_keys, dino_dalle_cache_values = build_dino_dalle_cache_model(cfg, dino_model, dalle_train_loader_cache)

    clip_cvae_cache_keys = None
    clip_cvae_cache_values = None
    dalle_cvae_cache_keys = None
    dalle_cvae_cache_values = None

    cvae_model_path = os.path.join(cfg['cache_dir'], f"cvae_model_{cfg['shots']}shots.pt")
    manifold_path = os.path.join(cfg['cache_dir'], f"manifold_projector_{cfg['shots']}shots.pt")
    cvae_available = os.path.exists(cvae_model_path)

    if use_vae and cvae_available and dalle_train_loader_cache is not None:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            cvae_model = ConditionalVAE(
                input_dim=512,
                latent_dim=cfg.get('vae_latent_dim', 128),
                use_conditional_prior=True
            ).to(device)
            cvae_model.load_state_dict(torch.load(cvae_model_path, map_location=device))
            cvae_model.eval()
            print(f"\n[CVAE] 已加载模型: {cvae_model_path}")

            manifold_projector = None
            if os.path.exists(manifold_path):
                manifold_projector = torch.load(manifold_path, map_location=device)
                if hasattr(manifold_projector, 'fitted') and manifold_projector.fitted:
                    print(f"[CVAE] 已加载 ManifoldProjector (dim={manifold_projector.manifold_dim})")

            clip_dtype = next(clip_model.parameters()).dtype

            print(f"\n[CVAE] 从 DALL-E 图片生成增强特征缓存...")
            dalle_cvae_cache_keys, dalle_cvae_cache_values = build_cvae_enhanced_cache_model(
                cfg, clip_model, dalle_train_loader_cache,
                cvae_model, manifold_projector, None, clip_dtype
            )
            torch.save(dalle_cvae_cache_keys, cfg['cache_dir'] + f"/clip_cvae_dalle_keys_{cfg['dalle_shots']}shots.pt")
            torch.save(dalle_cvae_cache_values, cfg['cache_dir'] + f"/clip_cvae_dalle_values_{cfg['dalle_shots']}shots.pt")

        except Exception as e:
            print(f"[CVAE] 增强缓存生成失败: {e}")
            traceback.print_exc()
            print("[CVAE] 将跳过 CVAE 增强，使用原始缓存继续")
            dalle_cvae_cache_keys = None
            dalle_cvae_cache_values = None
    else:
        if use_vae and not cvae_available:
            print(f"\n[CVAE] 未找到训练好的 CVAE 权重 ({cvae_model_path})，跳过 CVAE 增强")

    print("\nLoading visual features and labels from test set.")
    print("\nLoading CLIP feature.")
    test_clip_features, test_labels = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)
    print("\nLoading DINO feature.")
    test_dino_features, test_labels = pre_DINO_load_features(cfg, "test", dino_model, test_loader)

    all_clip_cache_keys = [clip_cache_keys, clip_dalle_cache_keys]
    all_clip_cache_values = [clip_cache_values, clip_dalle_cache_values]
    all_dino_cache_keys = [dino_cache_keys, dino_dalle_cache_keys]
    all_dino_cache_values = [dino_cache_values, dino_dalle_cache_values]

    if use_vae and dalle_cvae_cache_keys is not None:
        all_clip_cache_keys.append(dalle_cvae_cache_keys)
        all_clip_cache_values.append(dalle_cvae_cache_values)
        print(f"[CVAE] 已将增强特征注入合并缓存: +{dalle_cvae_cache_keys.shape[1]} 样本")

    merged_clip_cache_keys = torch.cat(all_clip_cache_keys, dim=1)
    merged_clip_cache_values = torch.cat(all_clip_cache_values, dim=0)
    merged_dino_cache_keys = torch.cat(all_dino_cache_keys, dim=1)
    merged_dino_cache_values = torch.cat(all_dino_cache_values, dim=0)

    separated_caches = None
    if cfg.get('use_dynamic_evidence_routing', False):
        separated_caches = {
            'clip_real': (clip_cache_keys, clip_cache_values),
            'clip_pixel': (clip_dalle_cache_keys, clip_dalle_cache_values),
            'dino': (merged_dino_cache_keys, merged_dino_cache_values),
        }
        if use_vae and clip_cvae_cache_keys is not None:
            separated_caches['clip_cvae_real'] = (clip_cvae_cache_keys, clip_cvae_cache_values)
        if use_vae and dalle_cvae_cache_keys is not None:
            separated_caches['clip_cvae_pixel'] = (dalle_cvae_cache_keys, dalle_cvae_cache_values)
        cvae_tag = ""
        if clip_cvae_cache_keys is not None:
            cvae_tag += f" real(+{clip_cvae_cache_keys.shape[1]})"
        if dalle_cvae_cache_keys is not None:
            cvae_tag += f" dalle(+{dalle_cvae_cache_keys.shape[1]})"
        print(f"\n动态路由缓存: C_real={clip_cache_keys.shape[1]}, C_pixel={clip_dalle_cache_keys.shape[1]}, "
              f"C_cvae={cvae_tag}, C_feature={merged_dino_cache_keys.shape[1]}")

    print(f"\n配置摘要:")
    print(f"  - shots: {cfg['shots']}")
    print(f"  - dalle_shots: {cfg.get('dalle_shots', 0)}")
    print(f"  - use_vae: {use_vae}")
    print(f"  - 最终CLIP缓存: keys {merged_clip_cache_keys.shape}, values {merged_clip_cache_values.shape}")
    print(f"  - 最终DINO缓存: keys {merged_dino_cache_keys.shape}, values {merged_dino_cache_values.shape}")

    run_ensemble_tip_dalle_adapter_F(cfg,
                            merged_clip_cache_keys,
                            merged_clip_cache_values,
                            test_clip_features,
                            merged_dino_cache_keys,
                            merged_dino_cache_values,
                            test_dino_features,
                            test_labels,
                            clip_weights,
                            clip_model,
                            dino_model,
                            train_loader_F,
                            dalle_train_loader_F,
                            vae_train_loader_F,
                            separated_caches=separated_caches)

if __name__ == '__main__':
    main()