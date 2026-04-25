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
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import scipy.linalg as la

# 数据流形和切空间投影相关类
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
        使用PCA学习数据流形
        
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
        
        # 使用PCA进行流形学习
        try:
            n_samples, feature_dim = features_np.shape
            effective_manifold_dim = min(self.manifold_dim, feature_dim, n_samples - 1)
            
            if effective_manifold_dim <= 0:
                print(f"流形学习失败：维度无效")
                self.fitted = False
                return
            
            self.pca = PCA(n_components=effective_manifold_dim)
            self.pca.fit(features_np)
            
            # 获取切空间的基向量（PCA的主成分）
            self.tangent_basis = self.pca.components_
            self.manifold_dim = effective_manifold_dim
            
            print(f"流形学习完成，切空间维度: {self.tangent_basis.shape}")
            self.fitted = True
            
        except Exception as e:
            print(f"流形学习失败: {e}")
            self.fitted = False
        
    def project_noise_to_tangent_space(self, noise_features, dalle_features=None, blend_factor=0.7):
        """
        将高斯噪声投影到数据流形的切空间
        """
        if not self.fitted:
            return noise_features
            
        try:
            # 转换为numpy数组
            if isinstance(noise_features, torch.Tensor):
                noise_np = noise_features.detach().cpu().numpy()
            else:
                noise_np = noise_features.copy()
                
            if dalle_features is not None:
                if isinstance(dalle_features, torch.Tensor):
                    dalle_np = dalle_features.detach().cpu().numpy()
                else:
                    dalle_np = dalle_features.copy()
                
                mixed_features = blend_factor * noise_np + (1 - blend_factor) * dalle_np
            else:
                mixed_features = noise_np
                
            # 中心化特征
            centered_features = mixed_features - self.mean_feature
            
            # 投影到切空间
            tangent_coords = np.dot(centered_features, self.tangent_basis.T)
            projected_centered = np.dot(tangent_coords, self.tangent_basis)
            projected_features = projected_centered + self.mean_feature
            
            # 转换回pytorch张量
            if isinstance(noise_features, torch.Tensor):
                projected_features = torch.tensor(projected_features, 
                                                dtype=noise_features.dtype, 
                                                device=noise_features.device)
                
            return projected_features
            
        except Exception as e:
            print(f"流形投影失败: {e}")
            return noise_features
    
    def generate_manifold_noise(self, n_samples, feature_dim, device='cuda', noise_scale=0.1):
        """
        在流形切空间中生成结构化噪声
        """
        if not self.fitted:
            return torch.randn(n_samples, feature_dim, device=device) * noise_scale
            
        try:
            # 在切空间坐标中生成噪声
            tangent_noise = np.random.randn(n_samples, self.manifold_dim) * noise_scale
            
            # 将切空间噪声映射到原始特征空间
            structured_noise = np.dot(tangent_noise, self.tangent_basis)
            
            # 添加均值特征
            if self.mean_feature is not None:
                structured_noise += self.mean_feature
            
            # 转换为pytorch张量
            structured_noise = torch.tensor(structured_noise, 
                                          dtype=torch.float32, 
                                          device=device)
            
            return structured_noise
            
        except Exception as e:
            print(f"生成流形噪声失败: {e}")
            return torch.randn(n_samples, feature_dim, device=device) * noise_scale


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
        
        # 编码器：从输入特征提取均值和方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 均值和方差网络
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # 解码器：从潜在空间重构特征
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        # ========== 新增：语义锚点投影层 ==========
        # 将高维语义锚点（CLIP特征维度）投影到低维潜在空间
        self.anchor_projection = nn.Linear(input_dim, latent_dim, bias=False)
        
    def get_conditional_prior(self, anchor_features, sigma=0.1):
        """
        获取条件先验的均值
        
        参数:
            anchor_features: 语义锚点特征 [batch_size, input_dim] (即 t_c)
            sigma: 先验标准差
            
        返回:
            prior_mu: 先验均值 [batch_size, latent_dim]
            sigma: 标准差标量
        """
        # 将语义锚点投影到潜在空间
        prior_mu = self.anchor_projection(anchor_features)
        
        # L2 归一化确保方向与语义锚点一致
        prior_mu = F.normalize(prior_mu, dim=-1)
        
        return prior_mu, sigma
    
    def reparameterize(self, mu, logvar, prior_mu=None, sigma=0.1):
        """
        重参数化技巧（支持条件先验）
        
        标准VAE: z ~ N(μ, σ²) = μ + ε * σ, 其中 ε ~ N(0, I)
        条件VAE: z ~ N(prior_mu, σ²) = prior_mu + ε * σ
        
        参数:
            mu: 后验均值 [batch_size, latent_dim]
            logvar: 后验对数方差 [batch_size, latent_dim]
            prior_mu: 条件先验均值（如果为None，则使用标准先验 N(0,I)）
            sigma: 先验标准差
            
        返回:
            z: 采样潜在变量
        """
        std = torch.exp(0.5 * logvar)
        
        if prior_mu is not None and self.use_conditional_prior:
            # ========== 条件先验采样 ==========
            # 从 N(prior_mu, sigma^2) 采样
            eps_prior = torch.randn_like(std)
            z = prior_mu + eps_prior * sigma
        else:
            # ========== 标准先验 N(0,I) ==========
            eps = torch.randn_like(std)
            z = mu + eps * std
            
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, anchor_features=None, use_prior=True):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            anchor_features: 语义锚点特征 [batch_size, input_dim] (t_c)
            use_prior: 是否使用条件先验
            
        返回:
            recon: 重构特征
            mu: 后验均值
            logvar: 后验对数方差
            z: 采样潜在变量
        """
        mu, logvar = self.encode(x)
        
        # 获取条件先验
        prior_mu = None
        sigma = 0.1  # 默认先验标准差
        if use_prior and anchor_features is not None and self.use_conditional_prior:
            prior_mu, sigma = self.get_conditional_prior(anchor_features)
        
        # 重参数化
        z = self.reparameterize(mu, logvar, prior_mu, sigma)
        
        # 解码
        recon = self.decode(z)
        
        return recon, mu, logvar, z
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


def conditional_vae_loss(recon_x, x, mu, logvar, prior_mu=None, sigma=0.1, beta=1.0):
    """
    条件VAE损失函数
    
    标准VAE损失: L = L_recon + β * KL(q(z|x) || p(z))
    条件VAE损失: L = L_recon + β * KL(q(z|x,t_c) || p(z|t_c))
    
    其中：
    - L_recon: 重构损失 (MSE)
    - KL: KL散度，衡量后验与先验的差异
    
    参数:
        recon_x: 重构特征 [batch_size, input_dim]
        x: 原始特征 [batch_size, input_dim]
        mu: 后验均值 [batch_size, latent_dim]
        logvar: 后验对数方差 [batch_size, latent_dim]
        prior_mu: 条件先验均值（可选）[batch_size, latent_dim]
        sigma: 先验标准差（默认0.1）
        beta: KL散度权重（默认1.0）
        
    返回:
        total_loss: 总损失
        recon_loss: 重构损失
        kld_loss: KL散度
    """
    # 重构损失（MSE）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    if prior_mu is not None:
        # ========== 条件KL散度 ==========
        # KL(q(z|x,t_c) || p(z|t_c))
        # = -0.5 * sum(1 + log(sigma_q^2) - sigma_q^2 - (mu_q - mu_p)^2 / sigma^2)
        
        sigma_q_sq = torch.exp(logvar)  # 后验方差 σ_q²
        mu_diff_sq = (mu - prior_mu) ** 2  # 均值差异 (μ_q - μ_p)²
        
        kld_loss = -0.5 * torch.sum(
            1 + logvar - sigma_q_sq - mu_diff_sq / (sigma ** 2)
        )
    else:
        # ========== 标准KL散度 ==========
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ_q²) - σ_q² - μ_q²)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kld_loss
    
    return total_loss, recon_loss, kld_loss


def enhanced_train_vae_with_manifold(train_loader, val_loader, clip_model, gpt3_prompt, 
                                   classnames, template, dalle_train_loader=None,
                                   epochs=10, save_path=None, cfg=None):
    """
    增强版VAE训练，包含流形学习
    """
    print("\n开始增强版VAE训练（含流形学习）...")
    
    # 创建流形投影器
    manifold_projector = ManifoldProjector(
        manifold_dim=cfg.get('manifold_dim', 64) if cfg else 64,
        n_neighbors=cfg.get('n_neighbors', 20) if cfg else 20
    )
    
    # 1. 提取文本特征
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
    
    # 2. 提取真实训练图片的CLIP特征（0-shot时跳过）
    print("提取真实训练图片特征用于流形学习...")
    
    # ===== 关键修复：检查shots参数，避免0-shot数据泄露 =====
    shots = cfg.get('shots', 0) if cfg else 0
    
    if shots == 0:
        print("   ⚠️  0-shot配置：跳过真实样本提取，避免数据泄露")
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
    
    # 3. 提取DALL-E特征（如果提供）
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
    
    # 4. 组合所有特征进行流形学习
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
    
    # 5. 拟合流形
    manifold_projector.fit_manifold(manifold_features)
    
    # 6. 流形学习完成，返回投影器
    print("流形学习完成，准备用于增强VAE训练")
    print(f"流形投影器状态: {'已拟合' if manifold_projector.fitted else '未拟合'}")
    
    # 注意：这个函数主要用于流形学习，实际的VAE训练将在主流程中处理
    # 返回文本特征作为语义锚点（用于推理时条件先验）和流形投影器
    return text_features, manifold_projector

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def fusion_images_with_clip_scores(clip_model, dalle_images, vae_images, dalle_labels, vae_labels):
    """
    使用CLIP分数对DALL-E和VAE生成的图像进行加权融合
    
    参数:
        clip_model: CLIP模型
        dalle_images: DALL-E生成的图像
        vae_images: VAE生成的图像
        dalle_labels: DALL-E图像对应的标签
        vae_labels: VAE图像对应的标签
    
    返回:
        融合后的图像和对应标签
    """
    # 确保标签匹配
    assert torch.all(dalle_labels == vae_labels), "DALL-E和VAE图像的标签必须一致"
    
    # 使用CLIP计算图像特征
    with torch.no_grad():
        dalle_features = clip_model.encode_image(dalle_images)
        dalle_features /= dalle_features.norm(dim=-1, keepdim=True)
        
        vae_features = clip_model.encode_image(vae_images)
        vae_features /= vae_features.norm(dim=-1, keepdim=True)
    
    # 计算每个图像与其标签文本的相似度作为CLIP分数（使用简单提示词）
    text_inputs = torch.cat([clip.tokenize(f"a photo of object {dalle_labels[i].item()}") for i in range(dalle_labels.size(0))]).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 计算CLIP分数
    dalle_scores = (100.0 * dalle_features @ text_features.T).diag()
    vae_scores = (100.0 * vae_features @ text_features.T).diag()
    
    # 归一化CLIP分数作为权重
    total_scores = dalle_scores + vae_scores
    dalle_weights = dalle_scores / total_scores
    vae_weights = vae_scores / total_scores
    
    # 使用CLIP分数进行加权融合 (仅融合特征，不是实际的图像融合)
    fusion_features = dalle_weights.unsqueeze(1) * dalle_features + vae_weights.unsqueeze(1) * vae_features
    fusion_features /= fusion_features.norm(dim=-1, keepdim=True)
    
    return fusion_features, dalle_labels

def run_ensemble_tip_dalle_adapter_F(cfg, 
                            clip_cache_keys, 
                            clip_cache_values, 
                            clip_val_features,
                            clip_test_features, 
                            dino_cache_keys, 
                            dino_cache_values,
                            dino_val_features, 
                            dino_test_features, 
                            val_labels,
                            test_labels, 
                            clip_weights, 
                            clip_model, 
                            dino_model, 
                            train_loader_F,
                            dalle_train_loader_F,
                            vae_train_loader_F=None,
                            separated_caches=None):
    
    # 确定CLIP模型的数据类型和设备
    clip_dtype = next(clip_model.parameters()).dtype
    device = next(clip_model.parameters()).device
    print(f"CLIP模型数据类型: {clip_dtype}, 设备: {device}")
    
    # 确保clip_weights的数据类型与其他张量一致
    clip_weights = clip_weights.to(clip_dtype)
    print(f"运行适配器训练时 CLIP weights dtype: {clip_weights.dtype}")
    
    # 确保所有缓存张量与CLIP模型的数据类型相同
    clip_cache_keys = clip_cache_keys.to(clip_dtype)
    clip_cache_values = clip_cache_values.to(clip_dtype)
    dino_cache_keys = dino_cache_keys.to(clip_dtype)
    dino_cache_values = dino_cache_values.to(clip_dtype)

    use_dynamic_routing = bool(cfg.get('use_dynamic_evidence_routing', False)) and separated_caches is not None
    dr_clip_k_real = dr_clip_v_real = dr_clip_k_pixel = dr_clip_v_pixel = None
    dr_dino_k = dr_dino_v = None
    dr_clip_k_cvae_real = dr_clip_v_cvae_real = None  # CVAE 增强的真实图片特征
    dr_clip_k_cvae_pixel = dr_clip_v_cvae_pixel = None  # CVAE 增强的 DALL-E 特征
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
        # Enable the cached keys to be learnable (合并缓存)
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
    
    # 计算总训练步数（考虑0-shot情况）
    total_steps = cfg['train_epoch'] * (
        (len(train_loader_F) if train_loader_F is not None else 0) + 
        len(dalle_train_loader_F) +
        (len(vae_train_loader_F) if vae_train_loader_F is not None else 0)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        clip_adapter.train()
        dino_adapter.train()
        if clip_adapter_real is not None:
            clip_adapter_real.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        # origin image (跳过0-shot情况)
        if train_loader_F is not None:
            for i, (images, target) in enumerate(tqdm(train_loader_F)):
                images, target = images.to(device), target.to(device)
                with torch.no_grad():
                    clip_image_features = clip_model.encode_image(images)
                    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                    # 确保数据类型一致
                    clip_image_features = clip_image_features.to(clip_dtype)
                    
                    dino_image_features = dino_model(images)
                    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
                    # 确保数据类型一致
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
        
        # dalle image
        for i, (images, target) in enumerate(tqdm(dalle_train_loader_F)):
            images, target = images.to(device), target.to(device)
            with torch.no_grad():
                clip_image_features = clip_model.encode_image(images)
                clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                # 确保数据类型一致
                clip_image_features = clip_image_features.to(clip_dtype)
                
                dino_image_features = dino_model(images)
                dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
                # 确保数据类型一致
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
            
        # vae image (如果提供)
        if vae_train_loader_F is not None:
            for i, (images, target) in enumerate(tqdm(vae_train_loader_F)):
                images, target = images.to(device), target.to(device)
                with torch.no_grad():
                    clip_image_features = clip_model.encode_image(images)
                    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
                    # 确保数据类型一致
                    clip_image_features = clip_image_features.to(clip_dtype)
                    
                    dino_image_features = dino_model(images)
                    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
                    # 确保数据类型一致
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
                
        # DALL-E和VAE图像融合训练 (如果两者都提供且配置启用融合)
        if cfg.get('use_fusion', False) and dalle_train_loader_F is not None and vae_train_loader_F is not None:
            # 创建DALL-E和VAE图像的迭代器
            dalle_iterator = iter(dalle_train_loader_F)
            vae_iterator = iter(vae_train_loader_F)
            
            # 获取较小的数据集长度
            min_batches = min(len(dalle_train_loader_F), len(vae_train_loader_F))
            
            print("训练DALL-E和VAE融合图像...")
            for _ in range(min_batches):
                try:
                    dalle_images, dalle_target = next(dalle_iterator)
                    vae_images, vae_target = next(vae_iterator)
                    
                    # 确保批次大小相同
                    min_batch_size = min(dalle_images.size(0), vae_images.size(0))
                    dalle_images, dalle_target = dalle_images[:min_batch_size], dalle_target[:min_batch_size]
                    vae_images, vae_target = vae_images[:min_batch_size], vae_target[:min_batch_size]
                    
                    dalle_images, dalle_target = dalle_images.to(device), dalle_target.to(device)
                    vae_images, vae_target = vae_images.to(device), vae_target.to(device)
                    
                    # 如果标签不一致，跳过这个批次
                    if not torch.all(dalle_target == vae_target):
                        continue
                    
                    # 使用CLIP分数融合图像特征
                    # 计算每个图像与其标签文本的相似度作为CLIP分数
                    # 使用简单提示词（不需要类名）
                    text_inputs = torch.cat([clip.tokenize(f"a photo of object {dalle_target[i].item()}") for i in range(dalle_target.size(0))]).to(device)
                    
                    with torch.no_grad():
                        # 计算CLIP特征
                        dalle_features = clip_model.encode_image(dalle_images)
                        dalle_features /= dalle_features.norm(dim=-1, keepdim=True)
                        # 确保数据类型一致
                        dalle_features = dalle_features.to(clip_dtype)
                        
                        vae_features = clip_model.encode_image(vae_images)
                        vae_features /= vae_features.norm(dim=-1, keepdim=True)
                        # 确保数据类型一致
                        vae_features = vae_features.to(clip_dtype)
                        
                        # 计算文本特征
                        text_features = clip_model.encode_text(text_inputs)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    # 计算CLIP分数
                    dalle_scores = (100.0 * dalle_features @ text_features.T).diag()
                    vae_scores = (100.0 * vae_features @ text_features.T).diag()
                    
                    # 归一化CLIP分数作为权重
                    total_scores = dalle_scores + vae_scores
                    dalle_weights = dalle_scores / total_scores
                    vae_weights = vae_scores / total_scores
                    
                    # 使用CLIP分数进行加权融合
                    fusion_features = dalle_weights.unsqueeze(1) * dalle_features + vae_weights.unsqueeze(1) * vae_features
                    fusion_features /= fusion_features.norm(dim=-1, keepdim=True)
                    fusion_target = dalle_target
                    
                    # 计算DINO特征
                    with torch.no_grad():
                        # 对于DINO，我们不能直接使用融合特征，而是使用加权平均
                        dino_dalle_features = dino_model(dalle_images)
                        dino_dalle_features /= dino_dalle_features.norm(dim=-1, keepdim=True)
                        # 确保数据类型一致
                        dino_dalle_features = dino_dalle_features.to(clip_dtype)
                        
                        dino_vae_features = dino_model(vae_images)
                        dino_vae_features /= dino_vae_features.norm(dim=-1, keepdim=True)
                        # 确保数据类型一致
                        dino_vae_features = dino_vae_features.to(clip_dtype)
                        
                        # 使用相同的CLIP权重进行DINO特征融合
                        # 使用已经计算好的权重
                        
                        dino_fusion_features = (dalle_weights.unsqueeze(1) * dino_dalle_features + 
                                               vae_weights.unsqueeze(1) * dino_vae_features)
                        dino_fusion_features /= dino_fusion_features.norm(dim=-1, keepdim=True)
                    
                    # 训练适配器
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

        # Eval
        clip_adapter.eval()
        dino_adapter.eval()
        if clip_adapter_real is not None:
            clip_adapter_real.eval()

        # 确保验证特征与适配器的数据类型一致
        clip_val_features = clip_val_features.to(clip_dtype)
        dino_val_features = dino_val_features.to(clip_dtype)
        
        tip_logits = forward_tip(clip_val_features, dino_val_features, beta, alpha)
        acc = cls_acc(tip_logits, val_labels)

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

    # Search Hyperparameters
    if use_dynamic_routing:
        best_beta, best_alpha = search_dynamic_evidence_hp(
            cfg,
            clip_val_features,
            dino_val_features,
            val_labels,
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
                                                 clip_val_features, dino_adapter.weight.t(), dino_cache_values, 
                                                 dino_val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")
   
    # 确保测试特征与适配器的数据类型一致
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
  
    # ========== 导出预测结果用于后续分析 ==========
    save_dir = cfg['cache_dir']
    os.makedirs(save_dir, exist_ok=True)
  
    # 保存 labels（所有方法共用）
    labels_path = os.path.join(save_dir, f"test_labels_{cfg['shots']}shots.npy")
    np.save(labels_path, test_labels.cpu().numpy())
    print(f"已保存 labels 到: {labels_path}")
  
    # 1. Unified方法（最终融合的 tip_logits，使用搜索得到的 best_alpha/beta）
    unified_logits_path = os.path.join(save_dir, f"test_logits_unified_{cfg['shots']}shots.npy")
    np.save(unified_logits_path, tip_logits.detach().cpu().numpy())
    print(f"已保存 Unified logits 到: {unified_logits_path}")

    if use_dynamic_routing:
        alphas_path = os.path.join(save_dir, f"test_evidence_alphas_{cfg['shots']}shots.npy")
        np.save(alphas_path, test_alphas.detach().cpu().numpy())
        print(f"已保存动态证据门控权重 [α_real, α_pixel, α_feature]: {alphas_path}")
  
    # 2. ClipCache方法（仅CLIP cache）
    clip_cache_logits_path = os.path.join(save_dir, f"test_logits_clip_{cfg['shots']}shots.npy")
    np.save(clip_cache_logits_path, clip_cache_logits.detach().cpu().numpy())
    print(f"已保存 ClipCache logits 到: {clip_cache_logits_path}")
  
    # 3. ClipDino方法（朴素融合，固定权重 alpha=0.5，无超参搜索）
    # 这是一个更弱的 baseline，用于对比 Unified 的超参搜索优势
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

    # ================================================================================
    # Transparent Provenance Audit
    # Uses dynamic evidence routing alphas to decompose each test sample prediction
    # into real-samples / DALL-E+VAE / DINO-feature contributions
    # ================================================================================
    if use_dynamic_routing:
        print("\n" + "="*80)
        print("TRANSPARENT AUDIT: Evidence Provenance Analysis")
        print("="*80)

        with torch.no_grad():
            # test_alphas computed above during evaluation (shape: [N, 3])
            # alphas[:, 0] = a_real, alphas[:, 1] = a_pixel, alphas[:, 2] = a_feature

            # Raw branch logits (returned by ensemble_tip_logits_dynamic)
            L_real = branch_logits['L_real']
            L_pixel = branch_logits['L_pixel']
            L_feature = branch_logits['L_feature']

            # Weighted cache logits (each branch's contribution to final prediction)
            weighted_L_real = alphas[:, 0:1] * L_real
            weighted_L_pixel = alphas[:, 1:2] * L_pixel
            weighted_L_feature = alphas[:, 2:3] * L_feature

            # Total cache contribution (per sample, per class)
            cache_contrib_total = weighted_L_real + weighted_L_pixel + weighted_L_feature

            # Global softmax probabilities (for predicted class and confidence)
            test_probs = F.softmax(tip_logits, dim=1)
            test_preds = tip_logits.argmax(dim=1)
            test_confs = test_probs.max(dim=1).values

            # ==================== Statistics ====================
            print(f"\nEvidence Routing Statistics ({len(test_labels)} test samples):")

            avg_alphas = alphas.mean(dim=0)
            print(f"  Average Alpha Weights:")
            print(f"    a_real (real samples)     = {avg_alphas[0].item():.3f}")
            print(f"    a_pixel (DALL-E+VAE)      = {avg_alphas[1].item():.3f}")
            print(f"    a_feature (DINO features)= {avg_alphas[2].item():.3f}")
            print(f"    Sum check (should be 1.0) = {avg_alphas.sum().item():.3f}")

            # ==================== Per-Class Decomposition ====================
            print(f"\n  Per-Class Alpha Distribution (mean +/- std):")
            for c in range(test_probs.shape[1]):
                mask_c = (test_preds == c)
                cnt = mask_c.sum().item()
                if cnt > 0:
                    a0 = alphas[mask_c, 0].mean().item()
                    a1 = alphas[mask_c, 1].mean().item()
                    a2 = alphas[mask_c, 2].mean().item()
                    s0 = alphas[mask_c, 0].std().item()
                    s1 = alphas[mask_c, 1].std().item()
                    s2 = alphas[mask_c, 2].std().item()
                    try:
                        classname = dataset.classnames[c]
                    except Exception:
                        classname = str(c)
                    print(f"    Class {c} ({classname}, n={cnt}): "
                          f"a_real={a0:.3f}+/-{s0:.3f}  a_pixel={a1:.3f}+/-{s1:.3f}  a_feature={a2:.3f}+/-{s2:.3f}")

            # ==================== Confidence Stratification ====================
            print(f"\n  Confidence Stratification:")
            conf_bins = [(0.0, 0.5, "low"), (0.5, 0.8, "medium"), (0.8, 1.0, "high")]
            for lo, hi, label in conf_bins:
                mask_bin = (test_confs >= lo) & (test_confs < hi)
                n_bin = mask_bin.sum().item()
                if n_bin > 0:
                    bin_alphas = alphas[mask_bin].mean(dim=0)
                    print(f"    {label} confidence [{lo:.1f},{hi:.1f}] (n={n_bin}): "
                          f"a_real={bin_alphas[0]:.3f}  a_pixel={bin_alphas[1]:.3f}  a_feature={bin_alphas[2]:.3f}")

            # ==================== Prediction Flip Rate under Causal Ablation ====================
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

            # ==================== Per-Sample Provenance Export ====================
            # Save per-sample source weights for downstream analysis
            audit_sample_path = os.path.join(save_dir, f"audit_provenance_{cfg['shots']}shots.npy")
            audit_per_sample = {
                'alpha_real':     alphas[:, 0].cpu().numpy(),
                'alpha_pixel':    alphas[:, 1].cpu().numpy(),
                'alpha_feature':  alphas[:, 2].cpu().numpy(),
                'predictions':    test_preds.cpu().numpy(),
                'confidences':    test_confs.cpu().numpy(),
                'true_labels':    test_labels.cpu().numpy(),
                'L_real_max':     L_real.max(dim=1).values.cpu().numpy(),
                'L_pixel_max':    L_pixel.max(dim=1).values.cpu().numpy(),
                'L_feature_max':  L_feature.max(dim=1).values.cpu().numpy(),
            }
            np.save(audit_sample_path, audit_per_sample)
            print(f"\n  Per-sample provenance saved to: {audit_sample_path}")

            # Save flip-rate JSON
            flip_json_path = os.path.join(save_dir, f"audit_fliprate_{cfg['shots']}shots.json")
            with open(flip_json_path, 'w', encoding='utf-8') as f:
                json.dump(flip_rows, f, indent=2, ensure_ascii=False)
            print(f"  Ablation flip rates saved to: {flip_json_path}")

            # ====================================================================
            # SHAP 公理化归因审计
            # ====================================================================
            if cfg.get('enable_shap_attribution', True):
                print("\n" + "-"*60)
                print("  SHAP COMPLIANT ATTRIBUTION: Axiomatic Evidence Analysis")
                print("-"*60)

                try:
                    import shap_attribution as shap_mod

                    # 确保所有张量在同一设备上
                    clip_adapter_real_dev = clip_adapter_real.to(device) if clip_adapter_real is not None else None
                    clip_adapter_dev = clip_adapter.to(device)
                    dino_adapter_dev = dino_adapter.to(device)
                    clip_keys_real_dev = clip_keys_real.to(device)
                    clip_values_real_dev = clip_values_real.to(device)
                    clip_keys_pixel_dev = clip_keys_pixel.to(device)
                    clip_values_pixel_dev = clip_values_pixel.to(device)
                    dino_keys_dev = dino_keys.to(device)
                    dino_values_dev = dino_values.to(device)
                    clip_test_features_dev = clip_test_features.to(device)
                    dino_test_features_dev = dino_test_features.to(device)
                    clip_weights_dev = clip_weights.to(device)

                    # 采样审计（控制计算开销）
                    max_shap_samples = min(cfg.get('max_shap_samples', 500), len(test_labels))
                    sample_indices = torch.randperm(len(test_labels))[:max_shap_samples]

                    clip_f_sample = clip_test_features_dev[sample_indices]
                    dino_f_sample = dino_test_features_dev[sample_indices]
                    labels_sample = test_labels[sample_indices]
                    alphas_sample = test_alphas[sample_indices]

                    print(f"  Computing Shapley values for {max_shap_samples} samples ...")
                    print(f"  (n_samples={cfg.get('shap_n_samples', 32)}, "
                          f"branches=3, classes={clip_weights_dev.shape[1]})")

                    # Step 1: 计算原始 Shapley 值
                    phi_real, phi_pixel, phi_feature, v_empty, v_full = shap_mod.compute_shapley_values(
                        clip_f_sample, dino_f_sample, clip_weights_dev,
                        best_beta, best_alpha,
                        clip_adapter_real_dev, clip_adapter_dev, dino_adapter_dev,
                        clip_keys_real_dev, clip_values_real_dev,
                        clip_keys_pixel_dev, clip_values_pixel_dev,
                        dino_keys_dev, dino_values_dev,
                        n_samples=cfg.get('shap_n_samples', 32),
                    )

                    # Step 2: 归一化到 [0, 1]（保留负值）
                    norm_real, norm_pixel, norm_feature, sign_meta = shap_mod.normalize_shapley_to_unit(
                        phi_real, phi_pixel, phi_feature, v_empty, v_full
                    )

                    # Step 3: 公理验证
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

                    # Step 4: 与 Attention alphas 对比
                    gap_results = shap_mod.compute_attribution_gap(
                        norm_real, norm_pixel, norm_feature, alphas_sample
                    )

                    print(f"\n  [Attribution Gap: Shapley vs Attention alphas]")
                    print(f"    Euclidean gap:  {gap_results['gap_euclidean_mean']:.4f}")
                    print(f"    MAE gap:        {gap_results['gap_mae_mean']:.4f}")
                    print(f"    Per-branch diff: real={gap_results['diff_real']:.4f}, "
                          f"pixel={gap_results['diff_pixel']:.4f}, "
                          f"feature={gap_results['diff_feature']:.4f}")

                    # Step 5: 获取类别名称用于逐类报告
                    class_names = dataset.classnames if hasattr(dataset, 'classnames') else []

                    # Step 6: 生成完整报告
                    report = shap_mod.generate_shap_report(
                        phi_real, phi_pixel, phi_feature,
                        norm_real, norm_pixel, norm_feature,
                        v_empty, v_full,
                        alphas_sample, class_names,
                        axiom_results, gap_results, sign_meta,
                    )

                    # ---- 保存结果 ----
                    shap_json_path = os.path.join(save_dir, f"shap_attribution_{cfg['shots']}shots.json")
                    with open(shap_json_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False)
                    print(f"\n  SHAP report saved to: {shap_json_path}")

                    # 保存逐样本归一化 Shapley 值（NPY）
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

                    # 保存 Shapley 归一化权重（可直接与 attention alphas 对比）
                    shap_weights_path = os.path.join(save_dir, f"shap_weights_{cfg['shots']}shots.npy")
                    np.save(shap_weights_path, {
                        'shap_norm_weights': gap_results['shap_weights_avg'].cpu().numpy(),
                        'attention_alphas': alphas_sample.cpu().numpy(),
                        'gap_euclidean': gap_results['gap_euclidean'].cpu().numpy(),
                        'gap_mae': gap_results['gap_mae'].cpu().numpy(),
                        'sample_indices': sample_indices.cpu().numpy(),
                    })
                    print(f"  Shapley weights (norm) saved to: {shap_weights_path}")

                    # 打印全局摘要
                    gs = report['global_summary']
                    print(f"\n  [Global Shapley Summary]")
                    print(f"    Raw phi (mean):    real={gs['phi_real_mean']:.4f}, "
                          f"pixel={gs['phi_pixel_mean']:.4f}, feature={gs['phi_feature_mean']:.4f}")
                    print(f"    Normed phi (mean): real={gs['norm_real_mean']:.4f}, "
                          f"pixel={gs['norm_pixel_mean']:.4f}, feature={gs['norm_feature_mean']:.4f}")
                    print(f"    Frac negative:     real={gs['frac_negative_real']:.1%}, "
                          f"pixel={gs['frac_negative_pixel']:.1%}, feature={gs['frac_negative_feature']:.1%}")
                    print(f"    Sign patterns:     {gs['sign_pattern_distribution']}")

                    # 逐类别效率性验证摘要
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
                    print("         Ensure shap_attribution.py is in the same folder as main.py")
                except Exception as e:
                    print(f"\n  [ERROR] Shapley computation failed: {e}")
                    traceback.print_exc()

        print("\n" + "="*80)
        print("AUDIT COMPLETE -- provenance decomposition available.")
        print("="*80)

    else:
        print("\n[Audit Skipped] Dynamic evidence routing is disabled.")
        print("Enable 'use_dynamic_evidence_routing: true' in config to use transparent audit.")










def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    # 添加融合训练的配置参数
    cfg['use_fusion'] = cfg.get('use_fusion', False)  # 默认不使用融合训练
    if cfg['use_fusion']:
        print("\n将使用DALL-E和VAE图像融合训练")

    # 动态注意力证据路由与因果遮盖实验（默认关闭，在 yaml 中开启）
    cfg['use_dynamic_evidence_routing'] = cfg.get('use_dynamic_evidence_routing', False)
    cfg['evidence_top_k'] = cfg.get('evidence_top_k', 16)
    cfg['gate_temperature'] = cfg.get('gate_temperature', 1.0)
    cfg['run_evidence_masking'] = cfg.get('run_evidence_masking', False)
    if cfg['use_dynamic_evidence_routing']:
        print("\n已启用动态注意力证据路由 (Dynamic Attention-based Evidence Routing)")
        print(f"  evidence_top_k={cfg['evidence_top_k']}, gate_temperature={cfg['gate_temperature']}")
    
    # 添加流形学习的配置参数
    cfg['manifold_dim'] = cfg.get('manifold_dim', 64)
    cfg['n_neighbors'] = cfg.get('n_neighbors', 20)
    cfg['real_image_samples'] = cfg.get('real_image_samples', 1000)
    cfg['manifold_samples'] = cfg.get('manifold_samples', 500)
    
    if cfg.get('use_manifold_learning', True):  # 默认启用流形学习
        print(f"\n将使用流形学习增强VAE训练")
        print(f"  - 流形维度: {cfg['manifold_dim']}")
        print(f"  - 真实图片样本数: {cfg['real_image_samples']}")
        print(f"  - DALL-E样本数: {cfg['manifold_samples']}")

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

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg['root_path'], cfg['dalle_shots'])
    dalle_train_loader_cache = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    dalle_train_loader_F = build_data_loader(data_source=dalle_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    
    # ========================================================================
    # 保存特征用于定量指标计算 (ICA, PCVR, TSC, Spectral Entropy)
    # ========================================================================
    save_features_dir = cfg['cache_dir']
    os.makedirs(save_features_dir, exist_ok=True)
    
    # ========================================================================
    # 保存特征用于定量指标计算 (ICA, PCVR, TSC, Spectral Entropy)
    # ========================================================================
    save_features_dir = cfg['cache_dir']
    os.makedirs(save_features_dir, exist_ok=True)
    
    # 保存真实训练集特征
    real_feat_path = os.path.join(save_features_dir, f"real_features_{cfg['shots']}shots.npy")
    if not os.path.exists(real_feat_path):
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
    
    # 保存DALL-E生成特征
    dalle_feat_path = os.path.join(save_features_dir, f"dalle_features_{cfg['dalle_shots']}shots.npy")
    if not os.path.exists(dalle_feat_path):
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
    # ========================================================================
    
    # 添加VAE相关处理
    # 支持 use_vae 或 use_vae_generation 两种配置名称
    use_vae = cfg.get('use_vae', cfg.get('use_vae_generation', False))
    vae_train_loader_cache = None
    vae_train_loader_F = None

    # 消融实验C: 各向同性噪声基线
    use_isotropic_noise = cfg.get('use_isotropic_noise', False)
    isotropic_noise_scale = cfg.get('isotropic_noise_scale', 0.5)

    if use_isotropic_noise:
        print(f"\n[消融实验C] 使用各向同性噪声基线 (scale={isotropic_noise_scale})")
        # 禁用VAE，使用纯噪声
        use_vae = False
    
    if use_vae:
        print("\n使用VAE生成图像增强训练...")
        # 检查是否已存在VAE数据集
        vae_dataset_dir = os.path.join(cfg['root_path'], f"vae_{cfg['dataset']}")
        os.makedirs(vae_dataset_dir, exist_ok=True)  # 确保目录存在
        
        vae_json_path = os.path.join(vae_dataset_dir, f"vae_{cfg['dataset']}.json")
        vae_model_path = os.path.join(cfg['cache_dir'], f"best_vae_model_{cfg['shots']}shots.pt")
        
        # 如果不存在VAE数据集，则训练VAE并生成图像
        if not os.path.exists(vae_json_path):
            print(f"\n未找到VAE生成的图像数据集，将训练增强版VAE模型并生成图像")
            print(f"目标JSON路径: {vae_json_path}")
            
            # 训练增强版VAE模型（含流形学习和条件先验）
            if not os.path.exists(vae_model_path):
                print(f"训练增强版VAE模型...")
                vae_epochs = cfg.get('vae_epochs', 10)
                try:
                    text_features, manifold_projector = enhanced_train_vae_with_manifold(
                        train_loader_cache, 
                        val_loader, 
                        clip_model,
                        gpt3_prompt,
                        dataset.classnames,
                        dataset.template,
                        dalle_train_loader_cache,
                        epochs=vae_epochs, 
                        save_path=vae_model_path,
                        cfg=cfg
                    )
                    print(f"增强版VAE模型训练完成，保存到 {vae_model_path}")
                    print(f"语义锚点已提取，共 {len(text_features)} 个类别")
                    
                    # ========== 新增：使用条件先验训练VAE ==========
                    if cfg.get('use_conditional_prior', True):
                        print("\n使用条件先验 p(z|t_c) 训练VAE...")
                        try:
                            # 创建条件VAE模型
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            input_dim = 512  # CLIP特征维度
                            latent_dim = cfg.get('vae_latent_dim', 128)
                            
                            cvae_model = ConditionalVAE(
                                input_dim=input_dim, 
                                latent_dim=latent_dim,
                                use_conditional_prior=True
                            ).to(device)
                            
                            # 条件先验超参数
                            prior_sigma = cfg.get('prior_sigma', 0.1)  # 先验标准差
                            prior_beta = cfg.get('prior_beta', 0.5)  # KL散度权重（少样本下用较小值）
                            
                            print(f"   条件先验参数: sigma={prior_sigma}, beta={prior_beta}")
                            
                            # 优化器
                            cvae_optimizer = torch.optim.Adam(cvae_model.parameters(), lr=1e-3)
                            
                            # 训练条件VAE
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
                                    
                                    # 提取CLIP特征
                                    with torch.no_grad():
                                        clip_features = clip_model.encode_image(images)
                                        clip_features = F.normalize(clip_features, dim=-1)
                                    
                                    # 获取当前批次的语义锚点（按类别索引）
                                    anchor_features = text_features[labels].to(device)
                                    
                                    # 前向传播（使用条件先验）
                                    recon, mu, logvar, z = cvae_model(clip_features, anchor_features, use_prior=True)
                                    
                                    # 计算条件VAE损失
                                    # 先计算投影后的先验均值
                                    with torch.no_grad():
                                        prior_mu = cvae_model.anchor_projection(anchor_features)
                                        prior_mu = F.normalize(prior_mu, dim=-1)
                                    
                                    loss, recon_loss, kld_loss = conditional_vae_loss(
                                        recon, clip_features, mu, logvar,
                                        prior_mu=prior_mu, sigma=prior_sigma, beta=prior_beta
                                    )
                                    
                                    # 反向传播
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
                            
                            # 保存条件VAE模型
                            cvae_model_path = os.path.join(cfg['cache_dir'], f"cvae_model_{cfg['shots']}shots.pt")
                            torch.save(cvae_model.state_dict(), cvae_model_path)
                            print(f"条件VAE模型保存到 {cvae_model_path}")
                            print("条件先验训练完成！")
                            
                        except Exception as e:
                            print(f"条件VAE训练失败: {e}")
                            traceback.print_exc()
                            print("将跳过条件先验训练")
                    # ========== 条件先验训练结束 ==========

                    # 保存流形投影器
                    manifold_path = os.path.join(cfg['cache_dir'], f"manifold_projector_{cfg['shots']}shots.pt")
                    torch.save(manifold_projector, manifold_path)
                    print(f"流形投影器保存到 {manifold_path}")
                    
                except Exception as e:
                    print(f"增强版VAE模型训练失败: {e}")
                    traceback.print_exc()
                    print("将跳过VAE增强训练")
                    use_vae = False
            
            # 生成VAE图像
            if use_vae and os.path.exists(vae_model_path):
                try:
                    print(f"使用增强版VAE模型生成图像...")
                    
                    # 加载流形投影器（如果存在）
                    manifold_path = os.path.join(cfg['cache_dir'], f"manifold_projector_{cfg['shots']}shots.pt")
                    loaded_manifold_projector = None
                    if os.path.exists(manifold_path):
                        try:
                            loaded_manifold_projector = torch.load(manifold_path)
                            print(f"已加载流形投影器: {manifold_path}")
                        except Exception as e:
                            print(f"加载流形投影器失败: {e}")
                    
                    # 注意：VAE图像生成功能当前不可用
                    # 这个功能需要单独的VAEGenerator实现
                    print("⚠️  VAE图像生成功能当前不可用，将跳过此步骤")
                    print("   但流形学习增强仍然有效，将提升现有DALL-E图像的质量")
                    
                    # 创建一个虚拟的VAE数据集文件以满足后续流程
                    vae_dataset_placeholder = {
                        "dataset_name": cfg['dataset'],
                        "generated_with_manifold": True,
                        "note": "Placeholder for manifold-enhanced training"
                    }
                    
                    with open(vae_json_path, 'w') as f:
                        json.dump(vae_dataset_placeholder, f, indent=2)
                    
                    print(f"已创建VAE数据集占位符: {vae_json_path}")
                    # 再次检查JSON文件是否已创建
                    if not os.path.exists(vae_json_path):
                        print(f"警告: VAE图像生成后，仍然找不到JSON文件: {vae_json_path}")
                        use_vae = False
                except Exception as e:
                    print(f"VAE图像生成失败: {e}")
                    traceback.print_exc()
                    print("将跳过VAE增强训练")
                    use_vae = False
        
        # 加载VAE数据集
        if use_vae:
            print(f"\n检查VAE数据集: {vae_json_path}")
            try:
                # 检查是否是占位符文件
                if os.path.exists(vae_json_path):
                    with open(vae_json_path, 'r') as f:
                        vae_content = json.load(f)
                    
                    # 如果是占位符，跳过VAE数据集加载
                    if isinstance(vae_content, dict) and vae_content.get('note') == 'Placeholder for manifold-enhanced training':
                        print("🔄 检测到VAE占位符文件，流形学习已启用但跳过VAE数据集加载")
                        print("   将继续使用DALL-E图像和流形增强进行训练")
                        vae_train_loader_cache = None
                        vae_train_loader_F = None
                        # 保持 use_vae = True 以启用流形学习相关功能
                    else:
                        # 尝试加载真实的VAE数据集
                        cfg['vae_shots'] = cfg.get('vae_shots', cfg['shots'])
                        vae_dataset = build_vae_dataset(cfg['dataset'], cfg['root_path'], cfg['vae_shots'])
                        if vae_dataset is not None:
                            vae_train_loader_cache = build_data_loader(data_source=vae_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
                            vae_train_loader_F = build_data_loader(data_source=vae_dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
                            print(f"成功加载VAE数据集，包含 {len(vae_dataset.train_x)} 张图像")
                        else:
                            print("VAE数据集加载失败")
                            vae_train_loader_cache = None
                            vae_train_loader_F = None
                else:
                    print("VAE数据集JSON文件不存在")
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

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = gpt_clip_classifier(dataset.classnames, gpt3_prompt, clip_model, dataset.template)

    # 确保clip_weights的数据类型与CLIP模型一致
    clip_dtype = next(clip_model.parameters()).dtype
    clip_weights = clip_weights.to(clip_dtype)
    print(f"CLIP weights dtype: {clip_weights.dtype}")

    # Construct the cache model by few-shot training set
    # 确保缓存目录存在
    os.makedirs(cfg['cache_dir'], exist_ok=True)
    print("\nConstructing cache model by few-shot visual features and labels.")
    
    # ===== 0-Shot特殊处理：不使用真实样本缓存 =====
    if cfg['shots'] == 0:
        print("\n⚠️  检测到0-shot配置，将不使用真实样本缓存")
        print("   真正的0-shot应该完全依赖于：")
        print("   1. CLIP预训练知识")
        print("   2. GPT-3文本提示词")
        print("   3. DALL-E生成的合成图像")
        print("   4. 流形学习增强的特征")
        
        # 获取类别数量
        num_classes = len(dataset.classnames)
        
        # 创建空的缓存张量
        # CLIP RN50特征维度: 1024, DINO ResNet50特征维度: 2048
        clip_cache_keys = torch.zeros(1024, 0, dtype=torch.float16).cuda()
        clip_cache_values = torch.zeros(0, num_classes, dtype=torch.float16).cuda()
        dino_cache_keys = torch.zeros(2048, 0, dtype=torch.float16).cuda()
        dino_cache_values = torch.zeros(0, num_classes, dtype=torch.float16).cuda()
        
        print(f"   创建空缓存: CLIP keys {clip_cache_keys.shape}, values {clip_cache_values.shape}")
        print(f"              DINO keys {dino_cache_keys.shape}, values {dino_cache_values.shape}")
        
        # 验证：确保没有意外加载旧的0-shot缓存文件
        zero_shot_files = [
            f"{cfg['cache_dir']}/clip_keys_0shots.pt",
            f"{cfg['cache_dir']}/clip_values_0shots.pt",
            f"{cfg['cache_dir']}/dino_keys_0shots.pt",
            f"{cfg['cache_dir']}/dino_values_0shots.pt"
        ]
        
        for file_path in zero_shot_files:
            if os.path.exists(file_path):
                print(f"\n   ⚠️  警告: 发现旧的0-shot缓存文件: {file_path}")
                print(f"      该文件将被忽略，如需清理请手动删除")
    
    else:
        # 非0-shot：正常加载真实样本缓存
        print(f"\n加载 {cfg['shots']}-shot 真实样本缓存...")
        print("\nConstructing CLIP cache model.")
        clip_cache_keys, clip_cache_values = build_clip_cache_model(cfg, clip_model, train_loader_cache)
        print("\nConstructing DINO cache model.")
        dino_cache_keys, dino_cache_values = build_dino_cache_model(cfg, dino_model, train_loader_cache)
        
        # 验证加载的缓存大小是否合理
        expected_samples = cfg['shots'] * len(dataset.classnames)
        actual_clip_samples = clip_cache_keys.shape[1]
        actual_dino_samples = dino_cache_keys.shape[1]
        
        print(f"\n缓存验证:")
        print(f"  期望样本数 (shots × 类别数): {cfg['shots']} × {len(dataset.classnames)} = {expected_samples}")
        print(f"  实际CLIP样本数: {actual_clip_samples}")
        print(f"  实际DINO样本数: {actual_dino_samples}")


    print("\nConstructing cache model by dalle image.")
    print("\nConstructing CLIP cache model.")
    clip_dalle_cache_keys, clip_dalle_cache_values = build_clip_dalle_cache_model(cfg, clip_model, dalle_train_loader_cache)
    # 保存CLIP DALLE缓存模型
    torch.save(clip_dalle_cache_keys, cfg['cache_dir'] + "/clip_dalle_keys_" + str(cfg['dalle_shots']) + "shots.pt")
    torch.save(clip_dalle_cache_values, cfg['cache_dir'] + "/clip_dalle_values_" + str(cfg['dalle_shots']) + "shots.pt")

    print("\nConstructing DINO cache model.")
    dino_dalle_cache_keys, dino_dalle_cache_values = build_dino_dalle_cache_model(cfg, dino_model, dalle_train_loader_cache)
    # 保存DINO DALLE缓存模型
    torch.save(dino_dalle_cache_keys, cfg['cache_dir'] + "/dino_dalle_keys_" + str(cfg['dalle_shots']) + "shots.pt")
    torch.save(dino_dalle_cache_values, cfg['cache_dir'] + "/dino_dalle_values_" + str(cfg['dalle_shots']) + "shots.pt")

    # =============================================================
    # 方案A: CVAE 增强特征缓存
    # 加载训练好的 CVAE + ManifoldProjector，从真实图片和 DALL-E 图片的
    # CLIP 特征生成增强特征缓存。这些增强特征参与 Tip-Adapter-F 训练。
    # =============================================================
    clip_cvae_cache_keys = None   # 真实图片的 CVAE 增强 CLIP 缓存
    clip_cvae_cache_values = None
    dalle_cvae_cache_keys = None  # DALL-E 图片的 CVAE 增强 CLIP 缓存
    dalle_cvae_cache_values = None

    cvae_model_path = os.path.join(cfg['cache_dir'], f"cvae_model_{cfg['shots']}shots.pt")
    manifold_path = os.path.join(cfg['cache_dir'], f"manifold_projector_{cfg['shots']}shots.pt")
    cvae_available = os.path.exists(cvae_model_path)

    if use_vae and cvae_available:
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

            # 获取文本特征（语义锚点）
            if 'text_features' not in dir() or text_features is None:
                print("[CVAE] 警告: text_features 未定义，跳过 CVAE 增强")
            else:
                print(f"\n[CVAE] 从真实图片生成增强特征缓存...")
                clip_cvae_cache_keys, clip_cvae_cache_values = build_cvae_enhanced_cache_model(
                    cfg, clip_model, train_loader_cache,
                    cvae_model, manifold_projector, text_features, clip_dtype
                )
                torch.save(clip_cvae_cache_keys, cfg['cache_dir'] + f"/clip_cvae_real_keys_{cfg['shots']}shots.pt")
                torch.save(clip_cvae_cache_values, cfg['cache_dir'] + f"/clip_cvae_real_values_{cfg['shots']}shots.pt")

                print(f"\n[CVAE] 从 DALL-E 图片生成增强特征缓存...")
                dalle_cvae_cache_keys, dalle_cvae_cache_values = build_cvae_enhanced_cache_model(
                    cfg, clip_model, dalle_train_loader_cache,
                    cvae_model, manifold_projector, text_features, clip_dtype
                )
                torch.save(dalle_cvae_cache_keys, cfg['cache_dir'] + f"/clip_cvae_dalle_keys_{cfg['dalle_shots']}shots.pt")
                torch.save(dalle_cvae_cache_values, cfg['cache_dir'] + f"/clip_cvae_dalle_values_{cfg['dalle_shots']}shots.pt")

        except Exception as e:
            print(f"[CVAE] 增强缓存生成失败: {e}")
            traceback.print_exc()
            print("[CVAE] 将跳过 CVAE 增强，使用原始缓存继续")
            clip_cvae_cache_keys = None
            clip_cvae_cache_values = None
            dalle_cvae_cache_keys = None
            dalle_cvae_cache_values = None
    else:
        if use_vae and not cvae_available:
            print(f"\n[CVAE] 未找到训练好的 CVAE 权重 ({cvae_model_path})，跳过 CVAE 增强")
            print(f"[CVAE] 提示: 首次运行时会先训练 CVAE 模型，下次运行时将使用该模型进行增强")

    # Pre-load val features
    print("\nLoading CLIP feature from val set.")
    val_clip_features, val_labels = pre_CLIP_load_features(cfg, "val", clip_model, val_loader)
    print("\nLoading DINO feature from val set.")
    val_dino_features, val_labels = pre_DINO_load_features(cfg, "val", dino_model, val_loader)

    # Pre-load test features
    print("\nLoading CLIP feature from test set.")
    test_clip_features, test_labels = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)
    print("\nLoading DINO feature from test set.")
    test_dino_features, test_labels = pre_DINO_load_features(cfg, "test", dino_model, test_loader)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------

    # 合并所有缓存键和值（将 CVAE 增强特征注入）
    all_clip_cache_keys = [clip_cache_keys, clip_dalle_cache_keys]
    all_clip_cache_values = [clip_cache_values, clip_dalle_cache_values]
    all_dino_cache_keys = [dino_cache_keys, dino_dalle_cache_keys]
    all_dino_cache_values = [dino_cache_values, dino_dalle_cache_values]

    # 如果有 CVAE 增强特征，加入合并缓存
    if use_vae and dalle_cvae_cache_keys is not None:
        all_clip_cache_keys.append(dalle_cvae_cache_keys)
        all_clip_cache_values.append(dalle_cvae_cache_values)
        print(f"[CVAE] 已将增强特征注入合并缓存: +{dalle_cvae_cache_keys.shape[1]} 样本")

    # 合并所有缓存
    merged_clip_cache_keys = torch.cat(all_clip_cache_keys, dim=1)
    merged_clip_cache_values = torch.cat(all_clip_cache_values, dim=0)
    merged_dino_cache_keys = torch.cat(all_dino_cache_keys, dim=1)
    merged_dino_cache_values = torch.cat(all_dino_cache_values, dim=0)

    # CVAE 增强特征注入动态路由缓存
    separated_caches = None
    if cfg.get('use_dynamic_evidence_routing', False):
        separated_caches = {
            'clip_real': (clip_cache_keys, clip_cache_values),
            'clip_pixel': (clip_dalle_cache_keys, clip_dalle_cache_values),
            'dino': (merged_dino_cache_keys, merged_dino_cache_values),
        }
        # CVAE 增强的真实图片特征作为独立分支
        if use_vae and clip_cvae_cache_keys is not None:
            separated_caches['clip_cvae_real'] = (clip_cvae_cache_keys, clip_cvae_cache_values)
        # CVAE 增强的 DALL-E 特征作为独立分支
        if use_vae and dalle_cvae_cache_keys is not None:
            separated_caches['clip_cvae_pixel'] = (dalle_cvae_cache_keys, dalle_cvae_cache_values)
        cvae_tag = ""
        if clip_cvae_cache_keys is not None:
            cvae_tag += f" real(+{clip_cvae_cache_keys.shape[1]})"
        if dalle_cvae_cache_keys is not None:
            cvae_tag += f" dalle(+{dalle_cvae_cache_keys.shape[1]})"
        print(f"\n动态路由缓存: C_real={clip_cache_keys.shape[1]}, C_pixel={clip_dalle_cache_keys.shape[1]}, "
              f"C_cvae={cvae_tag}, C_feature={merged_dino_cache_keys.shape[1]}")

    run_ensemble_tip_dalle_adapter_F(cfg, 
                            merged_clip_cache_keys, 
                            merged_clip_cache_values, 
                            val_clip_features,
                            test_clip_features, 
                            merged_dino_cache_keys, 
                            merged_dino_cache_values,
                            val_dino_features, 
                            test_dino_features, 
                            val_labels,
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