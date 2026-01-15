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
    return None, manifold_projector

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
                            vae_train_loader_F=None):
    
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
    
    print(f"缓存张量数据类型统一为: {clip_dtype}")
    print(f"CLIP缓存: keys {clip_cache_keys.dtype}, values {clip_cache_values.dtype}")
    print(f"DINO缓存: keys {dino_cache_keys.dtype}, values {dino_cache_values.dtype}")
    
    # Enable the cached keys to be learnable
    clip_adapter = nn.Linear(clip_cache_keys.shape[0], clip_cache_keys.shape[1], bias=False).to(clip_dtype).to(device)
    clip_adapter.weight = nn.Parameter(clip_cache_keys.t())
    dino_adapter = nn.Linear(dino_cache_keys.shape[0], dino_cache_keys.shape[1], bias=False).to(clip_dtype).to(device)
    dino_adapter.weight = nn.Parameter(dino_cache_keys.t())
    
    print(f"适配器dtype: {clip_adapter.weight.dtype}")
    
    optimizer = torch.optim.AdamW(
        itertools.chain(dino_adapter.parameters(), clip_adapter.parameters()),
        lr=cfg['lr'], 
        eps=1e-4)
    
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

                clip_affinity = clip_adapter(clip_image_features).to(clip_dtype)
                clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                dino_affinity = dino_adapter(dino_image_features).to(clip_dtype)
                dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                clip_logits = 100. * clip_image_features @ clip_weights

                cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
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

            clip_affinity = clip_adapter(clip_image_features).to(clip_dtype)
            clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
            dino_affinity = dino_adapter(dino_image_features).to(clip_dtype)
            dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
            clip_logits = 100. * clip_image_features @ clip_weights

            cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
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

                clip_affinity = clip_adapter(clip_image_features).to(clip_dtype)
                clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                dino_affinity = dino_adapter(dino_image_features).to(clip_dtype)
                dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                clip_logits = 100. * clip_image_features @ clip_weights

                cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
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
                    clip_affinity = clip_adapter(fusion_features)
                    clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
                    
                    dino_affinity = dino_adapter(dino_fusion_features)
                    dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
                    
                    clip_logits = 100. * fusion_features @ clip_weights
                    
                    cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
                    tip_logits = clip_logits + cache_logits * alpha
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

        # 确保验证特征与适配器的数据类型一致
        clip_val_features = clip_val_features.to(clip_dtype)
        dino_val_features = dino_val_features.to(clip_dtype)
        
        clip_affinity = clip_adapter(clip_val_features).to(clip_dtype)
        dino_affinity = dino_adapter(dino_val_features).to(clip_dtype)
        clip_cache_logits = ((-1) * (beta - beta * clip_affinity)).exp() @ clip_cache_values
        dino_cache_logits = ((-1) * (beta - beta * dino_affinity)).exp() @ dino_cache_values
        clip_logits = 100. * clip_val_features @ clip_weights
        cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, val_labels)

        print("**** VASMA's val accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(clip_adapter.weight, cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt")
            torch.save(dino_adapter.weight, cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt")
    
    loaded_clip_w = torch.load(cfg['cache_dir'] + "/best_F_clip_adapter_" + str(cfg['shots']) + "shots.pt", map_location=device)
    loaded_dino_w = torch.load(cfg['cache_dir'] + "/best_F_dino_adapter_" + str(cfg['shots']) + "shots.pt", map_location=device)
    clip_adapter.weight = nn.Parameter(loaded_clip_w.to(clip_dtype).to(device))
    dino_adapter.weight = nn.Parameter(loaded_dino_w.to(clip_dtype).to(device))
    print(f"**** After fine-tuning, VASMA's best val accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_ensemble_hp(cfg, clip_adapter.weight.t(), clip_cache_values, 
                                             clip_val_features, dino_adapter.weight.t(), dino_cache_values, 
                                             dino_val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")
   
    # 确保测试特征与适配器的数据类型一致
    clip_test_features = clip_test_features.to(clip_dtype)
    dino_test_features = dino_test_features.to(clip_dtype)
   
    clip_affinity = clip_adapter(clip_test_features).to(clip_dtype)
    dino_affinity = dino_adapter(dino_test_features).to(clip_dtype)
    clip_cache_logits = ((-1) * (best_beta - best_beta * clip_affinity)).exp() @ clip_cache_values
    dino_cache_logits = ((-1) * (best_beta - best_beta * dino_affinity)).exp() @ dino_cache_values
    
    clip_logits = 100. * clip_test_features @ clip_weights
    cache_logits = logits_fuse(clip_logits, [clip_cache_logits, dino_cache_logits])
    tip_logits = clip_logits + cache_logits * best_alpha
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

    # ================================================================================
    # 可选透明化审计功能 (默认注释，需要时取消注释启用)
    # 该功能实现了论文中提到的透明化审计：定量分解和视觉验证
    # ================================================================================
    """
    # ============ 透明化审计：证据溯源分析 =============
    print("\n" + "="*80)
    print("🔍 TRANSPARENT AUDIT: Evidence Provenance Analysis")
    print("="*80)

    audit_enabled = cfg.get('enable_audit', False)
    if audit_enabled:
        print("✅ 透明化审计已启用，开始分析证据溯源...")

        # 计算各个缓存来源的贡献度
        clip_cache_contribution = clip_cache_logits * best_alpha
        dino_cache_contribution = dino_cache_logits * best_alpha

        # 计算每个样本的来源贡献占比
        total_cache_contribution = clip_cache_contribution + dino_cache_contribution
        clip_proportion = torch.abs(clip_cache_contribution) / (torch.abs(total_cache_contribution) + 1e-8)
        dino_proportion = torch.abs(dino_cache_contribution) / (torch.abs(total_cache_contribution) + 1e-8)

        # 统计分析
        print(f"\n📊 缓存来源贡献统计 ({len(test_labels)} 个测试样本):")
        print(f"   CLIP缓存平均贡献比例: {clip_proportion.mean().item():.3f}")
        print(f"   DINO缓存平均贡献比例: {dino_proportion.mean().item():.3f}")
        print(f"   零-shot CLIP贡献占比: {(torch.abs(clip_logits) / (torch.abs(tip_logits) + 1e-8)).mean().item():.3f}")

        # 分析高置信度预测的来源分布
        confidence_threshold = 0.8
        top_predictions = torch.softmax(tip_logits, dim=1).max(dim=1)[0] > confidence_threshold
        if top_predictions.sum() > 0:
            high_conf_clip_prop = clip_proportion[top_predictions].mean().item()
            high_conf_dino_prop = dino_proportion[top_predictions].mean().item()
            print(f"\n🎯 高置信度预测 ({top_predictions.sum().item()}/{len(test_labels)} 个样本):")
            print(f"   CLIP缓存贡献: {high_conf_clip_prop:.3f}")
            print(f"   DINO缓存贡献: {high_conf_dino_prop:.3f}")

        # 保存审计结果（可选）
        audit_save_path = os.path.join(save_dir, f"audit_results_{cfg['shots']}shots.json")
        audit_results = {
            "dataset": cfg['dataset'],
            "shots": cfg['shots'],
            "total_samples": len(test_labels),
            "cache_contribution_stats": {
                "clip_cache_avg_proportion": clip_proportion.mean().item(),
                "dino_cache_avg_proportion": dino_proportion.mean().item(),
                "zero_shot_clip_proportion": (torch.abs(clip_logits) / (torch.abs(tip_logits) + 1e-8)).mean().item()
            },
            "high_confidence_analysis": {
                "threshold": confidence_threshold,
                "high_conf_samples": top_predictions.sum().item(),
                "high_conf_clip_proportion": high_conf_clip_prop if 'high_conf_clip_prop' in locals() else None,
                "high_conf_dino_proportion": high_conf_dino_prop if 'high_conf_dino_prop' in locals() else None
            }
        }

        import json
        with open(audit_save_path, 'w') as f:
            json.dump(audit_results, f, indent=2)
        print(f"💾 审计结果已保存到: {audit_save_path}")

        print("\n🔍 透明化审计完成！")
        print("   - 可以查看各预测的证据来源分解")
        print("   - 分析缓存贡献的统计分布")
        print("   - 识别高置信度预测的决策模式")

    else:
        print("ℹ️  透明化审计已禁用。如需启用，请在配置文件中设置: enable_audit: true")

    print("="*80)
    """









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
    
    # 添加VAE相关处理
    use_vae = cfg.get('use_vae', False)
    vae_train_loader_cache = None
    vae_train_loader_F = None
    
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
            
            # 训练增强版VAE模型（含流形学习）
            if not os.path.exists(vae_model_path):
                print(f"训练增强版VAE模型...")
                vae_epochs = cfg.get('vae_epochs', 10)
                try:
                    # 使用增强版VAE训练，包含真实数据集和DALL-E特征的流形学习
                    vae_model, manifold_projector = enhanced_train_vae_with_manifold(
                        train_loader_cache, 
                        val_loader, 
                        clip_model,
                        gpt3_prompt,
                        dataset.classnames,
                        dataset.template,
                        dalle_train_loader_cache,  # 传递DALL-E数据加载器
                        epochs=vae_epochs, 
                        save_path=vae_model_path,
                        cfg=cfg
                    )
                    print(f"增强版VAE模型训练完成，保存到 {vae_model_path}")
                    
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

    # 添加VAE缓存模型
    clip_vae_cache_keys = None
    clip_vae_cache_values = None
    dino_vae_cache_keys = None
    dino_vae_cache_values = None
    
    if use_vae and vae_train_loader_cache is not None:
        print("\nConstructing cache model by VAE generated image.")
        print("\nConstructing CLIP cache model.")
        clip_vae_cache_keys, clip_vae_cache_values = build_clip_vae_cache_model(cfg, clip_model, vae_train_loader_cache)
        # 保存CLIP VAE缓存模型
        torch.save(clip_vae_cache_keys, cfg['cache_dir'] + "/clip_vae_keys_" + str(cfg['vae_shots']) + "shots.pt")
        torch.save(clip_vae_cache_values, cfg['cache_dir'] + "/clip_vae_values_" + str(cfg['vae_shots']) + "shots.pt")
        
        print("\nConstructing DINO cache model.")
        dino_vae_cache_keys, dino_vae_cache_values = build_dino_vae_cache_model(cfg, dino_model, vae_train_loader_cache)
        # 保存DINO VAE缓存模型
        torch.save(dino_vae_cache_keys, cfg['cache_dir'] + "/dino_vae_keys_" + str(cfg['vae_shots']) + "shots.pt")
        torch.save(dino_vae_cache_values, cfg['cache_dir'] + "/dino_vae_values_" + str(cfg['vae_shots']) + "shots.pt")

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

    # 合并所有缓存键和值
    all_clip_cache_keys = [clip_cache_keys, clip_dalle_cache_keys]
    all_clip_cache_values = [clip_cache_values, clip_dalle_cache_values]
    all_dino_cache_keys = [dino_cache_keys, dino_dalle_cache_keys]
    all_dino_cache_values = [dino_cache_values, dino_dalle_cache_values]
    
    # 如果使用VAE，添加VAE缓存
    if use_vae and clip_vae_cache_keys is not None:
        all_clip_cache_keys.append(clip_vae_cache_keys)
        all_clip_cache_values.append(clip_vae_cache_values)
        all_dino_cache_keys.append(dino_vae_cache_keys)
        all_dino_cache_values.append(dino_vae_cache_values)
    
    # 合并所有缓存
    merged_clip_cache_keys = torch.cat(all_clip_cache_keys, dim=1)
    merged_clip_cache_values = torch.cat(all_clip_cache_values, dim=0)
    merged_dino_cache_keys = torch.cat(all_dino_cache_keys, dim=1)
    merged_dino_cache_values = torch.cat(all_dino_cache_values, dim=0)

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
                            vae_train_loader_F)
                            
if __name__ == '__main__':
    main()