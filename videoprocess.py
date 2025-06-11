import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torchaudio
import transformers
from transformers import BertTokenizer, BertModel
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 创建存储多模态数据的目录
os.makedirs('e:/projectpython/blog/data/videos/images', exist_ok=True)
os.makedirs('e:/projectpython/blog/data/videos/audio', exist_ok=True)
os.makedirs('e:/projectpython/blog/data/videos/text', exist_ok=True)

# 模拟视频数据
videos_metadata = [
    {'id': 'v001', 'title': '王者荣耀精彩集锦', 'description': '本视频展示了王者荣耀游戏中的精彩操作和团战瞬间，包含多位英雄的精彩表现。'},
    {'id': 'v002', 'title': '如何提高厨艺', 'description': '跟着专业厨师学习家常菜的烹饪技巧，从食材选择到最终摆盘的全过程讲解。'},
    {'id': 'v003', 'title': '2023科技产品评测', 'description': '详细评测最新的智能手机、平板电脑和智能手表等电子产品，分析其性能、外观和使用体验。'},
    {'id': 'v004', 'title': '健身30分钟训练', 'description': '专业健身教练带来的高效30分钟全身训练，不需要器械，适合在家进行。'},
    {'id': 'v005', 'title': '英雄联盟职业比赛', 'description': 'LPL春季赛精彩比赛回顾，顶尖战队的激烈对决和关键团战分析。'},
    {'id': 'v006', 'title': '日本东京之旅', 'description': '东京旅游vlog，探访当地美食、文化景点和购物中心，分享旅行小贴士。'},
    {'id': 'v007', 'title': 'Python基础教程', 'description': '从零开始学习Python编程，包括基础语法、数据结构和简单项目实践。'},
    {'id': 'v008', 'title': '家居装修设计指南', 'description': '室内设计师分享家居装修的关键要点，从空间规划到色彩搭配的专业建议。'},
    {'id': 'v009', 'title': '瑜伽初学者指南', 'description': '适合瑜伽初学者的基础动作教学，强调正确姿势和呼吸方法。'},
    {'id': 'v010', 'title': '最新动漫推荐与点评', 'description': '2023年春季新番动漫推荐，包括剧情简介、制作质量和个人评价。'},
    {'id': 'v011', 'title': '手机摄影技巧大全', 'description': '使用智能手机拍出专业级照片的技巧，包括构图、光线利用和后期编辑。'},
    {'id': 'v012', 'title': '经典文学作品赏析', 'description': '解读世界经典文学作品的主题、写作手法和历史背景，分享阅读心得。'},
    {'id': 'v013', 'title': '电影解说：好莱坞经典', 'description': '深度解析好莱坞经典电影的剧情、导演手法和演员表演，探讨其艺术价值。'},
    {'id': 'v014', 'title': '投资理财入门指南', 'description': '理财专家讲解基础投资知识，包括股票、基金和储蓄的优缺点分析。'},
    {'id': 'v015', 'title': '宠物猫咪护理知识', 'description': '猫咪日常护理、喂养和常见健康问题的解决方法，适合新手猫主人。'}
]

# 模拟视频的文本特征 (使用标题和描述)
for video in videos_metadata:
    text_data = {
        'title': video['title'],
        'description': video['description'],
        'transcript': f"这是{video['title']}的详细解说文字，包含了视频中的对话和解说内容。" * 3  # 模拟视频文字记录
    }
    with open(f"e:/projectpython/blog/data/videos/text/{video['id']}.json", 'w', encoding='utf-8') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=4)

# 模拟视频的图像特征 (使用随机生成的特征向量代表视频关键帧)
image_categories = {
    'v001': 'gaming', 'v002': 'cooking', 'v003': 'technology', 'v004': 'fitness',
    'v005': 'gaming', 'v006': 'travel', 'v007': 'education', 'v008': 'home',
    'v009': 'fitness', 'v010': 'animation', 'v011': 'photography', 'v012': 'literature',
    'v013': 'movie', 'v014': 'finance', 'v015': 'pets'
}

for video_id, category in image_categories.items():
    # 为每个视频生成5个关键帧的特征
    frames_features = []
    for i in range(5):
        # 根据类别生成有一定相似性的特征
        if category == 'gaming':
            base = np.array([0.8, 0.2, 0.1, 0.05, 0.7])
        elif category == 'cooking':
            base = np.array([0.1, 0.9, 0.3, 0.2, 0.1])
        elif category == 'technology':
            base = np.array([0.3, 0.2, 0.9, 0.4, 0.1])
        elif category == 'fitness':
            base = np.array([0.2, 0.1, 0.1, 0.9, 0.2])
        elif category == 'travel':
            base = np.array([0.4, 0.3, 0.1, 0.2, 0.8])
        elif category == 'education':
            base = np.array([0.6, 0.1, 0.7, 0.1, 0.2])
        elif category == 'home':
            base = np.array([0.2, 0.8, 0.2, 0.1, 0.3])
        elif category == 'animation':
            base = np.array([0.7, 0.1, 0.2, 0.1, 0.6])
        elif category == 'photography':
            base = np.array([0.3, 0.4, 0.2, 0.7, 0.1])
        elif category == 'literature':
            base = np.array([0.9, 0.1, 0.3, 0.1, 0.2])
        elif category == 'movie':
            base = np.array([0.7, 0.2, 0.4, 0.3, 0.5])
        elif category == 'finance':
            base = np.array([0.2, 0.7, 0.8, 0.1, 0.1])
        elif category == 'pets':
            base = np.array([0.1, 0.6, 0.1, 0.3, 0.8])
        else:
            base = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
            
        # 添加随机噪声
        noise = np.random.normal(0, 0.1, 5)
        feature = base + noise
        # 归一化
        feature = feature / np.linalg.norm(feature)
        frames_features.append(feature.tolist())
    
    # 保存特征
    with open(f"e:/projectpython/blog/data/videos/images/{video_id}.json", 'w') as f:
        json.dump(frames_features, f)

# 模拟视频的音频特征
audio_categories = {
    'v001': 'game_sounds', 'v002': 'cooking_instructions', 'v003': 'tech_review',
    'v004': 'workout_music', 'v005': 'game_commentary', 'v006': 'travel_ambience',
    'v007': 'educational_lecture', 'v008': 'home_improvement', 'v009': 'calm_music',
    'v010': 'anime_soundtrack', 'v011': 'photography_tips', 'v012': 'audiobook',
    'v013': 'movie_analysis', 'v014': 'finance_advice', 'v015': 'pet_sounds'
}

for video_id, category in audio_categories.items():
    # 为每个视频生成音频特征
    if 'game' in category:
        base = np.array([0.8, 0.3, 0.1, 0.7, 0.2])
    elif 'cooking' in category:
        base = np.array([0.2, 0.7, 0.4, 0.1, 0.3])
    elif 'tech' in category:
        base = np.array([0.4, 0.2, 0.8, 0.3, 0.1])
    elif 'workout' in category or 'fitness' in category:
        base = np.array([0.3, 0.1, 0.2, 0.9, 0.4])
    elif 'travel' in category:
        base = np.array([0.2, 0.6, 0.3, 0.1, 0.7])
    elif 'educational' in category:
        base = np.array([0.7, 0.2, 0.6, 0.1, 0.3])
    elif 'home' in category:
        base = np.array([0.3, 0.7, 0.1, 0.2, 0.4])
    elif 'calm' in category:
        base = np.array([0.1, 0.3, 0.1, 0.2, 0.9])
    elif 'anime' in category:
        base = np.array([0.6, 0.2, 0.3, 0.4, 0.7])
    elif 'photography' in category:
        base = np.array([0.5, 0.3, 0.7, 0.2, 0.1])
    elif 'audiobook' in category or 'literature' in category:
        base = np.array([0.8, 0.1, 0.2, 0.1, 0.3])
    elif 'movie' in category:
        base = np.array([0.6, 0.4, 0.3, 0.5, 0.2])
    elif 'finance' in category:
        base = np.array([0.4, 0.8, 0.3, 0.1, 0.2])
    elif 'pet' in category:
        base = np.array([0.2, 0.3, 0.1, 0.4, 0.8])
    else:
        base = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    # 添加随机噪声
    noise = np.random.normal(0, 0.1, 5)
    feature = base + noise
    # 归一化
    feature = feature / np.linalg.norm(feature)
    
    # 保存特征
    with open(f"e:/projectpython/blog/data/videos/audio/{video_id}.json", 'w') as f:
        json.dump(feature.tolist(), f)

# 定义标签集合
all_tags = [
    '游戏', 'MOBA', '竞技', '王者荣耀', '英雄联盟', '电竞',
    '美食', '烹饪', '教程', '家常菜',
    '科技', '评测', '手机', '电子产品',
    '健身', '运动', '训练', '减肥', '瑜伽', '放松', '初学者',
    '旅游', '日本', 'vlog', '东京', '文化',
    '编程', 'Python', '技术', '计算机',
    '家居', '装修', '设计', '生活', '室内',
    '动漫', '二次元', '娱乐', '推荐',
    '摄影', '技巧', '艺术',
    '文学', '阅读', '经典', '赏析',
    '电影', '解说', '好莱坞', '影评',
    '理财', '投资', '金融', '经济',
    '宠物', '猫咪', '护理', '萌宠'
]

# 多模态Transformer模型
class MultiModalTransformer(nn.Module):
    def __init__(self, text_dim=768, image_dim=5, audio_dim=5, hidden_dim=128, num_heads=4, num_layers=2, num_tags=len(all_tags)):
        super(MultiModalTransformer, self).__init__()
        
        # 文本特征处理
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        
        # 图像特征处理 (处理多个关键帧)
        self.image_fc = nn.Linear(image_dim, hidden_dim)
        
        # 音频特征处理
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)
        
        # 多模态融合Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 标签预测层
        self.tag_classifier = nn.Linear(hidden_dim, num_tags)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text_features, image_features, audio_features):
        # 处理文本特征
        text_embed = self.text_fc(text_features)  # [batch_size, hidden_dim]
        
        # 处理图像特征 (多个关键帧)
        batch_size, num_frames, _ = image_features.shape
        image_features_flat = image_features.reshape(batch_size * num_frames, -1)
        image_embed_flat = self.image_fc(image_features_flat)
        image_embed = image_embed_flat.reshape(batch_size, num_frames, -1)
        # 取平均值作为整体图像特征
        image_embed = torch.mean(image_embed, dim=1)  # [batch_size, hidden_dim]
        
        # 处理音频特征
        audio_embed = self.audio_fc(audio_features)  # [batch_size, hidden_dim]
        
        # 将三种模态特征拼接为序列
        multimodal_features = torch.stack([text_embed, image_embed, audio_embed], dim=1)  # [batch_size, 3, hidden_dim]
        
        # 通过Transformer进行多模态融合
        fused_features = self.transformer_encoder(multimodal_features)  # [batch_size, 3, hidden_dim]
        
        # 取序列的平均值作为最终特征表示
        fused_features = torch.mean(fused_features, dim=1)  # [batch_size, hidden_dim]
        
        # 预测标签
        tag_logits = self.tag_classifier(fused_features)  # [batch_size, num_tags]
        tag_probs = self.sigmoid(tag_logits)  # [batch_size, num_tags]
        
        return tag_probs

# 数据集类
class VideoMultiModalDataset(Dataset):
    def __init__(self, video_ids, text_processor, tag_mapping):
        self.video_ids = video_ids
        self.text_processor = text_processor
        self.tag_mapping = tag_mapping
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # 加载文本特征
        with open(f"e:/projectpython/blog/data/videos/text/{video_id}.json", 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        
        # 使用BERT处理文本
        text = text_data['title'] + ". " + text_data['description']
        text_features = self.text_processor(text)
        
        # 加载图像特征
        with open(f"e:/projectpython/blog/data/videos/images/{video_id}.json", 'r') as f:
            image_features = json.load(f)
        image_features = torch.FloatTensor(image_features)
        
        # 加载音频特征
        with open(f"e:/projectpython/blog/data/videos/audio/{video_id}.json", 'r') as f:
            audio_features = json.load(f)
        audio_features = torch.FloatTensor(audio_features)
        
        # 获取标签 (使用预定义的映射)
        if video_id in self.tag_mapping:
            tags = self.tag_mapping[video_id]
            # 将标签转换为one-hot编码
            tag_vector = torch.zeros(len(all_tags))
            for tag in tags:
                if tag in all_tags:
                    tag_idx = all_tags.index(tag)
                    tag_vector[tag_idx] = 1.0
        else:
            tag_vector = torch.zeros(len(all_tags))
            
        return text_features, image_features, audio_features, tag_vector, video_id

# 文本处理函数
class TextProcessor:
    def __init__(self):
        # 使用简单的词袋模型模拟BERT嵌入
        self.word_to_idx = {}
        for tag in all_tags:
            if tag not in self.word_to_idx:
                self.word_to_idx[tag] = len(self.word_to_idx)
        
        # 添加一些常见词
        common_words = ['视频', '教程', '分享', '学习', '推荐', '精彩', '专业', '详细', '分析', '讲解']
        for word in common_words:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        self.vocab_size = len(self.word_to_idx)
        self.embedding_dim = 768  # 模拟BERT的输出维度
        
    def __call__(self, text):
        # 简单的词袋表示
        vector = torch.zeros(self.embedding_dim)
        for word in self.word_to_idx:
            if word in text:
                idx = self.word_to_idx[word] % self.embedding_dim
                vector[idx] += 1
        
        # 归一化
        if torch.norm(vector) > 0:
            vector = vector / torch.norm(vector)
            
        return vector

# 预定义的视频标签映射
predefined_tags = {
    'v001': ['游戏', 'MOBA', '竞技', '王者荣耀'],
    'v002': ['美食', '烹饪', '教程', '家常菜'],
    'v003': ['科技', '评测', '手机', '电子产品'],
    'v004': ['健身', '运动', '训练', '减肥'],
    'v005': ['游戏', 'MOBA', '竞技', '英雄联盟', '电竞'],
    'v006': ['旅游', '日本', 'vlog', '东京', '文化'],
    'v007': ['编程', 'Python', '教程', '技术', '计算机'],
    'v008': ['家居', '装修', '设计', '生活', '室内'],
    'v009': ['瑜伽', '健身', '放松', '初学者', '运动'],
    'v010': ['动漫', '二次元', '评测', '娱乐', '推荐'],
    'v011': ['摄影', '手机', '技巧', '教程', '艺术'],
    'v012': ['文学', '阅读', '经典', '文化', '赏析'],
    'v013': ['电影', '解说', '好莱坞', '娱乐', '影评'],
    'v014': ['理财', '投资', '金融', '教程', '经济'],
    'v015': ['宠物', '猫咪', '护理', '生活', '萌宠']
}

# 训练模型
def train_multimodal_model():
    # 初始化文本处理器
    text_processor = TextProcessor()
    
    # 准备数据集
    video_ids = [video['id'] for video in videos_metadata]
    dataset = VideoMultiModalDataset(video_ids, text_processor, predefined_tags)
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = MultiModalTransformer()
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for text_features, image_features, audio_features, tag_labels, _ in dataloader:
            # 前向传播
            tag_probs = model(text_features, image_features, audio_features)
            
            # 计算损失
            loss = criterion(tag_probs, tag_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印每个epoch的损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    print("模型训练完成！")
    return model, text_processor

# 使用模型预测视频标签
def predict_video_tags(model, text_processor, video_id):
    # 加载视频数据
    with open(f"e:/projectpython/blog/data/videos/text/{video_id}.json", 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    
    # 处理文本
    text = text_data['title'] + ". " + text_data['description']
    text_features = text_processor(text)
    
    # 加载图像特征
    with open(f"e:/projectpython/blog/data/videos/images/{video_id}.json", 'r') as f:
        image_features = json.load(f)
    image_features = torch.FloatTensor(image_features).unsqueeze(0)  # 添加批次维度
    
    # 加载音频特征
    with open(f"e:/projectpython/blog/data/videos/audio/{video_id}.json", 'r') as f:
        audio_features = json.load(f)
    audio_features = torch.FloatTensor(audio_features).unsqueeze(0)  # 添加批次维度
    
    # 预测标签
    model.eval()
    with torch.no_grad():
        text_features = text_features.unsqueeze(0)  # 添加批次维度
        tag_probs = model(text_features, image_features, audio_features)
        
    # 获取概率最高的标签
    threshold = 0.5
    predicted_indices = (tag_probs > threshold).nonzero(as_tuple=True)[1]
    predicted_tags = [all_tags[idx] for idx in predicted_indices]
    
    # 如果没有标签超过阈值，选择前3个概率最高的标签
    if not predicted_tags:
        top_indices = torch.topk(tag_probs[0], 3).indices
        predicted_tags = [all_tags[idx] for idx in top_indices]
    
    return predicted_tags

# 主函数
def main():
    print("开始训练多模态视频标签预测模型...")
    model, text_processor = train_multimodal_model()
    
    # 为所有视频预测标签
    print("\n使用训练好的模型预测视频标签:")
    predicted_video_tags = {}
    
    for video in videos_metadata:
        video_id = video['id']
        predicted_tags = predict_video_tags(model, text_processor, video_id)
        predicted_video_tags[video_id] = predicted_tags
        
        print(f"视频 {video_id} - {video['title']}:")
        print(f"  预测标签: {', '.join(predicted_tags)}")
        if video_id in predefined_tags:
            print(f"  原始标签: {', '.join(predefined_tags[video_id])}")
        print()
    
    # 保存预测结果
    with open('e:/projectpython/blog/predicted_video_tags.json', 'w', encoding='utf-8') as f:
        json.dump(predicted_video_tags, f, ensure_ascii=False, indent=4)
    
    print("预测标签已保存到 'predicted_video_tags.json'")
    
    # 返回预测的标签，用于替换原始项目中的标签
    return predicted_video_tags

if __name__ == "__main__":
    main()