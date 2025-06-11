import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import time
from user_clustering import perform_user_clustering, visualize_user_clusters  # 添加这行导入

# 设置matplotlib中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保输出目录存在
def ensure_directories():
    if not os.path.exists('./data2'):
        os.makedirs('./data2')
    print("已确保输出目录存在")

# 处理视频数据并生成标签
def process_video_data():
    # 导入多模态处理模块
    from videoprocess import main as process_videos
    
    if not os.path.exists('./data2/predicted_video_tags.json'):
        print("正在使用多模态Transformer处理视频数据并生成标签...")
        predicted_tags = process_videos()
    else:
        # 加载已经预测的标签
        with open('./data2/predicted_video_tags.json', 'r', encoding='utf-8') as f:
            predicted_tags = json.load(f)
        print("已加载现有视频标签数据")
    
    # 视频数据 - 使用多模态模型生成的标签
    videos_data = [
        {'id': 'v001', 'title': '王者荣耀精彩集锦', 'tags': predicted_tags['v001']},
        {'id': 'v002', 'title': '如何提高厨艺', 'tags': predicted_tags['v002']},
        {'id': 'v003', 'title': '2023科技产品评测', 'tags': predicted_tags['v003']},
        {'id': 'v004', 'title': '健身30分钟训练', 'tags': predicted_tags['v004']},
        {'id': 'v005', 'title': '英雄联盟职业比赛', 'tags': predicted_tags['v005']},
        {'id': 'v006', 'title': '日本东京之旅', 'tags': predicted_tags['v006']},
        {'id': 'v007', 'title': 'Python基础教程', 'tags': predicted_tags['v007']},
        {'id': 'v008', 'title': '家居装修设计指南', 'tags': predicted_tags['v008']},
        {'id': 'v009', 'title': '瑜伽初学者指南', 'tags': predicted_tags['v009']},
        {'id': 'v010', 'title': '最新动漫推荐与点评', 'tags': predicted_tags['v010']},
        {'id': 'v011', 'title': '手机摄影技巧大全', 'tags': predicted_tags['v011']},
        {'id': 'v012', 'title': '经典文学作品赏析', 'tags': predicted_tags['v012']},
        {'id': 'v013', 'title': '电影解说：好莱坞经典', 'tags': predicted_tags['v013']},
        {'id': 'v014', 'title': '投资理财入门指南', 'tags': predicted_tags['v014']},
        {'id': 'v015', 'title': '宠物猫咪护理知识', 'tags': predicted_tags['v015']}
    ]
    
    print(f"已处理 {len(videos_data)} 个视频数据")
    return videos_data

# 生成用户行为数据
def generate_user_behaviors(videos_data):
    print("开始生成用户行为数据...")
    np.random.seed(42)

    # 生成用户基础信息
    num_users = 20
    user_ids = [f'u{str(i+1).zfill(3)}' for i in range(num_users)]
    cities = ['北京', '上海', '广州', '深圳', '成都', '杭州', '武汉', '西安', '南京', '重庆']
    genders = ['M', 'F']

    # 生成用户行为数据
    user_behaviors = []
    np.random.seed(42)  # 初始种子

    for user_id in user_ids:
        # 为每个用户重新设置随机种子，增加随机性
        np.random.seed(int(user_id[1:]) + int(time.time()) % 1000)
        
        # 基础信息
        age = np.random.randint(18, 55)
        gender = np.random.choice(genders)
        city = np.random.choice(cities)
        
        # 为每个用户生成视频交互数据
        num_interactions = np.random.randint(5, 15)  # 每个用户观看的视频数量
        video_indices = np.random.choice(len(videos_data), num_interactions, replace=False)
        
        for v_idx in video_indices:
            video = videos_data[v_idx]
            
            # 生成行为数据
            watch_duration = np.random.uniform(0.1, 1.0)  # 观看时长比例(0.1-1.0)
            like = np.random.choice([0, 1], p=[0.7, 0.3])  # 是否点赞
            comment = np.random.choice([0, 1], p=[0.9, 0.1])  # 是否评论
            share = np.random.choice([0, 1], p=[0.95, 0.05])  # 是否分享
            favorite = np.random.choice([0, 1], p=[0.85, 0.15])  # 是否收藏
            
            # 记录交互数据
            user_behaviors.append({
                'user_id': user_id,
                'video_id': video['id'],
                'video_title': video['title'],
                'video_tags': video['tags'],
                'watch_duration': watch_duration,
                'like': like,
                'comment': comment,
                'share': share,
                'favorite': favorite,
                'timestamp': np.random.randint(1609459200, 1640995200),  # 2021年的时间戳
                'age': age,
                'gender': gender,
                'city': city
            })

    # 转换为DataFrame
    behaviors_df = pd.DataFrame(user_behaviors)

    # 保存用户行为数据
    behaviors_df.to_csv('./data2/user_behaviors.csv', index=False)
    print(f"已生成{len(behaviors_df)}条用户行为数据并保存到CSV文件")
    
    return behaviors_df, user_ids

# 生成用户画像
def generate_user_profiles(behaviors_df, user_ids):
    print("开始生成用户画像...")
    # 使用用户聚类模块生成用户画像，启用大模型特征提取
    user_profiles = perform_user_clustering(behaviors_df, user_ids, './data2', use_llm=True)
    
    # 可视化用户聚类结果
    visualize_user_clusters(user_profiles, './data2')
    
    print(f"已完成 {len(user_profiles)} 个用户画像生成")
    return user_profiles

# 构建注意力机制推荐模型
def build_attention_recommender(users_data, videos_data):
    print("开始构建注意力机制推荐模型...")
    # 准备数据：将标签和兴趣转换为文本字符串
    video_tags = [' '.join(video['tags']) for video in videos_data]
    user_interests = [' '.join(user['interests']) for user in users_data]

    # 将文本数据向量化
    vectorizer = TfidfVectorizer()
    # 先用所有文本训练向量化器
    all_texts = video_tags + user_interests
    vectorizer.fit(all_texts)

    # 转换用户和视频数据
    user_vectors = vectorizer.transform(user_interests).toarray()
    video_vectors = vectorizer.transform(video_tags).toarray()

    # 定义注意力机制模型
    class AttentionRecommender(nn.Module):
        def __init__(self, input_dim):
            super(AttentionRecommender, self).__init__()
            self.query_transform = nn.Linear(input_dim, 64)
            self.key_transform = nn.Linear(input_dim, 64)
            self.value_transform = nn.Linear(input_dim, 64)
            self.attention_combine = nn.Linear(64, 1)
            
        def forward(self, user_vector, video_vectors):
            # 转换用户向量为查询向量
            query = self.query_transform(user_vector)  # [batch_size, 64]
            
            # 转换视频向量为键和值
            keys = self.key_transform(video_vectors)  # [num_videos, 64]
            values = self.value_transform(video_vectors)  # [num_videos, 64]
            
            # 计算注意力分数
            query_expanded = query.unsqueeze(1)  # [batch_size, 1, 64]
            keys_expanded = keys.unsqueeze(0)  # [1, num_videos, 64]
            
            # 计算点积注意力
            attention_scores = torch.sum(query_expanded * keys_expanded, dim=2)  # [batch_size, num_videos]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, num_videos]
            
            return attention_weights

    # 初始化模型
    attention_model = AttentionRecommender(user_vectors.shape[1])

    # 将NumPy数组转换为PyTorch张量
    user_tensors = torch.FloatTensor(user_vectors)
    video_tensors = torch.FloatTensor(video_vectors)

    # 计算注意力权重
    attention_model.eval()
    with torch.no_grad():
        similarity_matrix = np.zeros((len(users_data), len(videos_data)))
        for i, user in enumerate(users_data):
            print(f"用户 {user['user_id']} 的基于注意力机制的推荐:")
            
            # 获取用户向量
            user_tensor = user_tensors[i].unsqueeze(0)  # 添加批次维度
            
            # 计算注意力权重
            attention_weights = attention_model(user_tensor, video_tensors).squeeze(0).numpy()
            similarity_matrix[i] = attention_weights
            
            # 获取相似度排序后的索引
            similar_indices = attention_weights.argsort()[::-1]
            
            # 显示前3个推荐结果
            for idx in similar_indices[:3]:
                print(f"  - {videos_data[idx]['title']} (注意力权重: {attention_weights[idx]:.4f})")
                print(f"    标签: {', '.join(videos_data[idx]['tags'])}")
            print()
    
    print("注意力机制推荐模型构建完成")
    return similarity_matrix

# 构建深度学习预测模型
def build_deep_learning_model(users_data, videos_data, similarity_matrix):
    print("开始构建深度学习预测模型...")
    # 生成模拟的历史交互数据
    np.random.seed(42)
    interactions = []
    for u_idx, user in enumerate(users_data):
        for v_idx, video in enumerate(videos_data):
            # 基于相似度生成完播率和互动概率（加入一些随机性）
            sim = similarity_matrix[u_idx][v_idx]
            completion_rate = min(1.0, max(0.1, sim + np.random.normal(0, 0.1)))
            interaction_prob = min(1.0, max(0.05, sim * 0.8 + np.random.normal(0, 0.15)))
            
            # 用户特征：年龄、性别(M=1,F=0)、城市编码
            gender_code = 1 if user['gender'] == 'M' else 0
            city_code = {'北京': 0, '上海': 1, '广州': 2, '深圳': 3, '成都': 4, 
                        '杭州': 5, '武汉': 6, '西安': 7, '南京': 8, '重庆': 9}.get(user['city'], 0)
            
            interactions.append({
                'user_id': user['user_id'],
                'video_id': video['id'],
                'user_idx': u_idx,
                'video_idx': v_idx,
                'age': user['age'] / 50.0,  # 归一化
                'gender': gender_code,
                'city': city_code / 10.0,  # 归一化
                'similarity': sim,
                'completion_rate': completion_rate,
                'interaction_prob': interaction_prob
            })

    # 转换为DataFrame
    interactions_df = pd.DataFrame(interactions)

    # 准备训练数据
    X_features = interactions_df[['user_idx', 'video_idx', 'age', 'gender', 'city', 'similarity']].values
    y_completion = interactions_df['completion_rate'].values
    y_interaction = interactions_df['interaction_prob'].values

    # 划分训练集和测试集
    X_train, X_test, y_train_comp, y_test_comp, y_train_inter, y_test_inter = train_test_split(
        X_features, y_completion, y_interaction, test_size=0.2, random_state=42
    )

    # 创建PyTorch数据集
    class UserVideoDataset(Dataset):
        def __init__(self, features, completion_labels, interaction_labels):
            self.features = torch.FloatTensor(features)
            self.completion_labels = torch.FloatTensor(completion_labels)
            self.interaction_labels = torch.FloatTensor(interaction_labels)
            
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.completion_labels[idx], self.interaction_labels[idx]

    # 创建数据加载器
    train_dataset = UserVideoDataset(X_train, y_train_comp, y_train_inter)
    test_dataset = UserVideoDataset(X_test, y_test_comp, y_test_inter)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 定义增强版的Transformer模型，包含多头自注意力机制
    class EnhancedTransformerPredictor(nn.Module):
        def __init__(self, input_dim=6, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.1):
            super(EnhancedTransformerPredictor, self).__init__()
            
            # 特征嵌入层
            self.embedding = nn.Linear(input_dim, hidden_dim)
            
            # Transformer编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 输出层 - 预测完播率和互动概率
            self.fc_completion = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, 1),
                nn.Sigmoid()
            )
            
            self.fc_interaction = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, 1),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            # 特征嵌入
            x = self.embedding(x)
            
            # 添加位置编码（这里简化处理，实际应用中可能需要更复杂的位置编码）
            x = x.unsqueeze(1)  # 添加序列维度
            
            # Transformer编码
            x = self.transformer_encoder(x)
            
            # 取序列的平均值作为特征表示
            x = x.squeeze(1)
            
            # 预测完播率和互动概率
            completion_rate = self.fc_completion(x).squeeze(-1)
            interaction_prob = self.fc_interaction(x).squeeze(-1)
            
            return completion_rate, interaction_prob

    # 初始化模型
    model = EnhancedTransformerPredictor()
    print(model)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for features, comp_labels, inter_labels in train_loader:
                # 前向传播
                comp_pred, inter_pred = model(features)
                
                # 计算损失
                loss_comp = criterion(comp_pred, comp_labels)
                loss_inter = criterion(inter_pred, inter_labels)
                loss = loss_comp + loss_inter
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 打印每个epoch的损失
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # 评估模型
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for features, comp_labels, inter_labels in test_loader:
                comp_pred, inter_pred = model(features)
                loss_comp = criterion(comp_pred, comp_labels)
                loss_inter = criterion(inter_pred, inter_labels)
                loss = loss_comp + loss_inter
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        print(f'测试损失: {avg_loss:.4f}')
        return avg_loss

    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, criterion, optimizer)

    # 评估模型
    print("评估模型...")
    evaluate_model(model, test_loader, criterion)
    
    print("深度学习预测模型构建完成")
    return model, interactions_df

# 生成推荐可视化
def generate_recommendation_visualizations(users_data, videos_data, similarity_matrix, model):
    print("开始生成推荐可视化...")
    
    # 定义预测函数
    def predict_for_user_video(model, user, video, user_idx, video_idx, similarity):
        model.eval()
        
        # 准备特征
        gender_code = 1 if user['gender'] == 'M' else 0
        city_code = {'北京': 0, '上海': 1, '广州': 2, '深圳': 3, '成都': 4, 
                    '杭州': 5, '武汉': 6, '西安': 7, '南京': 8, '重庆': 9}.get(user['city'], 0)
        
        features = torch.FloatTensor([[
            user_idx, video_idx, user['age'] / 50.0, gender_code, city_code / 10.0, similarity
        ]])
        
        # 预测
        with torch.no_grad():
            completion_rate, interaction_prob = model(features)
        
        return completion_rate.item(), interaction_prob.item()
    
    # 为所有用户生成基于注意力机制的推荐图表
    print("\n生成基于注意力机制的推荐图表...")

    # 创建一个图表，包含所有用户的注意力机制推荐
    plt.figure(figsize=(15, 10))

    # 为每个用户创建一个子图 (只展示前4个用户)
    for u_idx, user in enumerate(users_data[:4]):
        # 创建子图
        plt.subplot(2, 2, u_idx + 1)
        
        # 获取基于注意力机制的推荐
        attention_indices = similarity_matrix[u_idx].argsort()[::-1][:5]  # 取前5个
        attention_videos = [videos_data[idx]['title'] for idx in attention_indices]
        attention_scores = [similarity_matrix[u_idx][idx] for idx in attention_indices]
        
        # 创建横向条形图
        y_pos = np.arange(len(attention_videos))
        
        # 绘制基于注意力机制的推荐得分
        plt.barh(y_pos, attention_scores, 0.6, label='注意力机制推荐', color='skyblue')
        
        # 设置图表属性
        plt.yticks(y_pos, [title[:15] + '...' if len(title) > 15 else title for title in attention_videos])
        plt.xlabel('注意力权重')
        plt.title(f'用户 {user["user_id"]} ({user["gender"]}, {user["age"]}岁, {user["city"]}) 基础推荐')
        plt.legend()
        
        # 添加用户兴趣标签
        interests_text = '兴趣: ' + ', '.join(user['interests'])
        plt.figtext(0.5, 0.01 + u_idx * 0.25, interests_text, ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，留出空间给标题
    plt.suptitle('基于注意力机制的推荐分析', fontsize=16)
    plt.subplots_adjust(hspace=0.4)  # 增加子图之间的垂直间距

    # 保存图表
    plt.savefig('./data2/基础推荐分析.png', dpi=300, bbox_inches='tight')
    print("已保存基础推荐分析图表到当前文件夹的data2目录")

    # 创建另一个图表，展示每个用户的深度学习推荐
    print("\n生成深度学习推荐图表...")
    plt.figure(figsize=(15, 10))

    for u_idx, user in enumerate(users_data[:4]):  # 只展示前4个用户
        plt.subplot(2, 2, u_idx + 1)
        
        # 获取深度学习推荐
        dl_predictions = []
        for v_idx, video in enumerate(videos_data):
            similarity = similarity_matrix[u_idx][v_idx]
            completion_rate, interaction_prob = predict_for_user_video(
                model, user, video, u_idx, v_idx, similarity
            )
            combined_score = 0.6 * completion_rate + 0.4 * interaction_prob
            dl_predictions.append((video['title'], combined_score, completion_rate, interaction_prob))
        
        # 排序并获取前5个深度学习推荐
        dl_predictions.sort(key=lambda x: x[1], reverse=True)
        top_5 = dl_predictions[:5]
        
        # 提取数据
        titles = [item[0] for item in top_5]
        combined_scores = [item[1] for item in top_5]
        completion_rates = [item[2] for item in top_5]
        interaction_probs = [item[3] for item in top_5]
        
        # 创建分组柱状图
        x = np.arange(len(titles))
        width = 0.25
        
        plt.bar(x - width, combined_scores, width, label='综合得分', color='purple')
        plt.bar(x, completion_rates, width, label='完播率', color='green')
        plt.bar(x + width, interaction_probs, width, label='互动概率', color='orange')
        
        # 设置图表属性
        plt.xticks(x, [title[:15] + '...' if len(title) > 15 else title for title in titles], rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.ylabel('得分')
        plt.title(f'用户 {user["user_id"]} 的深度学习推荐详情')
        plt.legend()
        
        # 添加用户兴趣标签
        interests_text = '兴趣: ' + ', '.join(user['interests'])
        plt.figtext(0.5, 0.01 + u_idx * 0.25, interests_text, ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('深度学习推荐详细分析', fontsize=16)
    plt.subplots_adjust(hspace=0.4)

    # 保存图表
    plt.savefig('./data2/深度学习推荐分析.png', dpi=300, bbox_inches='tight')
    print("已保存深度学习推荐分析图表到当前文件夹的data2目录")
    
    print("推荐可视化生成完成")

# 主函数
def main():
    print("=== 开始运行推荐系统 ===")
    
    # 1. 确保目录存在
    ensure_directories()
    
    # 2. 处理视频数据
    videos_data = process_video_data()
    
    # 3. 生成用户行为数据
    behaviors_df, user_ids = generate_user_behaviors(videos_data)
    
    # 4. 生成用户画像
    users_data = generate_user_profiles(behaviors_df, user_ids)
    
    # 5. 构建注意力机制推荐模型
    similarity_matrix = build_attention_recommender(users_data, videos_data)
    
    # 6. 构建深度学习预测模型
    model, interactions_df = build_deep_learning_model(users_data, videos_data, similarity_matrix)
    
    # 7. 生成推荐可视化
    generate_recommendation_visualizations(users_data, videos_data, similarity_matrix, model)
    
    print("=== 推荐系统运行完成 ===")

# 程序入口
if __name__ == "__main__":
    main()