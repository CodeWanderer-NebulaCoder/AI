import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import os
import requests  # 添加用于发送HTTP请求
import time     # 添加用于请求限流
import websockets
import asyncio
import base64   # 添加 base64 模块导入

# 设置matplotlib中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_cluster_description(cluster_id, cluster_data):
    """根据聚类数据生成用户画像描述"""
    avg_age = cluster_data['age'].mean()
    gender_ratio = cluster_data['gender'].value_counts(normalize=True)
    most_common_gender = gender_ratio.idxmax()
    gender_text = "男性" if most_common_gender == "M" else "女性"
    gender_percent = gender_ratio[most_common_gender] * 100
    
    most_common_city = cluster_data['city'].value_counts().idxmax()
    
    avg_videos = cluster_data['video_count'].mean()
    avg_watch = cluster_data['avg_watch_duration'].mean()
    avg_likes = cluster_data['like_count'].mean()
    
    # 互动倾向
    interaction_score = (cluster_data['like_count'] + 
                         cluster_data['comment_count'] * 2 + 
                         cluster_data['share_count'] * 3).mean()
    
    if cluster_id == 0:
        persona = f"年轻活跃用户：主要为{gender_text}（{gender_percent:.0f}%），平均年龄{avg_age:.1f}岁，"
        persona += f"居住在{most_common_city}等城市。观看视频数量多（平均{avg_videos:.1f}个），"
        persona += f"互动性强，经常点赞、评论和分享内容。完播率{avg_watch*100:.1f}%，"
        persona += "对短视频和娱乐内容有较高兴趣。"
    elif cluster_id == 1:
        persona = f"知识探索者：{gender_text}为主（{gender_percent:.0f}%），平均年龄{avg_age:.1f}岁，"
        persona += f"多分布在{most_common_city}等一线城市。观看视频较为专注（完播率{avg_watch*100:.1f}%），"
        persona += f"平均观看{avg_videos:.1f}个视频，互动适中。"
        persona += "偏好教育、科技和知识类内容，注重内容质量。"
    elif cluster_id == 2:
        persona = f"休闲浏览者：{gender_text}占比{gender_percent:.0f}%，平均年龄{avg_age:.1f}岁，"
        persona += f"来自{most_common_city}等多个城市。视频完播率较低（{avg_watch*100:.1f}%），"
        persona += f"平均观看{avg_videos:.1f}个不同视频，点赞次数少（{avg_likes:.1f}次）。"
        persona += "倾向于快速浏览多样内容，较少深度互动。"
    else:
        persona = f"忠实内容消费者：以{gender_text}为主（{gender_percent:.0f}%），平均{avg_age:.1f}岁，"
        persona += f"主要分布在{most_common_city}。完播率高（{avg_watch*100:.1f}%），"
        persona += f"观看视频数适中（{avg_videos:.1f}个），互动频率高。"
        persona += "对特定领域内容有持续关注，经常收藏和点赞喜欢的内容。"
    
    return persona

def load_env_from_file(env_file='./LLM_API_KEY.txt'):
    """从文件加载环境变量"""
    env_vars = {}
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
        return env_vars
    except Exception as e:
        print(f"警告: 读取环境变量文件失败: {str(e)}")
        return {}

# 添加 create_url 函数
def create_url(app_id, api_key, api_secret):
    """生成讯飞星火 WebSocket URL"""
    import hmac
    import base64
    import hashlib
    from datetime import datetime
    from urllib.parse import urlencode

    # 生成RFC1123格式的时间戳
    date = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    # 拼接字符串
    signature_origin = f"host: spark-api.xf-yun.com\ndate: {date}\nGET /v1/chat HTTP/1.1"
    
    # 使用hmac-sha256算法结合apiSecret对上述字符串签名
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'), hashlib.sha256).digest()
    signature = base64.b64encode(signature_sha).decode()
    
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode()
    
    # 组装鉴权参数
    params = {
        "authorization": authorization,
        "date": date,
        "host": "spark-api.xf-yun.com"
    }
    
    return "?" + urlencode(params)

def extract_features_with_llm(user_behaviors, api_key=None):
    """使用大模型API提取用户行为数据的高级特征"""
    # 从文件加载环境变量
    env_vars = load_env_from_file()
    
    # 获取API凭证
    api_password = env_vars.get("SPARK_API_PASSWORD", "tAmTATTmepcouxNiuqiE:AiTvHBjUXxvbGUvzklip")
    
    # 准备API请求的URL和头信息
    api_url = "https://spark-api-open.xf-yun.com/v2/chat/completions"
    headers = {
        'Authorization': f"Bearer {api_password}",
        'Content-Type': "application/json"
    }
    
    enhanced_features = {}
    
    # 对每个用户进行特征提取
    for user_id, behaviors in user_behaviors.items():
        print(f"使用大模型分析用户 {user_id} 的行为数据...")
        
        # 提取用户观看的视频标题和标签
        video_titles = [b['video_title'] for b in behaviors if 'video_title' in b]
        video_tags = []
        for b in behaviors:
            if 'video_tags' in b:
                video_tags.extend(b['video_tags'])
        
        # 提取用户互动行为
        likes = sum(b.get('like', 0) for b in behaviors)
        comments = sum(b.get('comment', 0) for b in behaviors)
        shares = sum(b.get('share', 0) for b in behaviors)
        favorites = sum(b.get('favorite', 0) for b in behaviors)
        
        # 构建提示词 - 修改提示词，更明确要求JSON格式
        prompt = f"""
        分析以下用户行为数据，提取用户兴趣特征和行为模式:
        
        观看的视频标题: {', '.join(video_titles[:10])}
        视频标签: {', '.join(set(video_tags))}
        互动数据: 点赞 {likes}次, 评论 {comments}次, 分享 {shares}次, 收藏 {favorites}次
        
        请提供以下分析，并严格按照JSON格式返回，不要有任何额外的文字说明:
        1. 用户的主要兴趣领域(最多5个)
        2. 用户的内容偏好(如短视频、教育内容、娱乐内容等)
        3. 用户的互动风格(如积极互动型、被动浏览型等)
        4. 用户的消费习惯特征
        5. 用户可能的人口统计学特征(如年龄段、职业可能性等)
        
        返回格式必须是有效的JSON，示例如下:
        {{
          "interests": ["兴趣1", "兴趣2", "兴趣3"],
          "content_preferences": ["短视频", "教育内容"],
          "interaction_style": "积极互动型",
          "consumption_habits": ["碎片化消费", "深度阅读"],
          "demographic_features": {{
            "age_range": "25-35",
            "possible_occupation": "技术从业者"
          }}
        }}
        
        请直接返回JSON，不要有任何前缀或后缀说明。
        """
        
        # 发送API请求
        try:
            # 准备消息 - 修改系统提示，强调返回JSON
            messages = [
                {"role": "system", "content": "你是一个专业的用户行为分析师，擅长从用户行为数据中提取有价值的特征。你的回复必须是有效的JSON格式，不包含任何额外的文字说明。"},
                {"role": "user", "content": prompt}
            ]
            
            # 构建请求体
            payload = {
                "model": "x1",
                "user": f"user_{user_id}",
                "messages": messages,
                "stream": False,  # 不使用流式响应
                "temperature": 0.1  # 降低温度，使输出更确定性
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result['choices'][0]['message']['content']
                
                # 尝试清理响应，移除可能的非JSON部分
                llm_response = llm_response.strip()
                # 查找JSON开始和结束的位置
                start_idx = llm_response.find('{')
                end_idx = llm_response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    llm_response = llm_response[start_idx:end_idx]
                
                # 解析JSON响应
                try:
                    enhanced_feature = json.loads(llm_response)
                    enhanced_features[user_id] = enhanced_feature
                    print(f"成功提取用户 {user_id} 的高级特征")
                except json.JSONDecodeError as e:
                    print(f"警告: 无法解析大模型返回的JSON数据: {e}")
                    print(f"原始响应: {llm_response}")
                    # 提供默认特征
                    default_feature = {
                        "interests": ["未知兴趣1", "未知兴趣2"],
                        "content_preferences": ["短视频", "娱乐内容"],
                        "interaction_style": "一般互动型",
                        "consumption_habits": ["碎片化消费"],
                        "demographic_features": {
                            "age_range": "25-35",
                            "possible_occupation": "一般职业"
                        }
                    }
                    enhanced_features[user_id] = default_feature
            else:
                print(f"警告: API请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                # API请求失败时也提供默认特征
                default_feature = {
                    "interests": ["未知兴趣1", "未知兴趣2"],
                    "content_preferences": ["短视频", "娱乐内容"],
                    "interaction_style": "一般互动型",
                    "consumption_habits": ["碎片化消费"],
                    "demographic_features": {
                        "age_range": "25-35",
                        "possible_occupation": "一般职业"
                    }
                }
                enhanced_features[user_id] = default_feature
            
            # 避免API限流
            time.sleep(1)
            
        except Exception as e:
            print(f"错误: 调用大模型API时出现异常: {str(e)}")
            # 异常情况下也提供默认特征
            default_feature = {
                "interests": ["未知兴趣1", "未知兴趣2"],
                "content_preferences": ["短视频", "娱乐内容"],
                "interaction_style": "一般互动型",
                "consumption_habits": ["碎片化消费"],
                "demographic_features": {
                    "age_range": "25-35",
                    "possible_occupation": "一般职业"
                }
            }
            enhanced_features[user_id] = default_feature
    
    return enhanced_features

# 在perform_user_clustering函数中添加大模型特征提取
def perform_user_clustering(behaviors_df, user_ids, output_dir='./data2', num_clusters=4, use_llm=False):
    """
    执行用户聚类分析并生成用户画像
    
    参数:
    behaviors_df: 用户行为数据DataFrame
    user_ids: 用户ID列表
    output_dir: 输出目录
    num_clusters: 聚类数量
    use_llm: 是否使用大模型进行特征提取
    
    返回:
    user_profiles: 用户画像列表
    """
    print("开始生成用户画像...")
    
    # 1. 按用户聚合行为数据
    user_stats = behaviors_df.groupby('user_id').agg({
        'watch_duration': ['mean', 'sum', 'count'],
        'like': 'sum',
        'comment': 'sum',
        'share': 'sum',
        'favorite': 'sum',
        'age': 'first',
        'gender': 'first',
        'city': 'first'
    }).reset_index()
    
    # 整理列名
    user_stats.columns = ['user_id', 'avg_watch_duration', 'total_watch_time', 
                         'video_count', 'like_count', 'comment_count', 
                         'share_count', 'favorite_count', 'age', 'gender', 'city']
    
    # 计算额外的行为特征
    user_stats['engagement_rate'] = (user_stats['like_count'] + user_stats['comment_count'] * 2 + 
                                    user_stats['share_count'] * 3) / user_stats['video_count'].clip(lower=1)
    user_stats['completion_rate'] = user_stats['avg_watch_duration']  # 假设watch_duration是完播率(0-1之间)
    user_stats['favorite_rate'] = user_stats['favorite_count'] / user_stats['video_count'].clip(lower=1)
    
    # 2. 提取用户的内容偏好 - 基于标签的TF-IDF分析
    user_tags = {}
    tag_set = set()  # 收集所有标签用于后续向量化
    
    for user_id in user_ids:
        # 获取用户观看过的所有视频标签
        user_videos = behaviors_df[behaviors_df['user_id'] == user_id]
        all_tags = []
        for _, row in user_videos.iterrows():
            all_tags.extend(row['video_tags'])
            tag_set.update(row['video_tags'])
        
        # 计算标签频率
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 保存用户标签
        user_tags[user_id] = tag_counts
    
    # 3. 使用大模型进行高级特征提取（如果启用）
    llm_features = {}
    if use_llm:
        print("使用大模型进行高级特征提取...")
        # 准备用户行为数据
        user_behaviors = {}
        for user_id in user_ids:
            user_behaviors[user_id] = behaviors_df[behaviors_df['user_id'] == user_id].to_dict('records')
        
        # 调用大模型API提取特征
        llm_features = extract_features_with_llm(user_behaviors)
        
        # 保存大模型提取的特征
        with open(f'{output_dir}/llm_features.json', 'w', encoding='utf-8') as f:
            json.dump(llm_features, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        
        print(f"已保存大模型提取的特征到 {output_dir}/llm_features.json")
    
    # 创建标签向量化特征
    tag_list = list(tag_set)
    tag_vectors = np.zeros((len(user_ids), len(tag_list)))
    
    for i, user_id in enumerate(user_ids):
        user_tag_counts = user_tags[user_id]
        total_tags = sum(user_tag_counts.values()) or 1  # 避免除零错误
        
        for j, tag in enumerate(tag_list):
            # 使用TF-IDF思想：标签频率/总标签数
            tag_vectors[i, j] = user_tag_counts.get(tag, 0) / total_tags
    
    # 4. 特征向量化处理
    # 4.1 One-hot编码处理类别特征
    encoder = OneHotEncoder(sparse_output=False)
    categorical_features = pd.DataFrame({
        'gender': user_stats['gender'],
        'city': user_stats['city']
    })
    encoded_features = encoder.fit_transform(categorical_features)
    
    # 4.2 标准化数值特征
    scaler = StandardScaler()
    numerical_features = user_stats[['avg_watch_duration', 'video_count', 'like_count', 
                                    'comment_count', 'share_count', 'favorite_count', 'age',
                                    'engagement_rate', 'completion_rate', 'favorite_rate']]
    scaled_numerical = scaler.fit_transform(numerical_features)
    
    # 4.3 合并所有特征
    all_features = np.hstack((scaled_numerical, encoded_features, tag_vectors))
    
    # 如果有大模型特征，添加到特征矩阵
    if llm_features and len(llm_features) > 0:
        # 提取大模型特征并转换为数值向量
        llm_vectors = []
        for user_id in user_ids:
            if user_id in llm_features:
                # 提取兴趣向量 (示例)
                interests = llm_features[user_id].get('interests', [])
                content_prefs = llm_features[user_id].get('content_preferences', [])
                
                # 简单处理：将兴趣和内容偏好的数量作为特征
                interest_count = len(interests)
                content_pref_count = len(content_prefs)
                
                # 互动风格编码 (示例)
                interaction_style = llm_features[user_id].get('interaction_style', '')
                style_code = 0
                if '积极' in interaction_style:
                    style_code = 1
                elif '被动' in interaction_style:
                    style_code = -1
                
                # 组合特征
                user_llm_vector = [interest_count, content_pref_count, style_code]
            else:
                # 如果没有大模型特征，使用默认值
                user_llm_vector = [0, 0, 0]
            
            llm_vectors.append(user_llm_vector)
        
        # 将大模型特征转换为numpy数组并标准化
        llm_array = np.array(llm_vectors)
        if llm_array.shape[1] > 0:  # 确保有特征
            llm_scaler = StandardScaler()
            scaled_llm = llm_scaler.fit_transform(llm_array)
            
            # 合并到所有特征中
            all_features = np.hstack((all_features, scaled_llm))
            print(f"已添加 {scaled_llm.shape[1]} 个大模型提取的特征")
    
    # 5. 划分训练集和测试集 (80%训练, 20%测试)
    from sklearn.model_selection import train_test_split
    
    train_indices, test_indices = train_test_split(
        np.arange(len(user_ids)), test_size=0.2, random_state=42
    )
    
    train_features = all_features[train_indices]
    test_features = all_features[test_indices]
    
    # 6. 使用K-means聚类分析
    from sklearn.metrics import silhouette_score
    
    # 可选：寻找最佳聚类数
    if len(user_ids) > num_clusters * 3:  # 确保有足够的样本
        silhouette_scores = []
        k_range = range(2, min(num_clusters + 3, len(user_ids) // 2))
        
        for k in k_range:
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans_test.fit_predict(train_features)
            score = silhouette_score(train_features, cluster_labels)
            silhouette_scores.append(score)
        
        # 找出最佳K值
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"根据轮廓系数分析，最佳聚类数为: {best_k}")
        
        # 如果最佳K与预设值不同，使用最佳K
        if best_k != num_clusters:
            print(f"调整聚类数从{num_clusters}到{best_k}")
            num_clusters = best_k
    
    # 在训练集上训练K-means模型
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(train_features)
    
    # 预测所有用户的聚类
    clusters = kmeans.predict(all_features)
    
    # 将聚类结果添加到用户统计数据中
    user_stats['cluster'] = clusters
    
    # 7. 评估聚类质量
    from sklearn.metrics import davies_bouldin_score
    
    silhouette_avg = silhouette_score(all_features, clusters)
    davies_bouldin_avg = davies_bouldin_score(all_features, clusters)
    
    print(f"聚类评估 - 轮廓系数: {silhouette_avg:.4f} (越高越好)")
    print(f"聚类评估 - Davies-Bouldin指数: {davies_bouldin_avg:.4f} (越低越好)")
    
    # 8. 生成用户画像
    user_profiles = []
    
    # 为每个用户生成画像
    for user_id in user_ids:
        user_data = user_stats[user_stats['user_id'] == user_id].iloc[0]
        cluster_id = user_data['cluster']
        
        # 获取该用户所属聚类的所有用户数据
        cluster_data = user_stats[user_stats['cluster'] == cluster_id]
        
        # 获取用户标签偏好
        user_tag_prefs = user_tags[user_id]
        top_tags = sorted(user_tag_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tags = [tag for tag, count in top_tags]
        
        # 生成用户画像
        cluster_description = generate_cluster_description(cluster_id, cluster_data)
        
        # 创建用户画像对象
        user_profile = {
            'user_id': user_id,
            'age': int(user_data['age']),
            'gender': user_data['gender'],
            'city': user_data['city'],
            'cluster': int(cluster_id),
            'interests': top_tags,
            'engagement_rate': float(user_data['engagement_rate']),
            'completion_rate': float(user_data['completion_rate']),
            'favorite_rate': float(user_data['favorite_rate']),
            'persona_description': cluster_description
        }
        
        # 如果有大模型特征，添加到用户画像
        if user_id in llm_features:
            user_profile['llm_features'] = llm_features[user_id]
            
            # 使用大模型提取的兴趣替换或补充原有兴趣
            if 'interests' in llm_features[user_id] and llm_features[user_id]['interests']:
                # 合并两种方式提取的兴趣
                combined_interests = set(top_tags)
                combined_interests.update(llm_features[user_id]['interests'])
                user_profile['interests'] = list(combined_interests)[:5]  # 限制最多5个兴趣
        
        user_profiles.append(user_profile)
    
    # 保存用户画像数据
    with open(f'{output_dir}/user_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(user_profiles, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    
    # 保存聚类模型
    import pickle
    with open(f'{output_dir}/kmeans_model.pkl', 'wb') as f:
        pickle.dump({
            'kmeans': kmeans,
            'scaler': scaler,
            'encoder': encoder,
            'tag_list': tag_list
        }, f)
    
    print(f"已生成{len(user_profiles)}个用户画像并保存到JSON文件")
    print(f"聚类模型已保存，可用于预测新用户的聚类")
    
    return user_profiles

def load_user_profiles(file_path='./data2/user_profiles.json'):
    """加载用户画像数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        user_profiles = json.load(f)
    return user_profiles

def visualize_user_clusters(user_profiles, output_dir='./data2', num_clusters=4):
    """
    可视化用户聚类结果
    
    参数:
    user_profiles: 用户画像列表
    output_dir: 输出目录
    num_clusters: 聚类数量
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 转换为DataFrame便于分析
    profiles_df = pd.DataFrame(user_profiles)
    
    # 1. 创建聚类分布图
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    cluster_counts = profiles_df['cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    plt.title('用户聚类分布')
    plt.xlabel('聚类ID')
    plt.ylabel('用户数量')
    plt.xticks(range(num_clusters))
    
    # 2. 创建年龄分布图 (按聚类)
    plt.subplot(2, 2, 2)
    for cluster in range(num_clusters):
        cluster_data = profiles_df[profiles_df['cluster'] == cluster]
        if not cluster_data.empty:
            plt.hist(cluster_data['age'], alpha=0.5, bins=10, 
                     label=f'聚类 {cluster}')
    
    plt.title('各聚类的年龄分布')
    plt.xlabel('年龄')
    plt.ylabel('用户数量')
    plt.legend()
    
    # 3. 创建互动率分布图
    plt.subplot(2, 2, 3)
    for cluster in range(num_clusters):
        cluster_data = profiles_df[profiles_df['cluster'] == cluster]
        if not cluster_data.empty and 'engagement_rate' in cluster_data.columns:
            plt.scatter(cluster_data['completion_rate'], 
                       cluster_data['engagement_rate'],
                       alpha=0.7, label=f'聚类 {cluster}')
    
    plt.title('完播率 vs 互动率')
    plt.xlabel('完播率')
    plt.ylabel('互动率')
    plt.legend()
    
    # 4. 创建用户兴趣标签分布图
    plt.subplot(2, 2, 4)
    all_interests = []
    for user in user_profiles:
        all_interests.extend(user['interests'])
        
    interest_counts = Counter(all_interests)
    top_interests = interest_counts.most_common(10)  # 只显示前10个最常见的标签
    
    interests = [tag for tag, _ in top_interests]
    counts = [count for _, count in top_interests]
    
    # 绘制兴趣分布条形图
    plt.barh(range(len(interests)), counts, color='lightgreen')
    plt.yticks(range(len(interests)), interests)
    plt.title('热门用户兴趣标签')
    plt.xlabel('出现次数')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/用户画像分析_概览.png', dpi=300, bbox_inches='tight')
    print(f"已保存用户画像概览图表到{output_dir}目录")
    
    # 5. 创建每个聚类的详细分析图
    for cluster in range(num_clusters):
        cluster_data = profiles_df[profiles_df['cluster'] == cluster]
        if cluster_data.empty:
            continue
            
        plt.figure(figsize=(15, 10))
        
        # 5.1 性别分布
        plt.subplot(2, 3, 1)
        gender_counts = cluster_data['gender'].value_counts()
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
               colors=['lightblue', 'lightpink'])
        plt.title(f'聚类 {cluster} 性别分布')
        
        # 5.2 城市分布
        plt.subplot(2, 3, 2)
        city_counts = cluster_data['city'].value_counts().head(5)  # 前5个城市
        plt.bar(city_counts.index, city_counts.values, color='lightgreen')
        plt.title(f'聚类 {cluster} 主要城市分布')
        plt.xticks(rotation=45, ha='right')
        
        # 5.3 年龄分布
        plt.subplot(2, 3, 3)
        plt.hist(cluster_data['age'], bins=10, color='orange', alpha=0.7)
        plt.title(f'聚类 {cluster} 年龄分布')
        plt.xlabel('年龄')
        plt.ylabel('用户数量')
        
        # 5.4 兴趣标签
        plt.subplot(2, 3, 4)
        cluster_interests = []
        for _, user in cluster_data.iterrows():
            cluster_interests.extend(user['interests'])
            
        interest_counts = Counter(cluster_interests)
        top_interests = interest_counts.most_common(8)
        
        interests = [tag for tag, _ in top_interests]
        counts = [count for _, count in top_interests]
        
        plt.barh(range(len(interests)), counts, color='purple', alpha=0.7)
        plt.yticks(range(len(interests)), interests)
        plt.title(f'聚类 {cluster} 兴趣标签')
        plt.xlabel('出现次数')
        
        # 5.5 行为特征
        plt.subplot(2, 3, 5)
        if 'engagement_rate' in cluster_data.columns and 'completion_rate' in cluster_data.columns:
            plt.scatter(cluster_data['completion_rate'], 
                       cluster_data['engagement_rate'],
                       alpha=0.7, color='red')
            plt.title(f'聚类 {cluster} 用户行为特征')
            plt.xlabel('完播率')
            plt.ylabel('互动率')
        
        # 5.6 聚类描述
        plt.subplot(2, 3, 6)
        description = cluster_data.iloc[0]['persona_description']
        plt.text(0.5, 0.5, description, 
                ha='center', va='center', wrap=True,
                fontsize=9)
        plt.axis('off')
        plt.title(f'聚类 {cluster} 用户画像描述')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/用户画像分析_聚类{cluster}.png', dpi=300, bbox_inches='tight')
        print(f"已保存聚类 {cluster} 的详细分析图表")
        
    # 6. 输出每个聚类的用户画像描述
    print("\n用户画像聚类分析:")
    for cluster_id in range(num_clusters):
        cluster_users = [user for user in user_profiles if user['cluster'] == cluster_id]
        if cluster_users:
            print(f"\n聚类 {cluster_id} ({len(cluster_users)}个用户):")
            print(f"代表性描述: {cluster_users[0]['persona_description']}")
            
            # 计算聚类特征
            cluster_df = profiles_df[profiles_df['cluster'] == cluster_id]
            
            # 输出聚类统计信息
            print(f"平均年龄: {cluster_df['age'].mean():.1f}岁")
            print(f"性别比例: 男性 {(cluster_df['gender'] == 'M').mean()*100:.1f}%, 女性 {(cluster_df['gender'] == 'F').mean()*100:.1f}%")
            
            if 'engagement_rate' in cluster_df.columns:
                print(f"平均互动率: {cluster_df['engagement_rate'].mean():.4f}")
            if 'completion_rate' in cluster_df.columns:
                print(f"平均完播率: {cluster_df['completion_rate'].mean():.4f}")
            
            print("典型用户:")
            for user in cluster_users[:3]:  # 只显示前3个用户
                print(f"  - 用户ID: {user['user_id']}, 年龄: {user['age']}, 性别: {'男' if user['gender']=='M' else '女'}, 城市: {user['city']}")
                print(f"    兴趣标签: {', '.join(user['interests'])}")

# 如果直接运行此文件，则执行示例
# 修改主程序入口
if __name__ == "__main__":
    # 简单示例：从文件加载用户行为数据并执行聚类
    print("用户聚类模块 - 示例运行")
    
    # 检查是否存在用户行为数据
    if os.path.exists('./data2/user_behaviors.csv'):
        # 加载用户行为数据
        behaviors_df = pd.read_csv('./data2/user_behaviors.csv')
        
        # 提取用户ID列表
        user_ids = behaviors_df['user_id'].unique().tolist()
        
        # 检查是否启用大模型特征提取
        use_llm = os.getenv("USE_LLM", "False").lower() in ("true", "1", "yes")
        
        # 执行用户聚类
        user_profiles = perform_user_clustering(behaviors_df, user_ids, use_llm=use_llm)
        
        # 可视化聚类结果
        visualize_user_clusters(user_profiles)
        
        # 示例：预测新用户的聚类
        if os.path.exists('./data2/kmeans_model.pkl'):
            print("\n示例：预测新用户的聚类")
            
            # 创建一个新用户行为数据
            new_user = {
                'avg_watch_duration': 0.7,
                'video_count': 8,
                'like_count': 3,
                'comment_count': 1,
                'share_count': 0,
                'favorite_count': 2,
                'age': 28,
                'gender': 'F',
                'city': '上海',
                'video_tags': ['教育', '科技', '知识'],
                'engagement_rate': 0.5,
                'completion_rate': 0.7,
                'favorite_rate': 0.25
            }
            
            # 预测聚类
            cluster_id = predict_user_cluster(new_user)
            
            # 获取该聚类的描述
            cluster_users = [user for user in user_profiles if user['cluster'] == cluster_id]
            if cluster_users:
                print(f"新用户预测所属聚类: {cluster_id}")
                print(f"聚类描述: {cluster_users[0]['persona_description']}")
    else:
        # 如果没有用户行为数据，显示提示信息
        print("未找到用户行为数据文件 (./data2/user_behaviors.csv)")
        print("请先运行 tuijianmodel.py 生成用户行为数据")
        
        # 尝试加载已有的用户画像数据
        if os.path.exists('./data2/user_profiles.json'):
            user_profiles = load_user_profiles()
            print(f"已加载{len(user_profiles)}个用户画像")
            
            # 可视化聚类结果
            visualize_user_clusters(user_profiles)
        else:
            print("也未找到用户画像数据文件 (./data2/user_profiles.json)")


def predict_user_cluster(user_behavior, model_path='./data2/kmeans_model.pkl'):
    """
    预测新用户所属的聚类
    
    参数:
    user_behavior: 用户行为数据字典
    model_path: 聚类模型路径
    
    返回:
    cluster_id: 预测的聚类ID
    """
    # 加载模型
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    kmeans = model_data['kmeans']
    scaler = model_data['scaler']
    encoder = model_data['encoder']
    tag_list = model_data['tag_list']
    
    # 准备用户特征
    # 1. 数值特征
    numerical_features = np.array([
        user_behavior.get('avg_watch_duration', 0),
        user_behavior.get('video_count', 0),
        user_behavior.get('like_count', 0),
        user_behavior.get('comment_count', 0),
        user_behavior.get('share_count', 0),
        user_behavior.get('favorite_count', 0),
        user_behavior.get('age', 25),
        user_behavior.get('engagement_rate', 0),
        user_behavior.get('completion_rate', 0),
        user_behavior.get('favorite_rate', 0)
    ]).reshape(1, -1)
    
    # 标准化数值特征
    scaled_numerical = scaler.transform(numerical_features)
    
    # 2. 类别特征
    categorical_features = pd.DataFrame({
        'gender': [user_behavior.get('gender', 'M')],
        'city': [user_behavior.get('city', '北京')]
    })
    
    # One-hot编码
    try:
        encoded_features = encoder.transform(categorical_features)
    except ValueError:
        # 如果遇到未知类别，使用全零向量
        encoded_features = np.zeros((1, encoder.get_feature_names_out().shape[0]))
    
    # 3. 标签特征
    tag_vectors = np.zeros((1, len(tag_list)))
    
    user_tags = user_behavior.get('video_tags', [])
    tag_counts = {}
    for tag in user_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    total_tags = sum(tag_counts.values()) or 1
    
    for j, tag in enumerate(tag_list):
        tag_vectors[0, j] = tag_counts.get(tag, 0) / total_tags
    
    # 合并所有特征
    all_features = np.hstack((scaled_numerical, encoded_features, tag_vectors))
    
    # 预测聚类
    cluster_id = kmeans.predict(all_features)[0]
    
    return int(cluster_id)