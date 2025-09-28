# 🚀 PhoBERT Toxic Comment Classifier
### *Mô hình phân loại bình luận độc hại tiếng Việt hiệu quả*

<div align="center">

<!-- Animated Title Banner -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=35&duration=4000&pause=1000&color=DC143C&center=true&vCenter=true&width=800&height=100&lines=🤖+PhoBERT+Toxic+Classifier;🇻🇳+Vietnamese+Toxic+Detection;🎯+Binary+Classification;⚡+Powered+by+AI+%26+Focal+Loss" alt="Typing SVG" />

<!-- Dynamic Badges -->
![PhoBERT](https://img.shields.io/badge/Model-PhoBERTv2-blue?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF6B6B&color=4ECDC4)
![Vietnamese](https://img.shields.io/badge/Language-Vietnamese-red?style=for-the-badge&logo=google-translate&logoColor=white&labelColor=45B7D1&color=96CEB4)
![AI](https://img.shields.io/badge/AI-NLP-green?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=FFA07A&color=98D8C8)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logoColor=white&labelColor=F7DC6F&color=BB8FCE)

<!-- Glowing Links -->
[![🤗 Hugging Face Model](https://img.shields.io/badge/🤗%20Model-vijjj1/toxic--comment--phobert-ff6b35?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=FF6B35&color=F7931E)](https://huggingface.co/vijjj1/toxic-comment-phobert)
[![📊 Dataset](https://img.shields.io/badge/📊%20Dataset-Vietnamese%20Toxic%20Comments-purple?style=for-the-badge&logo=database&logoColor=white&labelColor=9B59B6&color=8E44AD)](https://github.com/your_username/your_repo/blob/main/comments_labels_binary.csv) <!-- Cập nhật link dataset nếu có -->
[![🎮 Demo](https://img.shields.io/badge/🎮%20Demo-Gradio%20App-orange?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=E67E22&color=D35400)](https://huggingface.co/spaces/vijjj1/toxic-comment-phobert-demo) <!-- Tạo link Gradio nếu có -->

<!-- GitHub Stats -->
![GitHub stars](https://img.shields.io/github/stars/vijjj1/toxic-comment-phobert?style=for-the-badge&logo=star&logoColor=white&labelColor=FFD700&color=FFA500)
![GitHub forks](https://img.shields.io/github/forks/vijjj1/toxic-comment-phobert?style=for-the-badge&logo=git&logoColor=white&labelColor=32CD32&color=228B22)
![GitHub issues](https://img.shields.io/github/issues/vijjj1/toxic-comment-phobert?style=for-the-badge&logo=github&logoColor=white&labelColor=FF6347&color=DC143C)

<!-- Animated Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,2,5,30&height=150&section=header&text=🇻🇳%20Vietnamese%20AI%20🤖&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%">

</div>

---

## 🎯 **Tổng quan dự án**

> 💡 **Sứ mệnh**: Xây dựng công cụ AI hiện đại để phát hiện và phân loại bình luận độc hại trong tiếng Việt trên mạng xã hội, sử dụng mô hình PhoBERT v2 và Focal Loss để tối ưu hiệu suất.

<div align="center">

<!-- Animated Stats Table -->
<table>
<tr>
<td width="50%" align="center">

### 🎭 **Khả năng phân loại**
```mermaid
pie title Toxic Classification
    "🟢 Non-Toxic" : 50
    "🔴 Toxic" : 50
</td>
<td width="50%" align="center">
📱 Nguồn dữ liệu
code
Mermaid
flowchart TD
    A[🌐 Social Media] --> F[🤖 PhoBERT Model]
    A --> B[🎵 TikTok]
    A --> C[📘 Facebook]
    A --> D[🎬 YouTube]
    A --> E[💬 Other Platforms]
    B --> G[🔍 Preprocessing]
    C --> G
    D --> G
    E --> G
    G --> F
</td>
</tr>
</table>
</div>
<!-- Gradient Line -->
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">
📊 Thông tin Dataset
<div align="center">
<!-- Animated Counter -->
<img src="https://readme-typing-svg.herokuapp.com?font=Roboto&size=25&duration=2000&pause=500&color=36BCF7&center=true&vCenter=true&width=600&height=60&lines=📝+Tổng+số+comments%3A+~11K;🏷️+2+Classes%3A+Toxic%2C+Non-Toxic;🌐+Multi-Platform+Data;🎯+Cân+bằng+dữ+liệu+với+Oversampling" alt="Stats Typing" />
📈 Metric	📋 Value	🎯 Description
📝 Comments	
![alt text](https://img.shields.io/badge/~11K-comments-blue?style=flat-square&logo=comment&logoColor=white)
Tổng số bình luận được thu thập và tăng cường
🏷️ Labels	
![alt text](https://img.shields.io/badge/2-classes-green?style=flat-square&logo=tag&logoColor=white)
non-toxic (0), toxic (1)
🌐 Sources	
![alt text](https://img.shields.io/badge/Multi-platform-orange?style=flat-square&logo=globe&logoColor=white)
Tổng hợp từ nhiều nguồn mạng xã hội
📊 Fields	
![alt text](https://img.shields.io/badge/2-columns-purple?style=flat-square&logo=table&logoColor=white)
comment, label
</div>
<details>
<summary>🔍 <strong>Chi tiết phân bố dữ liệu sau Oversampling</strong></summary>
code
Ascii
📊 Label Distribution (Resampled):
╭─────────────────────────────────────────────────╮
│                                                 │
│  🟢 Non-Toxic: ██████████████████████████ (50%) │
│  🔴 Toxic:    ██████████████████████████ (50%) │
│                                                 │
╰─────────────────────────────────────────────────╯
</details>
⚡ Cài đặt nhanh
<div align="center">
<!-- Installation Animation -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=20&duration=3000&pause=1000&color=FF6B6B&center=true&vCenter=true&width=500&height=50&lines=pip+install+transformers;pip+install+datasets;pip+install+torch;Ready+to+use!+🚀" alt="Installation" />
</div>
🛠️ Requirements
code
Bash
# 📦 Cài đặt các thư viện cần thiết
pip install transformers datasets scikit-learn sentencepiece torch pandas numpy matplotlib seaborn imblearn tqdm huggingface_hub

# 🎨 Hoặc cài đặt từ requirements.txt
# pip install -r requirements.txt
<details>
<summary>💻 <strong>Chi tiết dependencies</strong></summary>
code
Txt
transformers>=4.21.0     # 🤗 Hugging Face Transformers
datasets>=2.4.0          # 📊 Dataset processing
scikit-learn>=1.1.0      # 🔬 Machine Learning utilities
sentencepiece>=0.1.97    # 📝 Text tokenization
torch>=1.12.0            # 🔥 PyTorch framework
gradio>=3.0.0            # 🎮 Demo interface (nếu có)
numpy>=1.21.0            # 🔢 Numerical computing
pandas>=1.3.0            # 📈 Data manipulation
matplotlib>=3.5.0        # 📊 Data visualization
seaborn>=0.11.0          # 🎨 Statistical visualization
imblearn>=0.10.0         # ⚖️ Imbalanced-learn (oversampling)
tqdm>=4.0.0              # ⏳ Progress bars
huggingface_hub>=0.10.0  # 🤝 Hugging Face Hub interaction
</details>
🏗️ Hướng dẫn Training
🚀 Quick Start
code
Python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 🔧 Khởi tạo model và tokenizer
print("🤖 Loading PhoBERT model...")
model_name = "vinai/phobert-base-v2" # Hoặc "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NON-TOXIC", 1: "TOXIC"},
    label2id={"NON-TOXIC": 0, "TOXIC": 1}
)

print("✅ Model loaded successfully!")
print(f"🎯 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Chuyển model về chế độ đánh giá nếu chỉ để dự đoán
model.eval()
📋 Training Process
<div align="center">
code
Mermaid
graph TD
    A[📊 Load CSV (comments_labels_binary.csv)] --> A1(🔍 Xử lý trùng lặp & thống kê)
    A1 --> A2(📉 Phân bố nhãn trước oversampling)
    A2 --> B[🔧 Tăng cường dữ liệu Toxic (Data Augmentation)]
    B --> C[⚖️ Oversampling với RandomOverSampler]
    C --> C1(📈 Phân bố nhãn sau oversampling)
    C1 --> D[✂️ Chia Train/Test (70%/30%)]
    D --> E[📝 Tokenization với PhoBERT]
    E --> F[🏋️ Training Loop (Focal Loss)]
    F --> G[📈 Validation & Early Stopping]
    G --> H[💾 Save Best Model]
    H --> I[🚀 Push to Hugging Face Hub]

    style A fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
    style A1 fill:#FDDA0D,stroke:#333,stroke-width:2px,color:#333
    style A2 fill:#FDDA0D,stroke:#333,stroke-width:2px,color:#333
    style B fill:#4ECDC4,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#45B7D1,stroke:#333,stroke-width:2px,color:#fff
    style C1 fill:#FDDA0D,stroke:#333,stroke-width:2px,color:#333
    style D fill:#96CEB4,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#FECA57,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#FF9FF3,stroke:#333,stroke-width:2px,color:#fff
    style G fill:#54A0FF,stroke:#333,stroke-width:2px,color:#fff
    style H fill:#5F27CD,stroke:#333,stroke-width:2px,color:#fff
    style I fill:#00ADB5,stroke:#333,stroke-width:2px,color:#fff
</div>
<table>
<tr>
<td width="50%">
🎯 Bước 1: Chuẩn bị dữ liệu
code
Python
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("comments_labels_binary.csv", encoding='utf-8-sig')
df = df.drop_duplicates(subset=['comment']) # Loại bỏ trùng lặp
df['label'] = pd.to_numeric(df['label'], errors='coerce').dropna().astype(int)

# Tăng cường và Oversampling
# (Xem chi tiết trong code training để hiểu cách thực hiện)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[['comment']], df['label'])
df_resampled = pd.DataFrame({'comment': X_resampled['comment'], 'label': y_resampled})

train_df, test_df = train_test_split(df_resampled, test_size=0.3, random_state=42, stratify=df_resampled['label'])

print(f"📈 Training samples: {len(train_df)}")
print(f"🧪 Test samples: {len(test_df)}")
</td>
<td width="50%">
🏃‍♂️ Bước 2: Huấn luyện
code
Python
# Khởi tạo DataLoader, Model, Optimizer, Scheduler
# (Đã được mô tả chi tiết trong file training.py)

# Lớp Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Sử dụng Focal Loss
loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

# Bắt đầu vòng lặp huấn luyện
# (Xem chi tiết trong file training.py)

# Sau huấn luyện, lưu và đẩy model lên Hugging Face Hub
# model.push_to_hub("vijjj1/toxic-comment-phobert")
# tokenizer.push_to_hub("vijjj1/toxic-comment-phobert")
print("✅ Hoàn tất quá trình huấn luyện và đẩy lên Hugging Face!")
</td>
</tr>
</table>
📈 Kết quả Performance
<div align="center">
🏆 Model Performance
<!-- Animated Performance Metrics -->
<img src="https://readme-typing-svg.herokuapp.com?font=Roboto+Mono&size=22&duration=2500&pause=800&color=36BCF7&center=true&vCenter=true&width=700&lines=🎯+Accuracy%3A+~93%25;📊+F1-Score%3A+~93%25;✅+High+Recall+for+Toxic;🔍+Effective+Toxic+Detection" alt="Performance" />
📊 Metric	📈 Score	🎯 Details
🎯 Accuracy	
![alt text](https://img.shields.io/badge/93%25-success-brightgreen?style=for-the-badge&logo=target)
Độ chính xác tổng thể (ví dụ)
📊 F1-Score	
![alt text](https://img.shields.io/badge/93%25-good-green?style=for-the-badge&logo=chart-line)
F1-score trung bình (ví dụ)
🟢 Recall (Toxic)	
![alt text](https://img.shields.io/badge/Toxic%20Recall-~95%25-red?style=for-the-badge&logo=shield)
Khả năng phát hiện đúng bình luận độc hại
⚪ Precision (Non-Toxic)	
![alt text](https://img.shields.io/badge/Non--Toxic%20Precision-~92%25-blue?style=for-the-badge&logo=thumbs-up)
Độ chính xác khi phân loại không độc hại
</div>
📊 Detailed Results
Dựa trên kết quả classification_report từ code của bạn:
code
Ascii
🎭 Classification Performance:
╭─────────────┬─────────────┬─────────────┬─────────────╮
│   Class     │ Precision   │   Recall    │   F1-Score  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 🟢 Non-Toxic│    0.9200   │    0.9100   │    0.9100   │
│ 🔴 Toxic    │    0.9400   │    0.9500   │    0.9400   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│   Accuracy  │             │             │    0.9300   │
│   Macro Avg │    0.9300   │    0.9300   │    0.9300   │
│ Weighted Avg│    0.9300   │    0.9300   │    0.9300   │
╰─────────────┴─────────────┴─────────────┴─────────────╯
Lưu ý: Các số liệu trên là ví dụ dựa trên hình ảnh. Kết quả thực tế có thể thay đổi tùy thuộc vào quá trình training cuối cùng và ngưỡng phân loại.
<div align="center">
code
Mermaid
xychart-beta
    title "📊 Model Performance by Class (F1-Score)"
    x-axis [Non-Toxic, Toxic]
    y-axis "F1-Score" 0 --> 1
    bar [0.91, 0.94]
</div>
Ma trận nhầm lẫn:
Với kết quả confusion_matrix của bạn:
[[2839 285]
[ 172 3192]]
