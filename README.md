# YouTube Video Summarizer

* 为了解决自己看视频太费时间的问题，就做了这个小东西
<br>
* 一个智能的 YouTube 视频总结工具，使用 AI 技术自动提取视频字幕并生成高质量的中文总结。

### 1. 环境要求

- Python 3.12 或更高版本
- 有效的 SiliconFlow API 密钥

### 2. 安装依赖

```bash
# 克隆项目
git clone 
cd 

# 安装依赖 
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 3. 环境配置

创建 `.env` 文件并添加你的 API 密钥：

```env
SILICONFLOW_API_KEY=sk-your-api-key-here
```

### 4. 运行程序

```bash
python main.py
```


##  输出文件

处理完成后，文件将保存在以下目录：

```
captions/          # 原始字幕文件 (.txt)
summaries/         # AI 总结文件 (.md)
```


##  配置选项

在 `main.py` 中的 `Config` 类可以调整以下参数：

```python
@dataclass
class Config:
    AI_MODEL: str = "deepseek-ai/DeepSeek-R1"     # AI 模型
    AI_TEMPERATURE: float = 0.6                   # 温度
    AI_MAX_TOKENS: int = 10000                    # 最大输出长度
    MIN_CAPTION_LENGTH: int = 100                 # 最小字幕长度
    MAX_RETRY_COUNT: int = 3                      # 重试次数
    REQUEST_TIMEOUT: int = 60                     # 请求超时时间
```

