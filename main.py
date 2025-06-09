import os
import re
import requests
import json
import time
from functools import wraps
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.contrib.search import Search, Filter

load_dotenv()

@dataclass
class Config:
    SILICONFLOW_API_KEY: str = os.getenv('SILICONFLOW_API_KEY')
    API_URL: str = "https://api.siliconflow.cn/v1/chat/completions"
    CAPTIONS_DIR: str = "./captions"
    SUMMARIES_DIR: str = "./summaries"
    AI_MODEL: str = "deepseek-ai/DeepSeek-R1"
    AI_TEMPERATURE: float = 0.6
    AI_MAX_TOKENS: int = 10000
    MIN_CAPTION_LENGTH: int = 100
    MAX_RETRY_COUNT: int = 3
    REQUEST_TIMEOUT: int = 60

@dataclass
class ProcessingStats:
    start_time: float = time.time()
    successful_videos: int = 0
    failed_videos: int = 0
    total_caption_length: int = 0
    total_summary_length: int = 0
    
    def add_success(self, caption_length: int, summary_length: int):
        self.successful_videos += 1
        self.total_caption_length += caption_length
        self.total_summary_length += summary_length
    
    def add_failure(self):
        self.failed_videos += 1
    
    def print_final_stats(self):
        elapsed_time = time.time() - self.start_time
        total_videos = self.successful_videos + self.failed_videos
        print(f"\n{'='*50}")
        print("处理完成统计:")
        print(f"总耗时: {elapsed_time:.1f}秒")
        print(f"成功处理: {self.successful_videos} 个视频")
        print(f"失败/跳过: {self.failed_videos} 个视频")
        if total_videos > 0:
            print(f"成功率: {self.successful_videos/total_videos*100:.1f}%")
        if self.successful_videos > 0:
            print(f"平均字幕长度: {self.total_caption_length/self.successful_videos:.0f} 字符")
            print(f"平均总结长度: {self.total_summary_length/self.successful_videos:.0f} 字符")
        print(f"{'='*50}")

def retry_on_failure(max_retries: int = 3, delay: float = 2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"尝试 {attempt + 1} 失败: {e}, {delay}秒后重试...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

config = Config()

def sanitize_filename(filename: str) -> str:
    illegal_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(illegal_chars, '_', filename)
    sanitized = sanitized.strip(' .')
    return sanitized[:200]

def clean_caption_text(srt_text: str) -> str:
    lines = srt_text.split('\n')
    cleaned_lines = [
        line.strip() for line in lines
        if line.strip() and not line.strip().isdigit() and '-->' not in line
    ]
    
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def check_existing_files(safe_title: str) -> bool:
    summary_path = f"{config.SUMMARIES_DIR}/{safe_title}_summary.md"
    caption_path = f"{config.CAPTIONS_DIR}/{safe_title}.txt"
    return os.path.exists(summary_path) and os.path.exists(caption_path)

def create_optimized_prompt(title: str, text: str) -> str:
    base_prompt = f"""请对以下YouTube视频进行专业总结，要求：

    **总结结构：**
    1. **关键要点** (3-5个核心观点)
    2. **主要内容** (按逻辑分段总结)
    3. **重要概念/术语** (技术名词保持英文)
    4. **实用价值** (适用场景/受众)

    **格式要求：**
    - 使用Markdown格式
    - 中文输出，技术术语保持英文
    - 重点内容使用**加粗**
    - 代码/命令使用`代码块`

    视频标题：{title}

    字幕内容：
    {text}"""

    if any(keyword in title.lower() for keyword in ['git', 'github']):
        base_prompt += "\n\n**特别关注：** Git命令、GitHub功能、版本控制概念"
    elif any(keyword in title.lower() for keyword in ['linux', 'command']):
        base_prompt += "\n\n**特别关注：** Linux命令语法、参数说明、使用场景"
    elif any(keyword in title.lower() for keyword in ['programming', 'code']):
        base_prompt += "\n\n**特别关注：** 编程概念、代码示例、最佳实践"
    
    return base_prompt

@retry_on_failure(max_retries=3)
def summarize_with_ai(text: str, title: str) -> Optional[tuple[str, str]]:
    if not config.SILICONFLOW_API_KEY:
        print(f"警告: 未找到API密钥，跳过AI总结 - {title}")
        return None
    
    headers = {
        "Authorization": f"Bearer {config.SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = create_optimized_prompt(title, text)
    
    data = {
        "model": config.AI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.AI_TEMPERATURE,
        "max_tokens": config.AI_MAX_TOKENS,
        "stream": True,
        "enable_thinking": True,
        "thinking_budget": 4096
    }
    
    response = requests.post(
        config.API_URL, 
        headers=headers, 
        json=data, 
        stream=True, 
        timeout=config.REQUEST_TIMEOUT
    )
    response.raise_for_status()
    
    print("AI分析过程:")
    print("=" * 50)
    
    full_content = ""
    thinking_content = ""
    in_thinking = False
    current_section = ""
    
    for line in response.iter_lines():
        if not line:
            continue
            
        line_text = line.decode('utf-8')
        if not line_text.startswith('data: '):
            continue
            
        line_text = line_text[6:]
        if line_text.strip() == '[DONE]':
            break
            
        try:
            chunk_data = json.loads(line_text)
            if 'choices' in chunk_data and chunk_data['choices']:
                delta = chunk_data['choices'][0].get('delta', {})
                content = delta.get('content')
                reasoning_content = delta.get('reasoning_content')
                
                # 处理推理内容（思考过程）
                if reasoning_content:
                    if not in_thinking:
                        print("\n🤔 思考过程:")
                        print("-" * 30)
                        in_thinking = True
                        current_section = "thinking"
                    thinking_content += reasoning_content
                    print(reasoning_content, end='', flush=True)
                
                # 处理主要内容
                if content:
                    full_content += content
                    
                    # 如果之前在显示思考过程，现在切换到总结
                    if in_thinking and content.strip():
                        print("\n" + "-" * 30)
                        print("\n📝 生成总结:")
                        print("-" * 30)
                        in_thinking = False
                        current_section = "summary"
                    
                    # 显示主要内容（如果不在思考阶段）
                    if not in_thinking:
                        print(content, end='', flush=True)
                        
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"\n处理流数据时出错: {e}")
            continue
    
    print("\n" + "=" * 50)
    
    if not full_content.strip():
        print("警告: 未获取到AI分析内容")
        return None
    
    final_summary = full_content.replace('<think>', '').replace('</think>', '')
    thinking_clean = thinking_content.strip()
        
    return final_summary, thinking_clean

def save_summary(title: str, url: str, caption: str, summary: Optional[str], safe_title: str, thinking: Optional[str] = None):
    summary_path = f"{config.SUMMARIES_DIR}/{safe_title}_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**视频链接**: [{url}]({url})\n\n")
        
        if thinking and thinking.strip():
            f.write("## AI思考过程\n\n")
            f.write(f"```\n{thinking}\n```\n\n")
        
        f.write("## AI总结\n\n")
        f.write(summary if summary else "AI总结失败，请稍后重试。")
        
        f.write("\n\n## 字幕内容\n\n")
        f.write(caption)

def get_search_config() -> tuple[str, str]:
    print("\n搜索配置选项:")
    print("1. 按观看量排序 (默认)")
    print("2. 按相关性排序")
    print("3. 按上传时间排序")
    
    choice = input("请选择排序方式 (1-3, 默认1): ").strip()
    sort_options = {"1": "views", "2": "relevance", "3": "upload_date"}
    sort_by = sort_options.get(choice, "views")
    
    print("\n视频时长过滤:")
    print("1. 不限制 (默认)")
    print("2. 短视频 (<4分钟)")
    print("3. 中等长度 (4-20分钟)")
    print("4. 长视频 (>20分钟)")
    
    duration_choice = input("请选择时长过滤 (1-4, 默认1): ").strip()
    duration_options = {"2": "short", "3": "medium", "4": "long"}
    duration = duration_options.get(duration_choice, None)
    
    return sort_by, duration

def create_search_filters(sort_by: str = "views", duration: Optional[str] = None) -> Dict[str, Any]:
    filters = {
        "type": Filter.get_type("video"),
        "sort_by": Filter.get_sort_by(sort_by),
    }
    
    if duration:
        filters["duration"] = Filter.get_duration(duration)
    
    return filters

@retry_on_failure(max_retries=2)
def process_video(video, video_index: int, total_videos: int, stats: ProcessingStats) -> bool:
    try:
        url = video.watch_url
        yt = YouTube(url, on_progress_callback=on_progress)
        safe_title = sanitize_filename(yt.title)
        
        print(f"处理视频 {video_index + 1}/{total_videos}: {yt.title}")
        
        if check_existing_files(safe_title):
            print(f"文件已存在，跳过: {safe_title}")
            stats.add_success(0, 0)
            return True
        
        if 'a.en' not in yt.captions:
            print(f"跳过视频（无英文字幕）: {yt.title}")
            stats.add_failure()
            return False
        
        caption = yt.captions['a.en']
        caption_text = caption.generate_srt_captions()
        caption.save_captions(f"{config.CAPTIONS_DIR}/{safe_title}.txt")
        
        cleaned_caption = clean_caption_text(caption_text)
        
        if len(cleaned_caption.strip()) < config.MIN_CAPTION_LENGTH:
            print(f"跳过视频（字幕内容过短）: {yt.title}")
            stats.add_failure()
            return False
        
        summary = summarize_with_ai(cleaned_caption, yt.title)
        
        if summary:
            final_summary, thinking = summary
            save_summary(yt.title, url, cleaned_caption, final_summary, safe_title, thinking)
            summary_length = len(final_summary) if final_summary else 0
        else:
            save_summary(yt.title, url, cleaned_caption, None, safe_title)
            summary_length = 0
        stats.add_success(len(cleaned_caption), summary_length)
        
        status = "已保存总结" if summary else "AI总结失败，但已保存字幕"
        print(f"{status}: {config.SUMMARIES_DIR}/{safe_title}_summary.md")
        
        return True
        
    except KeyError as e:
        print(f"字幕访问错误: {e} - 跳过视频: {getattr(yt, 'title', '未知')}")
        stats.add_failure()
        return False
    except Exception as e:
        print(f"处理视频时出错: {e} - 跳过视频")
        stats.add_failure()
        return False
def main():
    os.makedirs(config.CAPTIONS_DIR, exist_ok=True)
    os.makedirs(config.SUMMARIES_DIR, exist_ok=True)
    
    search_keyword = input("请输入您想要搜索的关键词: ").strip()
    while not search_keyword:
        print("搜索关键词不能为空!")
        search_keyword = input("请输入您想要搜索的关键词: ").strip()
    
    while True:
        try:
            num_videos = int(input("请输入您想要处理的视频数量: "))
            if num_videos > 0:
                break
            else:
                print("请输入一个正整数!")
        except ValueError:
            print("请输入有效的数字!")
    
    sort_by, duration = get_search_config()
    filters = create_search_filters(sort_by, duration)
    
    print(f"开始搜索关键词: '{search_keyword}'...")
    search_results = Search(search_keyword, filters)
    
    stats = ProcessingStats()
    successful_count = 0
    
    for video_index, video in enumerate(search_results.videos):
        if successful_count >= num_videos:
            print(f"已成功处理 {num_videos} 个视频，程序结束。")
            break
        
        if process_video(video, video_index, num_videos, stats):
            successful_count += 1
        
        print(f"进度: 成功 {successful_count}/{num_videos}, 总尝试 {video_index + 1}\n")
        
        if video_index >= num_videos * 3:
            print("尝试次数过多，程序结束。")
            break
    
    stats.print_final_stats()

if __name__ == "__main__":
    main()
