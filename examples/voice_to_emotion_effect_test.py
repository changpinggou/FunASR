import os
import sys
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# 生成符合Go语言格式的时间戳
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"emotion_test_{timestamp}.log"

full_log_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file_name)

# 配置日志输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename= full_log_file_name,  # 指定日志文件
    filemode='a'  # 'a' 追加模式，'w' 覆盖模式
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_detector_emotion import VoiceDetectorEmotion as VoiceDetector

# voice_detector = VoiceDetector(model_id="iic/emotion2vec_base_finetuned")
voice_detector = VoiceDetector(model_id="iic/emotion2vec_plus_large")

# 假设的emotion_infer_task函数，需要根据实际情况实现
def emotion_infer_task(emotion: str, voice_path: str) -> Tuple[bool, str, Exception]:
    """情感推理任务
    
    Args:
        emotion: 预期情感
        voice_path: 音频文件路径   
    Returns:
        tuple: (是否成功, 推理结果情感, 错误信息)
    """
    try:
        # TODO: 实现实际的情感推理逻辑)
        result_dict = voice_detector.detect(voice_path, source_type="path")
        # 过滤掉 prompt
        if "prompt" in result_dict:
            del result_dict["prompt"]
        
        result_emotion = result_dict.get('result', '')
        result = result_emotion.split('/')[-1]
        success = (result == emotion)

        score = result_dict.get('score')
        elapse = result_dict.get('elapsed_time')

        voiceFileName = os.path.basename(voice_path)
        logger.info("[applechang-emotion-effect] task-end, voicePath=%s, elpase: %d ms, emotion:%s, score:%f", voiceFileName, elapse, result_emotion, score)

        return success, result, None

    except Exception as e:
        return False, "", e


def process_emotion_files(emotion_files: Dict[str, List[str]]) -> None:
    """处理情感文件并进行推理
    
    Args:
        emotion_files: 情感文件字典，key为情感名称，value为文件路径列表
    """
    for emotion, voice_list in emotion_files.items():
        print(f"开始处理 {emotion} 目录里的所有音频: {len(voice_list)}")
        total_count = len(voice_list)
        success_count = 0
        fail_count = 0

        for voice_path in voice_list:
            success, result_emotion, err = emotion_infer_task(emotion, voice_path)
            
            if err is not None:
                logger.error(f"[applechang-emotion-effect] emotion_infer_task fail: {err}")
                fail_count += 1
                continue

            if success:
                success_count += 1
            else:
                fail_count += 1
                voice_file_name = os.path.basename(voice_path)
                logger.warning(f"[applechang-emotion-effect] not match, voicePath={voice_file_name}, Emotion={emotion}, resultEmotion={result_emotion}")

        success_rate = 0.0
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
        
        logger.info(f"[applechang-emotion-effect] {emotion} 目录里的所有音频处理结果, totalCount={total_count}, successCount={success_count}, failCount={fail_count}, successRate={success_rate:.2f}%")

def traverse_emotion_voice(root_dir: str) -> Dict[str, List[str]]:
    """遍历情感语音目录，返回每个情感对应的WAV文件列表
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        字典，key为情感名称，value为该情感下的WAV文件路径列表
    """
    emotion_files = {}
    
    # 预期的情感目录
    emotions = ["angry", "fear", "happy", "neutral", "sad", "hate", "surprised"]
    # emotions = ["angry"]  # 测试用
    
    for emotion in emotions:
        emotion_path = os.path.join(root_dir, emotion)
        
        # 检查目录是否存在
        if not os.path.isdir(emotion_path):
            continue  # 跳过不存在的目录
        
        wav_files = []
        
        try:
            # 遍历目录中的文件
            for entry in os.listdir(emotion_path):
                entry_path = os.path.join(emotion_path, entry)
                
                # 检查是否是文件且以.wav或.WAV结尾
                if os.path.isfile(entry_path) and entry.lower().endswith('.wav'):
                    wav_files.append(entry_path)
                    
        except (PermissionError, OSError):
            # 跳过无法访问的目录
            continue
            
        emotion_files[emotion] = wav_files
    
    return emotion_files

# 初始化语音检测器
curDir = os.path.dirname(os.path.abspath(__file__))
# audios_dir = os.path.join(curDir, "jialin_pi_class_dataset", "segments3")
audios_dir = "/data/applechang/emotion_audio_dataset/validate/segments9"
emotionFiles = traverse_emotion_voice(audios_dir)
process_emotion_files(emotion_files=emotionFiles)