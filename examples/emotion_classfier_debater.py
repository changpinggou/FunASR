import os
import sys
import csv
import logging
import shutil  # 新增：用于文件复制
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import wave
import contextlib

# 生成符合Go语言格式的时间戳
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"emotion_test_{timestamp}.log"

full_log_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file_name)

# 配置日志输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename= full_log_file_name,
    filemode='a'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 假设这两个类在本地文件中存在，保持引用不变
from voice_detector_emotion import VoiceDetectorEmotion
from voice_detector_emotion_sencevoice import VoiceDetectorEmotionSenceVoice

voice_detector_emotion2vec = VoiceDetectorEmotion(model_id="iic/emotion2vec_plus_large")
voice_detector_sencevoice = VoiceDetectorEmotionSenceVoice(model_id="iic/SenseVoiceSmall")


def emotion_prediction(voice_path: str) -> Tuple[str, Exception]:
    """情感推理任务"""
    try:
        result_emotion2vec_dict = voice_detector_emotion2vec.detect(voice_path, source_type="path")
        result_sencevoice_dict = voice_detector_sencevoice.detect(voice_path, source_type="path")
        # 过滤掉 prompt
        if "prompt" in result_emotion2vec_dict:
            del result_emotion2vec_dict["prompt"]
        
        emotion2vec_result = result_emotion2vec_dict.get('result', '')
        emotion2vec_score = result_emotion2vec_dict.get('score')
        elapse = result_emotion2vec_dict.get('elapsed_time')
        # 处理结果字符串，确保拿到纯净的标签
        emotion2vec_result = emotion2vec_result.split('/')[-1]
        if emotion2vec_result == 'disgusted':
            emotion2vec_result = 'hate'
        elif emotion2vec_result == 'fearful':
            emotion2vec_result = 'fear'

        sencevoice_result = result_sencevoice_dict.get('result', '')
        final_result = "other"
        
        if emotion2vec_result == '<unk>' and sencevoice_result != 'EMO_UNKNOWN':
            final_result = sencevoice_result
            logger.info("[applechang-test-emotion-predict] emotion2vec_result == '<unk>' and sencevoice_result != 'EMO_UNKNOWN', final_result=%s", final_result)

        elif emotion2vec_result == sencevoice_result:
            final_result = emotion2vec_result
            logger.info("[applechang-test-emotion-predict] emotion2vec_result == sencevoice_result, final_result=%s", final_result)
        elif emotion2vec_result != '<unk>' and emotion2vec_score > 0.8 :
            final_result = emotion2vec_result
            logger.info("[applechang-test-emotion-predict] emotion2vec_score > 0.8, final_result=%s, emotion2vec_score=%f",final_result, emotion2vec_score)

        # logger.info("[emotion-predict] finalEmotion=%s, emotion2vec_result=%s, sencevoice_result=%s, elapse=%d ms, score=%s,  file=%s", 
        #             final_result, emotion2vec_result, sencevoice_result, elapse, emotion2vec_score, voice_path)

        return final_result, None

    except Exception as e:
        logger.error(f"Prediction failed for {voice_path}: {e}")
        return "", e
    
def get_wav_duration(file_path):
    """
    获取wav文件的时长（秒）
    """
    with contextlib.closing(wave.open(file_path, 'rb')) as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration
    
def traverse_emotion_voice(root_dir: str, target_dir: str):
    """
    遍历情感语音目录，根据预测结果将文件复制到 target_dir 下对应的子目录中。
    
    Args:
        root_dir: 源文件根目录
        target_dir: 目标分类根目录
    """
    # 允许的情感范围（白名单）
    valid_emotions = {"angry", "fear", "happy", "neutral", "sad", "hate", "surprised", "other"}

    # 检查源目录是否存在
    if not os.path.isdir(root_dir):
        logger.error(f"Root dir not found: {root_dir}")
        return
    
    # 确保目标总目录存在，不存在则创建
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create target directory {target_dir}: {e}")
            return

    logger.info(f"Start traversing from {root_dir} to {target_dir}")

    # 统计信息
    stats = {
        'total_files': 0,
        'skipped_short': 0,
        'skipped_error': 0,
        'processed': 0,
        'other': 0
    }

    try:
        # 遍历目录中的文件
        for entry in os.listdir(root_dir):
            entry_path = os.path.join(root_dir, entry)
            
            # 检查是否是文件且以.wav或.WAV结尾
            if os.path.isfile(entry_path) and entry.lower().endswith('.wav'):
                stats['total_files'] += 1
                try:
                    duration = get_wav_duration(entry_path)
                    if duration < 0.8:
                        logger.info(f"Skipping {entry}: Duration {duration:.2f}s < 0.8s")
                        stats['skipped_short'] += 1
                        continue
                except Exception as e:
                    logger.warning(f"Skipping {entry}: Failed to get audio duration - {e}")
                    stats['skipped_error'] += 1
                    continue
        
                # 1. 获取情感预测结果
                predicted_emotion, error = emotion_prediction(entry_path)
                
                if error or not predicted_emotion: 
                    logger.warning(f"Skipping {entry}: Prediction failed or empty.")
                    stats['skipped_error'] += 1
                    continue
                
                # 2. 标准化情感标签 (转小写，去空格)
                clean_emotion = predicted_emotion.lower().strip()

                # 3. 确定目标文件夹名称
                # 如果预测结果在白名单内，直接使用；否则归类为 "other"
                if clean_emotion in valid_emotions:
                    target_folder_name = clean_emotion
                else:
                    logger.warning(f"Emotion '{clean_emotion}' for {entry} is not in valid list, moving to 'other'.")
                    target_folder_name = "other"
                    stats['other'] += 1
                
                # 4. 构建目标路径
                target_emotion_dir = os.path.join(target_dir, target_folder_name)
                
                # 5. 创建对应的子目录 (如果不存在)
                if not os.path.exists(target_emotion_dir):
                    os.makedirs(target_emotion_dir, exist_ok=True)
                
                # 6. 执行复制操作
                target_file_path = os.path.join(target_emotion_dir, entry)
                try:
                    # 使用 copy2 保留文件的时间戳等元数据
                    shutil.copy2(entry_path, target_file_path)
                    logger.info(f"Copied {entry} -> {target_folder_name}")
                    stats['processed'] += 1
                except Exception as copy_err:
                    logger.error(f"Failed to copy {entry}: {copy_err}")
                    stats['skipped_error'] += 1
      
    except (PermissionError, OSError) as e:
        logger.error(f"Error traversing directory: {e}")

    # 打印统计信息
    logger.info(f"Processing completed. Stats: {stats}")

# --- Main Execution ---

curDir = os.path.dirname(os.path.abspath(__file__))
# 源音频目录
audios_dir = "/data/applechang/emotion_audio_dataset/applechang_2_25/三国演义"
# 目标分类目录 (请根据实际情况修改)
classified_dir = "/data/applechang/emotion_audio_dataset/applechang_2_25_classify"

if __name__ == "__main__":
    logger.info("Task started.")
    traverse_emotion_voice(audios_dir, classified_dir)
    logger.info("Task finished.")