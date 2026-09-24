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

from voice_detector_emotion_sencevoice import VoiceDetectorEmotionSenceVoice

voice_detector_sencevoice = VoiceDetectorEmotionSenceVoice(model_id="iic/SenseVoiceSmall")


def emotion_prediction(voice_path: str) -> Tuple[str, Exception]:
    """情感推理任务"""
    try:

        result_sencevoice_dict = voice_detector_sencevoice.detect(voice_path, source_type="path")

        sencevoice_result = result_sencevoice_dict.get('result', '')
        final_result = "other"
        

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
    

# --- Main Execution ---

curDir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    logger.info("Task started.")
    emotion_prediction("/Users/zego/Documents/zsr/FunASR/examples/zsr_80s_0.wav")
    logger.info("Task finished.")