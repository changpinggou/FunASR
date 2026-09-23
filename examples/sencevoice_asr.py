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
        # result_emotion2vec_dict = voice_detector_emotion2vec.detect(voice_path, source_type="path")
        result_sencevoice_dict = voice_detector_sencevoice.detect(voice_path, source_type="path")
        # 过滤掉 prompt
        # if "prompt" in result_emotion2vec_dict:
        #     del result_emotion2vec_dict["prompt"]
        
        # emotion2vec_result = result_emotion2vec_dict.get('result', '')
        # emotion2vec_score = result_emotion2vec_dict.get('score')
        # elapse = result_emotion2vec_dict.get('elapsed_time')
        # 处理结果字符串，确保拿到纯净的标签
        # emotion2vec_result = emotion2vec_result.split('/')[-1]
        # if emotion2vec_result == 'disgusted':
        #     emotion2vec_result = 'hate'
        # elif emotion2vec_result == 'fearful':
        #     emotion2vec_result = 'fear'

        sencevoice_result = result_sencevoice_dict.get('result', '')
        final_result = "other"
        
        # if emotion2vec_result == '<unk>' and sencevoice_result != 'EMO_UNKNOWN':
        #     final_result = sencevoice_result
        #     logger.info("[applechang-test-emotion-predict] emotion2vec_result == '<unk>' and sencevoice_result != 'EMO_UNKNOWN', final_result=%s", final_result)

        # elif emotion2vec_result == sencevoice_result:
        #     final_result = emotion2vec_result
        #     logger.info("[applechang-test-emotion-predict] emotion2vec_result == sencevoice_result, final_result=%s", final_result)
        # elif emotion2vec_result != '<unk>' and emotion2vec_score > 0.8 :
        #     final_result = emotion2vec_result
        #     logger.info("[applechang-test-emotion-predict] emotion2vec_score > 0.8, final_result=%s, emotion2vec_score=%f",final_result, emotion2vec_score)

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
    

# --- Main Execution ---

curDir = os.path.dirname(os.path.abspath(__file__))
# 源音频目录
audios_dir = "/data/applechang/emotion_audio_dataset/applechang_2_25/三国演义"
# 目标分类目录 (请根据实际情况修改)
classified_dir = "/data/applechang/emotion_audio_dataset/applechang_2_25_classify"

if __name__ == "__main__":
    logger.info("Task started.")
    emotion_prediction("/data/applechang/FunASR/examples/applechang_audios/zsr_80s_0.wav")
    logger.info("Task finished.")