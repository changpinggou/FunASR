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
import re

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


import re

def universal_itn_repair(text, strict_mode=True):
    """
    通用的 ASR 中英文混合数字/序数词修复程序。
    
    :param text: 原始 ASR 识别文本
    :param strict_mode: 严格模式。开启时，仅修复存在“中英混杂”或“明显缺失进位”的异常数字块。
                        建议开启，否则会将“万一”、“一丝不苟”等成语/词语中的汉字强行转为数字。
    """
    
    digit_map = {
        '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '零': 0
    }
    unit_map = {'十': 10, '百': 100, '千': 1000}
    large_unit_map = {'万': 10000, '亿': 100000000}

    def parse_number_block(s):
        # 1. 如果已经是纯阿拉伯数字，直接放行
        if s.isdigit():
            return s
            
        # 2. 严格模式拦截：如果没有阿拉伯数字参与混合，且不是类似"二零"这种按位读法，直接跳过以防误伤成语
        if strict_mode:
            has_arabic = any(c.isdigit() for c in s)
            is_positional_anomaly = ('0' in s or '零' in s) and len(s) > 1
            if not (has_arabic or is_positional_anomaly):
                return s

        # 3. 判断是否包含单位（十百千万亿）
        has_unit = any(u in s for u in unit_map) or any(u in s for u in large_unit_map)
        
        # 策略 A：按位直译（无单位的情况，例如 "二0", "三2", "二零二四"）
        if not has_unit:
            res = ""
            for char in s:
                if char in digit_map:
                    res += str(digit_map[char])
                else:
                    res += char
            return res
            
        # 策略 B：算术累加 FSM（有单位的情况，例如 "十7", "3十2", "100万"）
        # 将字符串切分为连续的数字和单位 token
        tokens = re.findall(r'\d+|[一二两三四五六七八九十百千万亿零]', s)
        total = 0
        section = 0
        curr_val = 0
        
        for token in tokens:
            if token.isdigit():
                curr_val = int(token)
            elif token in digit_map:
                if token == '零':
                    pass # 零在此算法中仅占位，不影响累加
                else:
                    curr_val = digit_map[token]
            elif token in unit_map:
                # 处理“十7”这种情况：十前面没有数字，默认乘数为1
                multiplier = curr_val if curr_val != 0 else 1
                section += multiplier * unit_map[token]
                curr_val = 0
            elif token in large_unit_map:
                multiplier = curr_val if curr_val != 0 else 1
                section += multiplier
                total = (total + section) * large_unit_map[token]
                section = 0
                curr_val = 0
                
        section += curr_val
        total += section
        return str(total)

    # 捕获所有连续的中英文数字和单位（不依赖"第"字）
    pattern = r"[一二两三四五六七八九十百千万亿零0-9]+"
    
    return re.sub(pattern, lambda m: parse_number_block(m.group(0)), text)

# ==========================================
# 通用鲁棒性测试
# ==========================================

# test_cases = [
#     # 场景 1: ASR 典型中英混杂幻觉 (无"第"字前缀)
#     "我有十7个苹果",                # 预期: 我有17个苹果
#     "距离是3十2公里",               # 预期: 距离是32公里
#     "一共一百2十3人",               # 预期: 一共123人
#     "播放第二0首歌曲",              # 预期: 播放第20首歌曲
    
#     # 场景 2: ASR 大数混杂 (例如：100万，2千)
#     "成本大约是100万人民币",        # 预期: 成本大约是1000000人民币
#     "高度为2千米",                  # 预期: 高度为2000米
    
#     # 场景 3: 年份或编号（无单位按位直译）
#     "现在是二零二四年",             # 预期: 现在是2024年
#     "密码是二0四八",                # 预期: 密码是2048
    
#     # 场景 4: 严格模式防护（防误杀成语与日常用语）
#     "万一失败了怎么办",             # 预期: 万一失败了怎么办 (不变成 10001)
#     "这事必须一丝不苟",             # 预期: 这事必须一丝不苟 (不变成 1丝不苟)
#     "他是个十二岁的孩子",           # 预期: 他是个十二岁的孩子 (纯中文，交给ASR原样输出)
    
#     # 场景 5: 用户原始用例测试 (带"第"字也能兼容)
#     "第十8句一直往下说不要停顿",    # 预期: 第18句一直往下说不要停顿
#     "第三2句不要停顿",              # 预期: 第32句不要停顿
# ]

# print("--- 修复结果 ---")
# for case in test_cases:
#     fixed = universal_itn_repair(case, strict_mode=True)
#     print(f"原句: {case:<20} => 修复: {fixed}")


def emotion_prediction(voice_path: str) -> Tuple[str, Exception]:
    """情感推理任务"""
    try:

        raw_txt = '第一句一直往下说，不要停顿，现在是连续语音测试。第二句一直往下说，不要停顿，现在是连续语音测试。第三句一直往下说不要停顿，现在是连续语音测试。第四句一直往下说，不要停顿，现在是连续语音测试。第五句一直往下说不要停顿，现在是连续语音测试。第六句一直往下说，不要停顿，现在是连续语音顿，现在是连续语音测试。第十7句一直往下说不要停顿，现在是连续语音测试。第十8句一直往下说不要停顿，现在是连续语音测试。第十九句一直往下说不要停顿，现在是连续语音测试。第二0句一直往下说不要停顿，现在是连续语音测试。第21句一直往下说。不要停顿。现在是连续语音测试。第22测试。第三2句一直往下说不要停顿，现在是连续语音测试。第三3句一直往下说，不要停顿，现在是连续语音测试。第三4句一直往下说不要停顿，现在是连续语音测试。第35句一直往下说，不要停顿，现在是连续语音测试。第30。'
        raw_test_woint= '第一句一直往下说不要停顿现在是连续语音测试第二句一直往下说不要停顿现在是连续语音测试第三句一直往下说不要停顿现在是连续语音测试第四句一直往下说不要停顿现在是连续语音测试第五句一直往下说不要停顿现在是连续语音测试第六句一直往下说不要停顿现在是连续语音顿现在是连续语音测试第十七句一直往下说不要停顿现在是连续语音测试第十八句一直往下说不要停顿现在是连续语音测试第十九句一直往下说不要停顿现在是连续语音测试第二十句一直往下说不要停顿现在是连续语音测试第二十一句一直往下说不要停顿现在是连续语音测试第二十二测试第三十二句一直往下说不要停顿现在是连续语音测试第三十三句一直往下说不要停顿现在是连续语音测试第三十四句一直往下说不要停顿现在是连续语音测试第三十五句一直往下说不要停顿现在是连续语音测试第三十'
        re_text = universal_itn_repair(raw_txt)  # 修复混合数字

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