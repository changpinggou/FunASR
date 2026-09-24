from funasr import AutoModel
import torch
import time
import numpy as np
import requests
from io import BytesIO
import soundfile as sf
import re
from voice_detector import VoiceDetector

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class VoiceDetectorEmotionSenceVoice(VoiceDetector):
    def __init__(self, model_id="iic/SenseVoiceSmall", system_prompt=""):
        """
        初始化语音检测器
        
        Args:
            model_id: 模型路径
            system_prompt: 系统提示词
        """
        print("Initializing VoiceDetectorEmotion...")
        super().__init__(model_id, system_prompt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"GPU available: {device}")
        
        # 加载模型
        startTime = time.time()

        self.model = AutoModel(
            model=model_id,
            trust_remote_code=True,
            remote_code="./model.py",
            disable_update=True,
            device="cuda:0",
        )
        
        self.system_prompt = system_prompt
        
        # 加载耗时
        elapsed_time = time.time() - startTime
        print(f"Model loaded. Elapsed time: {elapsed_time:.2f} seconds")
        
    def extract_content(self, raw_str):
        try:
            # 1. 析取所有标签 (匹配 <| 和 |> 之间的内容)
            tags = re.findall(r'<\|(.*?)\|>', raw_str)
            
            # 2. 析取最后的文本内容 (匹配最后一个 |> 之后的所有字符)
            text_content = re.sub(r'<\|.*?\|>', '', raw_str).strip()
            
            return {
                "language": tags[0] if len(tags) > 0 else None,
                "emotion": tags[1] if len(tags) > 1 else None,
                "type": tags[2] if len(tags) > 2 else None,
                "other_id": tags[3] if len(tags) > 3 else None,
                "text": text_content
            }
        except Exception as e:
            return None
        
 
    def process_chat_generation(self, audioPath):
        """
        处理聊天生成
        
        Args:
            messages: 消息列表
        
        Returns:
            tuple: (生成的文本, 耗时)
        """

        startTime = time.time()
        res = self.model.generate(
            input=audioPath,
            cache={},
            language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
        )
        elapsed_time = round((time.time() - startTime) * 1000, 2)

        print(">>> model output: \n", res)
        # text = rich_transcription_postprocess(res[0]["text"])
        result = self.extract_content(res[0]["text"])
        if result == None:
            return "", 0, 0
        return result["emotion"], 0, elapsed_time

    def detect(self, audio_source, source_type="buffer", user_prompt=None)->dict:
        """
        主要的检测接口
        
        Args:
            audio_source: 音频源
            source_type: 音频源类型
            user_prompt: 自定义提示词
        
        Returns:
            dict: 包含结果和耗时的字典
        """
        try:
            # 生成提示词
            prompt_text = "" if user_prompt is None else user_prompt
            # 处理并返回结果
            result_emotion, score, elapsed_time = self.process_chat_generation(audio_source)

            result_emotion_modify = result_emotion
            if  result_emotion == "HAPPY":
                result_emotion_modify = "happy"
            elif result_emotion == "SAD":
                result_emotion_modify = "sad"
            elif result_emotion == "ANGRY":
                result_emotion_modify = "angry"
            elif result_emotion == "NEUTRAL":
                result_emotion_modify = "neutral"
            elif result_emotion == "FEARFUL":
                result_emotion_modify = "fear"
            elif result_emotion == "DISGUSTED":
                result_emotion_modify = "hate"
            elif result_emotion == "SURPRISED":
                result_emotion_modify = "surprised"


            print(f"Detection completed.\n{result_emotion_modify}: {score}, Elapsed time: {elapsed_time} ms")
            return {
                "result": result_emotion_modify,
                "score": round(score, 6),
                "elapsed_time": elapsed_time,
                "prompt": prompt_text,
                "success": True
            }
        except Exception as e:
            return {
                "result": f"Error: {str(e)}",
                "score": 0,
                "elapsed_time": 0,
                "success": False
            }
    
    def warmup(self):
        """
        预热模型
        """
        # 创建一个简短的随机音频进行预热
        random_audio = np.random.randn(16000)  # 1秒的随机音频
        messages = random_audio
        
        self.process_chat_generation(messages)
        print("Model warmup completed.")
