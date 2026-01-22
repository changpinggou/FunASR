
from abc import ABC, abstractmethod
import time
import numpy as np
import requests
from io import BytesIO
import soundfile as sf
import math
import librosa



class VoiceDetector(ABC):
    def __init__(self, model_id="", system_prompt=""):
        """
        初始化语音检测器
        
        Args:
            model_id: 模型路径
            system_prompt: 系统提示词
        """
        self.model_id = model_id
        self.system_prompt = system_prompt
    
    def get_name(self)->str:
        """
        获取检测器名称
        
        Returns:
            str: 检测器名称
        """
        return self.model_id
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def warmup(self):
        """
        预热模型
        """
        pass

    def get_default_prompt(self)->str:
        return "这个模型不支持自定义提示词"

    def load_audio_from_path(self, audio_path, data_type='float32'):
        """从文件路径加载音频"""
        audio_data, sample_rate = sf.read(audio_path, dtype=data_type)
        return audio_data, sample_rate
    
    def load_audio_from_url(self, audio_url, data_type='float32'):
        """从URL加载音频"""
        response = requests.get(audio_url)
        response.raise_for_status()
        audio_data, sample_rate = sf.read(BytesIO(response.content), dtype=data_type)
        return audio_data, sample_rate
    
    def preprocess_audio(self, audio_source, source_type="buffer", data_type='float32', target_sample_rate=16000):
        """
        预处理音频数据
        
        Args:
            audio_source: 音频源，可以是文件路径、URL或音频buffer
            source_type: 音频源类型，可选值："path", "url", "buffer"
        
        Returns:
            处理后的音频数据
        """
        if source_type == "path":
            audio_data, sample_rate = self.load_audio_from_path(audio_source, data_type)
            if sample_rate != target_sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            # 如果是双声道, 改成单声道
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
        elif source_type == "url":
            audio_data, sample_rate = self.load_audio_from_url(audio_source, data_type)
            if sample_rate != target_sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
            # 如果是双声道, 改成单声道
            if audio_data.ndim == 2:
                audio_data = np.mean(audio_data, axis=1)
        elif source_type == "buffer":
            # 假设输入已经是正确格式的numpy数组
            if isinstance(audio_source, np.ndarray):
                audio_data = audio_source.astype(data_type)
                # 如果是双声道, 改成单声道
                if audio_data.ndim == 2:
                    audio_data = np.mean(audio_data, axis=1)
            else:
                raise TypeError("Buffer must be a numpy array")
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return audio_data