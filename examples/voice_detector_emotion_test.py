from funasr import AutoModel
import torch
import time
import numpy as np
import requests
from io import BytesIO
import soundfile as sf
import librosa
from voice_detector import VoiceDetector

def float2bytes_int16(wav_np: np.ndarray) -> bytes:
    """
    wav_np : float32，幅度范围 [-1.0, 1.0]
    返回   : bytes，每样本 2 字节，小端 int16
    """
    wav_np = np.asarray(wav_np, dtype=np.float32)
    # 1. 裁剪防溢出
    wav_np = np.clip(wav_np, -1.0, 1.0)
    # 2. 缩放并转 int16
    pcm16 = (wav_np * 32767).astype('<i2')   # '<i2' 等价于 np.int16
    # 3. 转字节
    return pcm16.tobytes()

def get_max_label_score(result):
    """
    从推理结果中提取得分最高的标签及其分数。

    参数:
        result (list): 包含推理结果的字典列表，每个字典包含 'labels' 和 'scores' 键。

    返回:
        tuple: 得分最高的标签及其对应的分数，格式为 (label, score)。
    """
    if not result:
        return None, None

    # 获取第一个结果中的 labels 和 scores
    labels = result[0]['labels']
    scores = result[0]['scores']

    # 找到最大分数的索引
    max_index = scores.index(max(scores))

    # 返回对应的 label 和 score
    return labels[max_index], scores[max_index]

class VoiceDetectorEmotion(VoiceDetector):
    def __init__(self, model_id="iic/emotion2vec_plus_large", system_prompt=""):
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
            disable_update=True,
            vad_model=None,
            hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
        )
        
        self.system_prompt = system_prompt
        
        # 加载耗时
        elapsed_time = time.time() - startTime
        print(f"Model loaded. Elapsed time: {elapsed_time:.2f} seconds")
        
    
    def process_chat_generation(self, nparray, abs_path):
        """
        处理聊天生成
        
        Args:
            messages: 消息列表
        
        Returns:
            tuple: (生成的文本, 耗时)
        """
        # nparray 转换成 pcm bytes
        audio_datas = float2bytes_int16(nparray)
        with torch.no_grad():
            
            startTime = time.time()
            # 下面代码段是直接调用FunASR 封装的推理函数
            output = self.model.generate(audio_datas, output_dir="./outputs", granularity="utterance", extract_embedding=False)
            
            # 下面代码段是用来对比上面直接调用self.model.generate函数, 总共分如下几步
            # 第一步：读取音频文件
            wav, sr = sf.read(abs_path)

            # 重采样很关键
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000) 
            
            # 限制最大长度 (例如 20秒)，防止单条音频撑爆显存
            if len(wav) > 16000 * 20:
                wav = wav[:16000 * 20]
            wav_tensor = torch.from_numpy(wav).float().to("cuda")
            wav_tensor = torch.nn.functional.layer_norm(wav_tensor, wav_tensor.shape) # 执行归一化
            wav_tensor = wav_tensor.unsqueeze(0) # 变为 (1, T)
            # 第二步：提取音频特征
            outputs = self.model.model.extract_features(wav_tensor, padding_mask=None)
            feats = outputs['x'] # (1, T_feat, 1024)
            # 句子级池化 (Mean Pooling)
            pooled_feat = feats.mean(dim=1)
            # 第三步：线性投影
            off_logits = self.model.model.proj(pooled_feat)
            # 第四步：找出最大值索引
            max_index = torch.argmax(off_logits, dim=-1).item()

            elapsed_time = round((time.time() - startTime) * 1000, 2)
            print(">>> model output: \n", output)
            # 提取模型生成的内容（通常是最后一部分）
            if output:
                result, score = get_max_label_score(output)
                return result, score, elapsed_time
            return "", 0, elapsed_time

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
            # 预处理音频
            audio_data = self.preprocess_audio(audio_source, source_type)
            
            # 生成提示词
            prompt_text = "" if user_prompt is None else user_prompt
            
            # 构建消息
            messages = audio_data
            
            # 处理并返回结果
            result, score, elapsed_time = self.process_chat_generation(messages, audio_source)
            print(f"Detection completed.\n{result}: {score}, Elapsed time: {elapsed_time} ms")
            return {
                "result": result,
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
