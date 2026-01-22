import onnxruntime as ort
import numpy as np
# import soundfile as sf
import time
import math
import os
import json
# import array_to_file
from voice_detector import VoiceDetector
# 情感标签映射
EMOTION_LABELS = [
    "生气",
    "厌恶",
    "恐惧",
    "开心",
    "中立",
    "其他",
    "难过",
    "吃惊",
    "未知"
]

def get_max_label_score(scores, labels):
    """
    从推理结果中提取得分最高的标签及其分数。
    
    参数:
        scores (list or np.ndarray): 情感分类的分数列表或数组。
        labels (list): 对应的情感标签列表。
    
    返回:
        tuple: 得分最高的标签及其对应的分数，格式为 (label, score)。
    """
    
    # 确保scores不为空且是有效的numpy数组或列表
    if isinstance(scores, np.ndarray):
        if scores.size == 0:
            return None, None
    elif not scores:  # 处理普通列表
        return None, None
    
    # 找到最大分数的索引
    max_index = np.argmax(scores)
    
    # 返回对应的 label 和 score
    return labels[max_index], scores[max_index]



class VoiceDetectorEmotion(VoiceDetector):
    def __init__(self, model_path="/data/chat_engine/emotion2vec_export_finetuned/emotion2vec.onnx"):
        """
        初始化ONNX情感检测器
        
        Args:
            model_path: ONNX模型路径
        """
        print(f"Initializing ONNXEmotionDetector with model: {model_path}...")
        super().__init__(model_id="emotion2vec_plus_large")
        startTime = time.time()
        
        # add by applechang
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # session_options.optimized_model_path = "/root/Kimi-Audio/emotion2vec_model/optimized_model.onnx"  # 保存优化后的模型
    
        # 内存优化
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True

        # 创建ONNX运行时会话
        self.ort_session = ort.InferenceSession(model_path, session_options)

        dir = os.path.dirname(model_path)
        self.linear_weight = np.load(os.path.join(dir, 'linear_weight.npy'))
        self.linear_bias = np.load(os.path.join(dir, 'linear_bias.npy'))

        
        # 获取模型输入输出信息
        inputs = self.ort_session.get_inputs()
        print(inputs)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        outputs = self.ort_session.get_outputs()
        self.output_names = [output.name for output in outputs]
        
        print(f"Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"Outputs: {self.output_names}")
        
        elapsed_time = time.time() - startTime
        print(f"Model loaded. Elapsed time: {elapsed_time:.2f} seconds")

    def convert_npy_to_json(self, npy_path, json_path):
        data = np.load(npy_path)
        
        # 如果是二维数组，转换为嵌套列表
        if len(data.shape) == 2:
            result = data.tolist()
        else:
            result = data.tolist()
        
        with open(json_path, 'w') as f:
            json.dump(result, f)
        
        print(f"Converted {npy_path} to {json_path}")
        print(f"Original shape: {data.shape}")
        
    def postprocess(self, outputs):
        """
        后处理ONNX模型的输出结果，逻辑与PyTorch模型一致
        
        Args:
            outputs: 模型输出，格式为[batch_size, sequence_length, embedding_dim]
                    对于emotion2vec模型，输出通常是[1, seq_len, 1024]的3维张量
        
        Returns:
            tuple: (scores, labels)，情感得分列表和对应标签列表
        """
        print(f"Postprocessing outputs of shape: {[o.shape for o in outputs]}")
        
        # 获取第一个输出（通常是特征嵌入）
        x = outputs[0]
        print(f"Embeddings shape: {x.shape}")
        
        # 计算序列的平均嵌入 (对应PyTorch代码中的 x.mean(dim=1))
        x = x.mean(axis=1)
        print(f"Average embedding shape: {x.shape}")
        # array_to_file.numpy_array_to_file(x, '/root/Kimi-Audio/emotion2vec_export/numpy_x.txt')
        # 模拟线性投影层 (对应PyTorch代码中的 self.proj)
        # 创建一个简单的线性层权重和偏置（在实际应用中应该从模型中加载）
        embedding_dim = x.shape[-1]
        vocab_size = len(EMOTION_LABELS)

        # 执行线性投影: x = x @ proj_weight.T + proj_bias
        x = np.dot(x, self.linear_weight.T) + self.linear_bias
        print(f"After projection shape: {x.shape}")
        
        # 应用softmax归一化 (对应PyTorch代码中的 torch.softmax(x, dim=-1))
        # 为了数值稳定性，先减去最大值
        x_max = np.max(x, axis=-1, keepdims=True)
        x_exp = np.exp(x - x_max)
        x = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
        
        # 获取第一个样本的分数列表 (对应PyTorch代码中的 x[0].tolist())
        scores = x[0].tolist()
        print(f"All scores: {[f'{v:.6f}' for v in scores]}")
        
        return scores, EMOTION_LABELS
    
    def detect(self, audio_source, source_type="buffer", user_prompt=None)->dict:
        """
        进行情感推理
        
        Args:
            audio_path: 音频文件路径（二选一）
            audio_data: 音频数据（numpy数组，二选一）
        
        Returns:
            dict: 包含推理结果的字典
        """
        try:
            # 预处理音频
            audio_data = self.preprocess_audio(audio_source, source_type, data_type='float32')

            print(f"Processed audio shape: {audio_data.shape}")
            # 扩展维度以匹配模型输入要求 (batch_size, sequence_length)
            audio_input = audio_data[np.newaxis, :]
            
            # 运行推理
            startTime = time.time()
            
            # 使用预处理后的音频数据作为输入（2维数组：[batch_size, sequence_length]）
            outputs = self.ort_session.run(self.output_names, {self.input_name: audio_input})
            # 输出格式为 [batch_size, sequence_length,1024]
            print(f"Inference outputs shape: {[o.shape for o in outputs]}")
            
            # 对结果做后处理
            scores, labels = self.postprocess(outputs)

            elapsed_time = time.time() - startTime
            elapsed_time = round((time.time() - startTime) * 1000, 2)
            result, score = get_max_label_score(scores, labels)
            score = str(round(score, 6))
            print(f"Detection completed.\n{result}: {score}, Elapsed time: {elapsed_time} ms")
            # 把 scores 和 labels 合并成一条文本消息, 标签:得分这样子
            scores_labels = [f"{label}:{score:.6f}" for label, score in zip(labels, scores)]
            scores_labels = ", ".join(scores_labels)

            return {
                "result": result + ":"+ score,
                "score": score,
                "elapsed_time": elapsed_time,
                "prompt": user_prompt,
                "success": True
            }
        except Exception as e:
            print(f"Error in detect: {e}")
            return {
                "result": str(e),
                "score": 0,
                "elapsed_time": 0,
                "prompt": user_prompt,
                "success": False
            }
    def warmup(self):
        """
        预热模型
        """
        # 创建一个简短的随机音频进行预热
        dummy_audio = np.random.randn(16000).astype(np.float32)
        # 扩展维度以匹配模型输入要求 (batch_size, sequence_length)
        dummy_input = dummy_audio[np.newaxis, :]
        
        # 运行预热推理
        self.ort_session.run(self.output_names, {self.input_name: dummy_input})


def main():
    # 模型路径
    model_path = "/root/Kimi-Audio/emotion2vec_model/emotion2vec_plus_large-int8.onnx"
    
    # 测试音频路径
    test_audio_path = "/root/Kimi-Audio/test_audios2/温柔.wav"
    
    # 创建检测器
    detector = VoiceDetectorEmotion(model_path)
    
    # 进行推理
    print(f"Inferencing on {test_audio_path}...")
    result = detector.detect(test_audio_path, source_type="path")
    
    print(f"\nInference Results: {result}")

if __name__ == "__main__":
    main()
    
        