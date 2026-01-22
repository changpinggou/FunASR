'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted_
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''
import torch
from funasr import AutoModel
import soundfile as sf
import numpy as np
import json
import os

def convert_npy_to_json(npy_path, json_path):
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

model_id="iic/emotion2vec_base"
# model_id="iic/emotion2vec_base_finetuned"
# model="iic/emotion2vec_plus_seed"
# model="iic/emotion2vec_plus_base"
# model_id = "iic/emotion2vec_plus_large"



model = AutoModel(
    model=model_id,
    disable_update=True,
    vad_model=None,
    quantize=False,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)


wav_file = f"/root/Kimi-Audio/test_audios2/温柔.wav"

# from voice_detector_emotion import float2bytes_int16, get_max_label_score

# rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
# print(get_max_label_score(rec_result))

# audio_data, sample_rate = sf.read(wav_file)
# audio_data = float2bytes_int16(audio_data)
# rec_result = model.generate(audio_data, output_dir="./outputs", granularity="utterance", extract_embedding=False)
# print(get_max_label_score(rec_result))


curDir = os.path.dirname(os.path.abspath(__file__))
export_model_dir = os.path.join(curDir, "emotion2vec_base_onnx")
os.makedirs(export_model_dir, exist_ok=True)

# 导出权重
print(np.__version__) # 检查是否能打印版本
# 如果还是报错，尝试用这个方式转换：
weight = model.model.proj.weight.detach().cpu()
bias = model.model.proj.bias.detach().cpu()

# 强制使用 torch 保存，之后再转 npy
torch.save(weight, "weight.pt")
torch.save(bias, "bias.pt")

# 或者使用这种写法尝试转换
weight_np = np.array(weight.tolist(), dtype=np.float32)
bias_np = np.array(bias.tolist(), dtype=np.float32)

linear_weight_npy = os.path.join(export_model_dir,"linear_weight.npy")
linear_bias_npy = os.path.join(export_model_dir,"linear_bias.npy")

linear_weight_json = os.path.join(export_model_dir,"linear_weight.json")
linear_bias_json = os.path.join(export_model_dir,"linear_bias.json")

np.save(linear_weight_npy, weight_np)
np.save(linear_bias_npy, bias_np)

convert_npy_to_json(linear_weight_npy, linear_weight_json)
convert_npy_to_json(linear_bias_npy, linear_bias_json)


# # 导出模型
model.export(output_dir=export_model_dir, type="onnx")

