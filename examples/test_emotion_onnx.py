import os
import sys
import csv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from voice_detector_emotion_onnx import VoiceDetectorEmotion as VoiceDetector

from voice_detector_emotion_classfier_onnx import VoiceDetectorEmotion as VoiceDetector

curDir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(curDir, "emotion2vec_plus_large_onnx", "emotion2vec")
# 初始化语音检测器
voice_detector = VoiceDetector(model_path=model_path)
# 预热模型
# voice_detector.warmup()

print("=========================\n")

audios = []
results = []

# 读取 test_datas 下的所有wav文件保存到 audios 列表中

audio_path = os.path.join(curDir, "applechang_audios")
csv_file = os.path.join(curDir, "emotion_audios_results.csv")

for file in os.listdir(audio_path):
    if file.endswith(".wav"):
        audios.append(os.path.join(audio_path, file))

for audio in audios:
    print(f"Processing audio: {audio}")
    result_dict = voice_detector.detect(audio, source_type="path")
    # 过滤掉 prompt
    if "prompt" in result_dict:
        del result_dict["prompt"]
    print(f"[applechang-test] filePath:{audio}, Result: {result_dict}\n")
    print("=========================\n")
    
    # 提取文件名（不含路径和扩展名）
    file_name = os.path.basename(audio).split('.')[0]
    
    # 创建一个字典来存储当前音频的结果
    audio_result = {
        '文件名': file_name,
        '情感类别': result_dict.get('result', ''),
        '情感分数': result_dict.get('score', 0),
        '耗时(ms)': result_dict.get('elapsed_time', 0),
        '成功': result_dict.get('success', False)
    }
    
    # 将结果添加到results列表
    results.append(audio_result)

# 将结果保存到CSV文件

with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    if results:
        # 获取所有字段名
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据
        for result in results:
            writer.writerow(result)

print(f"结果已汇总到 {csv_file} 文件中。")
