import os
import random

import os
import random

def prepare_funasr_data_recursive(data_dir, output_root, train_ratio=0.9):
    # 定义情感名称到数字索引的映射
    emo_map = {
        "angry": 0,
        "hate" : 1,
        "fear" : 2,
        "happy": 3,
        "neutral":4,
        "sad":6,
        "surprised":7,
        "unknow":8
    }


    all_data = []

    # 1. 第一层遍历：遍历 dataset 下的各个子目录 (如 project1, project2...)
    for sub_dir in os.listdir(data_dir):
        sub_dir_path = os.path.join(data_dir, sub_dir)
        
        if not os.path.isdir(sub_dir_path):
            continue

        # 2. 第二层遍历：遍历子目录下的情绪文件夹 (angry, happy...)
        for emo_name, emo_idx in emo_map.items():
            emo_folder = os.path.join(sub_dir_path, emo_name)
            
            if not os.path.exists(emo_folder):
                continue
            
            # 3. 遍历音频文件
            for file in os.listdir(emo_folder):
                if file.endswith(".wav"):
                    wav_path = os.path.abspath(os.path.join(emo_folder, file))
                    
                    # 生成唯一 ID：子目录名_情绪_原文件名 (防止重名)
                    file_basename = os.path.splitext(file)[0]
                    utt_id = f"{sub_dir}_{emo_name}_{file_basename}"
                    
                    all_data.append((utt_id, wav_path, emo_idx))

    if not all_data:
        print("未找到任何符合条件的 .wav 文件，请检查目录结构和 emo_map 映射。")
        return

    # 4. 打乱并划分
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    def write_to_dir(data_list, subset):
        subset_dir = os.path.join(output_root, subset)
        os.makedirs(subset_dir, exist_ok=True)
        
        with open(os.path.join(subset_dir, "wav.scp"), "w", encoding="utf-8") as f_wav, \
             open(os.path.join(subset_dir, "text"), "w", encoding="utf-8") as f_txt:
            for utt_id, wav_path, label in data_list:
                f_wav.write(f"{utt_id} {wav_path}\n")
                f_txt.write(f"{utt_id} {label}\n")
        
        print(f"完成 {subset} 集生成: {len(data_list)} 条数据")

    write_to_dir(train_data, "train")
    write_to_dir(val_data, "val")

if __name__ == "__main__":
    # dataset 目录下有多个子目录，如 dataset/part1/angry/, dataset/part2/happy/ ...
    prepare_funasr_data_recursive(
        data_dir="/data/applechang/emotion_audio_dataset/jialin", 
        output_root="/data/applechang/FunASR/examples"
    )