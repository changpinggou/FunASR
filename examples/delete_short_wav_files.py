import os
import wave
import contextlib

def delete_short_wav_files(directory, min_duration=1.0):
    """
    遍历目录下所有.wav文件，删除时长少于指定秒数的文件
    
    Args:
        directory: 要遍历的目录路径
        min_duration: 最小时长（秒），默认1秒
    """
    # 统计信息
    total_files = 0
    deleted_files = 0
    error_files = 0
    
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                try:
                    # 获取音频时长
                    duration = get_wav_duration(file_path)
                    
                    if duration < min_duration:
                        # 删除文件
                        os.remove(file_path)
                        deleted_files += 1
                        print(f"已删除: {file_path} (时长: {duration:.2f}秒)")
                    else:
                        print(f"保留: {file_path} (时长: {duration:.2f}秒)")
                        
                except Exception as e:
                    error_files += 1
                    print(f"处理文件出错 {file_path}: {e}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"总文件数: {total_files}")
    print(f"已删除文件: {deleted_files}")
    print(f"出错文件: {error_files}")

def get_wav_duration(file_path):
    """
    获取wav文件的时长（秒）
    """
    with contextlib.closing(wave.open(file_path, 'rb')) as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

if __name__ == "__main__":
    # 使用示例
    target_directory = "/data/applechang/emotion_audio_dataset/applechang_2_25_classify"  # 替换为你的目录路径
    delete_short_wav_files(target_directory, min_duration=0.8)