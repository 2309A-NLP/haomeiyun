'''
Reference: https://github.com/alibaba-damo-academy/FunASR
pip install funasr
pip install modelscope
pip install -U rotary_embedding_torch
'''
# 实现了一个基于FunASR的语音识别（ASR）类，主要用于将音频文件转换为文本
try:
    from funasr import AutoModel
except:
    print("如果想使用FunASR，请先安装funasr，若使用Whisper，请忽略此条信息")
import os
import sys
sys.path.append('./')
from src.cost_time import calculate_time    

class FunASR:
    def __init__(self) -> None:
        # 定义模型的自定义路径
        model_path = "FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        # 主ASR模型(语音识别)：Paraformer大型ASR模型（中文，16kHz采样率）
        '''
        作用：将语音信号转换为文字（核心功能）
        好比：一个"听力"超好的翻译，把听到的声音写成文字
        特点：
            Paraformer架构（高性能非自回归模型）
            专门针对中文优化
            支持16kHz采样率（常见语音采样标准）
            词汇表包含8404个中文常用字词
        '''
        vad_model_path = "FunASR/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        # VAD模型：语音活动检测模型
        '''
        作用：检测音频中哪里有人说话，哪里是静音
        好比：像一个哨兵，标记出"这里有语音，该开始识别了"
        为什么需要：
            音频文件可能很长，中间有大量静音或背景噪音，VAD可以精确定位有效语音片段，提高识别效率
        '''
        punc_model_path = "FunASR/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        # 标点模型：添加标点符合的模型
        '''
        作用：为识别出的纯文字添加标点符号
        好比：像语文老师给文章加句号、逗号、问号
        为什么需要：
            ASR模型通常只输出文字，没有标点。比如：
            1.无标点："今天天气真好我们出去玩吧你吃饭了吗"
            2.有标点："今天天气真好，我们出去玩吧！你吃饭了吗？"
        '''
        # 检查文件是否存在于 FunASR 目录下
        model_exists = os.path.exists(model_path)   # 如果本地模型存在就用本地模型
        vad_model_exists = os.path.exists(vad_model_path)
        punc_model_exists = os.path.exists(punc_model_path)
        # Modelscope AutoDownload
        self.model = AutoModel(    # 如果模型不存在，就自动从ModelScope下载默认模型
            model=model_path if model_exists else "paraformer-zh",
            vad_model=vad_model_path if vad_model_exists else "fsmn-vad",
            punc_model=punc_model_path if punc_model_exists else "ct-punc-c",
        )
        # 自定义路径
        # self.model = AutoModel(model="FunASR/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", # model_revision="v2.0.4",
        #         vad_model="FunASR/speech_fsmn_vad_zh-cn-16k-common-pytorch", # vad_model_revision="v2.0.4",
        #         punc_model="FunASR/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", # punc_model_revision="v2.0.4",
        #         # spk_model="cam++", spk_model_revision="v2.0.2",
        #         )
    @calculate_time   # 装饰器自动记录识别耗时
    def transcribe(self, audio_file):
        res = self.model.generate(input=audio_file,    # 输入音频文件
            batch_size_s=300)    # 批次处理的音频时长（秒），这里设置为300秒
        print(res)
        return res[0]['text']   # 返回值是列表，包含识别结果字典，提取纯文本返回
    
        
if __name__ == "__main__":
    import os
    # 创建ASR对象并进行语音识别
    audio_file = "output.wav"  # 音频文件路径
    if not os.path.exists(audio_file):  # 不存在则使用edge-tts工具生成一个说"hello"的音频
        os.system('edge-tts --text "hello" --write-media output.wav')
    asr = FunASR()
    print(asr.transcribe(audio_file))

'''
核心设计特点
1.双重降级策略：
    本地模型缺失 → 自动下载默认模型
    支持离线使用

2.模块化设计：
    将ASR封装为独立的类
    可轻松集成到其他项目中

3.性能监控：
    通过装饰器自动计时

4.完整流程：
    VAD（检测语音段）+ ASR（识别）+ 标点恢复
'''