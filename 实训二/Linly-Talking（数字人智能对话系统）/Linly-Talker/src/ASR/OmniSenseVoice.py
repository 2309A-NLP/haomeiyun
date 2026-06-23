# 阿里达摩院的SenseVoice模型
from omnisense.models import OmniSenseVoiceSmall, OmniTranscription  # 转录结果的数据结构
from lhotse.utils import Pathlike   # 音频处理工具库
import torch

class OmniSenseVoice:
    def __init__(self, quantize) -> None:
        self.model = OmniSenseVoiceSmall("iic/SenseVoiceSmall",    # 模型名称（ModelScope上的ID）
                                         quantize=quantize,        # 是否量化（压缩模型）
                                         device_id = "0" if torch.cuda.is_available() else "-1")   # GPU/CPU选择
    '''
    参数说明：
        模型选择：iic/SenseVoiceSmall（SenseVoice轻量版）
        量化：quantize参数控制是否进行模型量化，可减少模型大小和加速推理
        设备：自动检测CUDA，如果有GPU使用设备0，否则使用CPU（-1）
    '''

    def transcribe(
        self,
        audio_path: Pathlike,    # 音频文件路径
        language: str = 'auto',  # 语音：auto自动检测，或者指定'zh','en'等
        textnorm : str = 'woitn', # 文本规范化：withitn（带逆文本正则化）或woitn（不带）
        timestamps : bool = False   # 是否返回时间戳
    ):
        result = self.model.transcribe(audio_path, language=language, textnorm=textnorm,
                                    batch_size=8,   # 批处理大小
                                    timestamps=timestamps)
        return result[0].text    # 返回第一个（也是唯一一个）结果的文本

if __name__ == '__main__':
    model = OmniSenseVoice(quantize=False)    # 不进行量化
    text = model.transcribe('./audio.wav')
    print(text)

'''
withitn（逆文本正则化）：将数字、日期等转为文字形式
    例如："123" → "一百二十三"
    "2024-01-01" → "二零二四年一月一日"

woitn（无逆文本正则化）：保留原始识别结果
    例如："123" → "123"
    "2024-01-01" → "2024 01 01"
'''