import os
from modelscope.pipelines import pipeline
from pydub import AudioSegment
from modelscope.utils.constant import Tasks

current_file_path = os.path.abspath(__file__)
path = os.path.dirname(current_file_path)

#对接收到的音频文件进行处理并转文字
#函数返回结果
def cl():
    audio = AudioSegment.from_file(f"{path}/temp/blob.webm", format="webm")

    # Export the audio file in WAV format
    audio.export(f"{path}/temp/voice.wav", format="wav")

    audio_file=f"{path}/temp/voice.wav"
    #arr, q = soundfile.read(audio_file,dtype="int16")
    #print(q)
    #IPython.display.Audio(audio_file)
    #p = pipeline('auto-speech-recognition', 'damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch')
    p = pipeline(task=Tasks.auto_speech_recognition,
   model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1')
    #res=p(audio_file)
    #print(res)
    data=p(audio_file)
    result = {'text': item['text'] for item in data}
    return result

#data=pipeline(task=Tasks.auto_speech_recognition,
 #  model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1')("voice.wav")
#result = {'text': item['text'] for item in data}

#print(result)
#print(pipeline('auto-speech-recognition', 'damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch')("voice.wav"))
