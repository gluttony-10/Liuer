# coding=utf-8

import os
import gradio as gr
import torch
from funasr import AutoModel
from funasr_onnx import SeacoParaformer, CT_Transformer
import gc

os.environ["MODELSCOPE_CACHE"] = ".glut/.modelscope"

class FunASRApp:
    def __init__(self):
        self.model = None
        self.hotwords = self._load_hotwords()


    def _load_hotwords(self):
        try:
            with open("hotwords.txt", "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "热词 用空格 隔开 十字鱼"


    def load_model(
            self, 
            model, 
            vad_model="fsmn-vad", 
            vad_kwargs={"max_single_segment_time": 30000}, 
            punc_model="ct-punc",  
            spk_model="cam++",
            device="cuda",
            disable_update=True,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if "选择" in model:
            self.model = None
            print(f'\033[31m请先选择加载模型\033[0m')
            return "请先选择加载模型", gr.update(interactive=True)
        elif "情感" in model:
            punc_model = None
            spk_model = None
            model_name = "iic/SenseVoiceSmall"
            model = "情感模型"
        elif model == "热词模型":
            model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        elif model == "热词模型onnx版":
            model_name = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            model_punc = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
            self.model = SeacoParaformer(model_name, batch_size=1, quantize=True)
            self.model2 = CT_Transformer(model_punc, batch_size=1, quantize=True)
            print(f'\033[32m{model}加载成功\033[0m')
            return f"{model}加载完成", gr.update(interactive=True)
        
        print(f'\033[32m开始加载{model}\033[0m')
        self.model = AutoModel(
            model=model_name,  
            vad_model=vad_model, 
            vad_kwargs=vad_kwargs,
            punc_model=punc_model, 
            spk_model=spk_model,
            device=device,
            disable_update=disable_update,
        )
        print(f'\033[32m{model}加载成功\033[0m')
        return f"{model}加载完成", gr.update(interactive=True)
    

    def model_inference(
            self,
            model,
            video_input,
            language,
            hotwords,
            format_selector,
            save_button,
            use_itn=True,
            batch_size_s=60, 
            merge_vad=True,
            merge_length_s=15,
            sentence_timestamp=True,
    ):        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        with open("hotwords.txt", "w") as f:
            f.write(hotwords)

        if "选择" in model:
            print(f'\033[31m请先选择加载模型\033[0m')
            return "请先选择加载模型", "请先选择加载模型"
        elif "情感" in model:
            if format_selector in ["LRC", "SRT"]:
                print(f'\033[31m情感模型仅支持TXT格式\033[0m')
                return "情感模型仅支持TXT格式", "情感模型仅支持TXT格式"
            sentence_timestamp = False
            model = "情感模型"
        elif model == "热词模型":
            model = "热词模型"
        elif model == "热词模型onnx版":
            for input in video_input:
                asr_result = self.model(
                input,
                hotwords=hotwords
                )
                punc_result = self.model2(asr_result[0]['preds'])
                res = self.merge_asr_and_punc(asr_result, punc_result, input)
                status_text, content = self.process_result(res, input, format_selector, save_button, model)
            return status_text, content

        language_abbr = {"自动": "auto", "中文": "zh", "英文": "en", "粤语": "yue", "日文": "ja", "韩文": "ko", "无语言": "nospeech"}
        language = "自动" if len(language) < 1 else language
        language = language_abbr[language]
        
        for input in video_input:
            res = self.model.generate(
                input=input,
                cache={},
                language=language,
                use_itn=use_itn,
                batch_size_s=batch_size_s, 
                merge_vad=merge_vad,
                merge_length_s=merge_length_s,
                sentence_timestamp=sentence_timestamp,
                hotwords=hotwords,
            )
            status_text, content = self.process_result(res, input, format_selector, save_button, model)
        print(res)
        return status_text, content
    

    def merge_asr_and_punc(self, asr_result, punc_result, audio_key):
        # 提取 asr_result 中的信息
        timestamps = asr_result[0]['timestamp']
        
        # 提取 punc_result 中的信息
        original_text = asr_result[0]['preds'].split()
        punctuated_text = punc_result[0]
        
        # 根据标点符号分割文本和时间戳
        sentences = []
        sentence_timestamps = []
        start_index = 0
        n = 1
        
        # 定义标点符号
        punctuation_marks = "，。！？；："
        
        for i, char in enumerate(punctuated_text):
            if char in punctuation_marks:
                end_index = i
                sentence = original_text[start_index:end_index-n+1] + [char]
                print(sentence)
                sentence_timestamp = timestamps[start_index:(end_index-n+1)]
                sentences.append(sentence)
                sentence_timestamps.append(sentence_timestamp)
                start_index = end_index-n+1
                n += 1
        
        # 处理最后一个句子（如果没有以标点符号结尾）
        if start_index < len(punctuated_text):
            sentence = punctuated_text[start_index:]
            sentence_timestamp = timestamps[start_index:]
            sentences.append(sentence)
            sentence_timestamps.append(sentence_timestamp)
        
        # 构建最终的输出格式
        sentence_info = []
        
        for sentence, timestamp in zip(sentences, sentence_timestamps):
            if timestamp:  # 确保 timestamp 不为空
                start_time = timestamp[0][0]
                end_time = timestamp[-1][1]
                sentence_info.append({
                    'text': sentence,
                    'start': start_time,
                    'end': end_time,
                    'timestamp': timestamp,
                    'spk': 0
                })
        
        output = [{
            'key': audio_key,
            'text': punctuated_text,
            'timestamp': [ts for sublist in sentence_timestamps for ts in sublist],
            'sentence_info': sentence_info
        }]
        
        return output
    

    def process_result(
            self, 
            res, 
            input, 
            format_selector, 
            save_button, model
    ):
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(input)
        filename_without_extension, extension = os.path.splitext(filename)
        
        # 根据格式生成内容
        base_path = os.path.join(output_dir, filename_without_extension)
        if format_selector == "LRC":
            content = self._generate_lrc(res)
            if save_button:
                with open(f"{base_path}.lrc", "w") as f:
                    f.write(content)
        elif format_selector == "SRT":
            content = self._generate_srt(res)
            if save_button:
                with open(f"{base_path}.srt", "w") as f:
                    f.write(content)
        else:
            content = res[0]["text"]
            if format_selector:
                with open(f"{base_path}.txt", "w") as f:
                    f.write(content)

        status_text = f"{model}识别成功"
        if save_button:
            status_text += f"，{format_selector}文件已保存至{output_dir}"
        else:
            status_text += "，未选择保存文件"

        return status_text, content


    def _generate_lrc(self, res):
        """生成标准LRC歌词格式"""
        lrc_lines = []
        for segment in res[0]["sentence_info"]:
            start = self._format_lrc_time(segment.get("start", 0.0))
            text = segment.get("text", "")
            # 添加类型转换
            if isinstance(text, list):
                text = "".join(text)
            text = text.strip()
            # 简化为只显示开始时间
            lrc_lines.append(f"[{start}]{text}") 
        return "\n".join(lrc_lines)
    

    def _format_lrc_time(self, seconds):
        """LRC时间格式转换（分:秒.厘秒）"""
        seconds /= 1000
        total_seconds = round(seconds, 2)  # 精确到厘秒
        mins = int(total_seconds // 60)
        secs = total_seconds % 60
        return f"{mins:02d}:{secs:05.2f}"  # 示例：37.28秒 → 00:37.28


    def _generate_srt(self, res):
        """生成标准SRT字幕格式（带序号和时间范围）"""
        srt_lines = []
        for index, segment in enumerate(res[0]["sentence_info"], 1):
            # 时间格式转换（新增带小时的三段式格式）
            start = self._format_srt_time(segment.get("start", 0.0))
            end = self._format_srt_time(segment.get("end", 0.0))
            text = segment.get("text", "")
            # 添加类型转换
            if isinstance(text, list):
                text = "".join(text)
            text = text.strip()
            
            # 构建字幕块
            srt_lines.append(f"{index}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"{text}\n")
        
        return "\n".join(srt_lines)
    

    def _format_srt_time(self, seconds):
        seconds /= 1000
        """SRT专用时间格式转换（小时:分钟:秒,毫秒）"""
        hours = int(seconds // 3600)
        remainder = seconds % 3600
        mins = int(remainder // 60)
        secs = remainder % 60
        return f"{hours:02d}:{mins:02d}:{secs:06.3f}".replace('.', ',')


    def launch(self):
        with gr.Blocks(theme=gr.themes.Soft(), fill_height=True) as demo:
            gr.HTML(html_content)
            with gr.Row():
                with gr.Column():
                    video_input = gr.File(
                        label="上传音视频文件（可多选）", 
                        file_count="multiple",
                        file_types=["video","audio"],
                        type="filepath"
                    )                           
                    with gr.Accordion(label="配置"):
                        model_inputs = gr.Dropdown(
                            label="模型", 
                            choices=["热词模型", "热词模型onnx版", "情感模型", "请先选择加载模型"], 
                            value="请先选择加载模型"
                        )
                        language_inputs = gr.Dropdown(
                            label="语言", 
                            choices=["自动", "中文", "英文", "粤语", "日文", "韩文", "无语言"], 
                            value="自动"
                        )
                        format_selector = gr.Dropdown(
                            label="格式",
                            choices=["TXT", "LRC", "SRT"],
                            value="TXT"
                        )
                        save_button = gr.Checkbox(label="保存文件", value=False)
                        hotwords_inputs = gr.Textbox(label="热词", value=self.hotwords)
                    fn_button = gr.Button("开始", variant="primary")
                with gr.Column():
                    status_text = gr.Textbox(label="状态", value="请先选择加载模型", interactive=False)
                    text_outputs = gr.Textbox(label="输出")
            model_inputs.change(
                self.load_model, 
                inputs=model_inputs, 
                outputs=[status_text, model_inputs]
            )
            fn_button.click(
                self.model_inference, 
                inputs=[
                    model_inputs,
                    video_input,
                    language_inputs, 
                    hotwords_inputs,
                    format_selector,
                    save_button
                ],
                outputs=[status_text, text_outputs],
                queue=True,
                show_progress=True
            )
        demo.launch(inbrowser=True, share=False, server_name="127.0.0.1")

html_content = """
<div>
    <h2 style="font-size: 22px;text-align: center;">语音识别 字幕制作</h2>
</div>
<div style="text-align: center; font-size: 15px; font-weight: bold; color: red;">
    ⚠️ 该演示仅供学术研究和体验使用。
</div>
<div style="text-align: center;">
    制作 by 十字鱼|
    <a href="https://space.bilibili.com/893892">🌐 bilibili</a> 
</div>
"""

if __name__ == "__main__":
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print("开源项目：https://github.com/gluttony-10/FunASR-webui bilibili@十字鱼 https://space.bilibili.com/893892 ")
    print(f'\033[32mCUDA版本：{torch.version.cuda}\033[0m')
    print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，使用FP16\033[0m')
        dtype = torch.float16
    app = FunASRApp()
    app.launch()