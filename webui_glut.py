# coding=utf-8

import os
import gradio as gr
import numpy as np
import torch
import torchaudio
from funasr import AutoModel
import gc

class FunASRApp:
    def __init__(self):
        self.model = None


    def load_model(
            self, 
            model, 
            vad_model="fsmn-vad", 
            vad_kwargs={"max_single_segment_time": 30000}, 
            punc_model="ct-punc",  
            spk_model="cam++",
            device="cuda:0",
            disable_update=True,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if model == "空载模型":
            self.model = None
            print("模型已卸载")
            return "模型已卸载", gr.update(interactive=True)
        elif model == "情感模型":
            punc_model = None
            spk_model = None
        
        print(f"开始加载{model}")
        model_abbr = {"热词模型": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "情感模型": "iic/SenseVoiceSmall"}        
        model = model_abbr[model]
        self.model = AutoModel(
            model=model,  
            vad_model=vad_model, 
            vad_kwargs=vad_kwargs,
            punc_model=punc_model, 
            spk_model=spk_model,
            device=device,
            disable_update=disable_update,
        )
        print(f"加载{model}成功")
        return "模型加载完成", gr.update(interactive=True)


    def model_inference(
            self, 
            model,
            input, 
            language, 
            hotword='glut',
            use_itn=True,
            batch_size_s=60, 
            merge_vad=True,
            merge_length_s=15,
            sentence_timestamp=True,
            fs=16000
    ):
        if self.model is None:
            return "请先选择并加载模型"
        if model == "情感模型":
            sentence_timestamp = False
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        language_abbr = {"自动": "auto", "中文": "zh", "英文": "en", "粤语": "yue", "日文": "ja", "韩文": "ko", "无语言": "nospeech"}
        language = "自动" if len(language) < 1 else language
        language = language_abbr[language]

        if isinstance(input, tuple):
            fs, input = input
            input = input.astype(np.float32) / np.iinfo(np.int16).max
            if len(input.shape) > 1:
                input = input.mean(-1)
            if fs != 16000:
                print(f"audio_fs: {fs}")
                resampler = torchaudio.transforms.Resample(fs, 16000)
                input_t = torch.from_numpy(input).to(torch.float32)
                input = resampler(input_t[None, :])[0, :].numpy()
        
        res = self.model.generate(
            input=input,
            cache={},
            language=language,
            use_itn=use_itn,
            batch_size_s=batch_size_s, 
            merge_vad=merge_vad,
            merge_length_s=merge_length_s,
            sentence_timestamp=sentence_timestamp,
            hotword=hotword,
        )
        print(res)
        text = res[0]["text"]
        return res, text


    def save_txt(self, text):
        save_path = os.path.join("outputs", "text_files", "output.txt")
        with open(save_path, "w") as f:
            f.write(text)
        return "Text saved to " + save_path


    def launch(self):
        with gr.Blocks(theme=gr.themes.Soft(), fill_height=True) as demo:
            gr.HTML(html_content)
            with gr.Row():
                with gr.Column():
                    audio_inputs = gr.Audio(label="上传音频或使用麦克风", format="wav", editable=True)
                    with gr.Accordion(label="配置"):
                        model_inputs = gr.Dropdown(label="模型", choices=["热词模型", "情感模型", "空载模型"], value="空载模型")
                        status_text = gr.Textbox(label="模型状态", value="模型未加载", interactive=False, visible=False)
                        language_inputs = gr.Dropdown(label="语言", choices=["自动", "中文", "英文", "粤语", "日文", "韩文", "无语言"], value="自动")
                        hotword_inputs = gr.Textbox(label="热词", value="热词 用空格 隔开 十字鱼")
                    fn_button = gr.Button("开始", variant="primary")
                with gr.Column():
                    res_outputs = gr.Textbox(label="结果", visible=False)
                    text_outputs = gr.Textbox(label="结果")
            model_inputs.change(self.load_model, inputs=model_inputs, outputs=[status_text, model_inputs])
            fn_button.click(self.model_inference, inputs=[model_inputs, audio_inputs, language_inputs, hotword_inputs], outputs=[res_outputs, text_outputs])

        demo.launch(inbrowser=True, share=False)

html_content = """
<div>
    <h2 style="font-size: 22px;text-align: center;">FunASR应用程序 FunASR-webui</h2>
</div>
<div style="text-align: center; font-size: 15px; font-weight: bold; color: red;">
    ⚠️ 该演示仅供学术研究和体验使用。
</div>
<div style="text-align: center;">
    制作 by 十字鱼|
    <a href="https://space.bilibili.com/893892">🌐 Bilibili</a> 
</div>
"""

if __name__ == "__main__":
    app = FunASRApp()
    app.launch()