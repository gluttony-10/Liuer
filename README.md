# 六耳 Liuer
善聆音，能察理，知前后，万物皆明。

目前支持的功能有：
1.语音识别
2.字幕制作
3.情感识别
4.说话人识别
5.视频自动下载识别
6.翻译功能

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)
## 更新
250513 新增翻译功能（基于openai类api，可在线可本地）

250503 新增视频自动下载识别功能 更新依赖 修复热词BUG 优化输出

250310 修正情感模型（带时间戳） 添加识别说话人功能（仅热词模型） 自动保存识别文档，增加下载按钮
## 安装
```
git clone https://github.com/gluttony-10/Liuer
cd Liuer
conda create -n Liuer python=3.10
conda activate Liuer
pip install -r requirements.txt
```
## 运行
```
conda activate Liuer
export OpenAI_BASE_URL=API_URL
export OpenAI_API_KEY=API_KEY
export MODEL_NAME=MODEL_NAME
python glut.py
```
## 参考项目
https://github.com/modelscope/FunASR

https://github.com/FunAudioLLM/SenseVoice

https://github.com/openai/whisper

https://github.com/yt-dlp/yt-dlp
