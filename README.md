# 六耳 Liuer
善聆音，能察理，知前后，万物皆明。

目前支持的功能有：
1.语音识别
2.字幕制作
3.情感识别
4.说话人识别
5.视频自动下载识别

待支持功能：
1.翻译功能

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)
## 更新
250503 新增视频自动下载识别功能 更新依赖 修复热词BUG 优化输出
250310 修正情感模型（带时间戳） 添加识别说话人功能（仅热词模型） 自动保存识别文档，增加下载按钮
## 安装
```
git clone https://github.com/gluttony-10/Liuer
cd Liuer
pip install -r requirements.txt
```
## 运行
```
python glut.py
```
## 参考项目
https://github.com/modelscope/FunASR

https://github.com/FunAudioLLM/SenseVoice

https://github.com/openai/whisper

https://github.com/yt-dlp/yt-dlp
