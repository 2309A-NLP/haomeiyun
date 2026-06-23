# 设备运行端口 (Device running port)
port = 6006
# api运行端口及IP (API running port and IP)
# 模型太大、想多个服务共享同一个模型,LLM 模型在另一个独立程序里跑，WebUI 通过网络请求去问它答案
mode = 'api' # api 需要先运行Linly-api-fast.py，暂时仅仅适用于Linly
ip = '127.0.0.1' 
api_port = 7871

# L模型路径 (Linly model path) 已不用了
# 单机使用，简单直接,LLM 模型直接加载到 WebUI 程序里，自己跑自己的
mode = 'offline'
model_path = 'Qwen/Qwen-1_8B-Chat'

'''
特性	         说明
参数量小	     1.8B（18亿参数），显存占用低
中文优化       阿里训练，中文对话效果好
本地可跑	     8GB 显存甚至 CPU 都能跑
开源免费	     无需 API 密钥，完全离线
'''

# ssl证书 (SSL certificate) 麦克风对话需要此参数
# 最好调整为绝对路径
ssl_certfile = "./deploy/https_cert/cert.pem"
ssl_keyfile = "./deploy/https_cert/key.pem"

# 浏览器的安全策略规定，网页如果要调用麦克风，，必须通过 HTTPS 或 localhost 访问。
"""
如果你通过 http://0.0.0.0:6006 从另一台电脑访问，或者某些浏览器环境下，麦克风按钮会失效。开启 SSL 后，Gradio 会以 https:// 启动，浏览器才会允许录音。
这两个文件是什么？
      cert.pem：SSL 证书（公钥）
      key.pem：SSL 私钥
"""
