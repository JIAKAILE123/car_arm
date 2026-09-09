 git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release

https://modelscope.cn/models/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/files
https://modelscope.cn/models/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/master/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
# llama.cpp
export PATH=~/llama.cpp/build/bin/:$PATH
 
llama-cli -m qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -n 512 -c 2048 -sys "你是编程小助手" -co -cnv --threads 8
 
 
先启动服务端
llama-server -m ~/llama.cpp/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -c 2048 --threads 8 --port 8081
对话程序：
cd ~/dev_ws/src/tools_demo/tools_demo
python3 chat_interactive.py
 
接入ros
ros2 run tools_demo chat_ros
 
聊天返回的结果会发布到/llama_command,你可以自定义提示词

 