import requests
from std_msgs.msg import String
import rclpy
from rclpy.node import Node
import json


class LlamaChat(Node):
    """LlamaChat 类,
    需要启动llama-server -m /home/sunrise/llama.cpp/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -c 2048 --threads 8 --port 8081
    """

    def __init__(self, base_url="http://localhost:8081"):
        super().__init__("LlamaChat")
        self.base_url = base_url
        self.chat_history = []
        self.system_prompt = '你是编程小助手。'

        # 创建话题发布者（替换服务客户端）
        self.command_publisher = self.create_publisher(String, "llama_command", 10)

        print("=== Llama 交互式聊天程序 ===")
        print("输入 'quit' 或 'exit' 退出程序")
        print("每次回复后可选择处理方式:")
        print("  N  - 不处理")
        print("  R  - 发布到话题")
        print("-" * 40)
        while True:
            try:
                # 获取用户输入
                user_input = input("\n> ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    print("再见！")
                    break

                if not user_input:
                    continue

                # 发送消息并获取回复
                print("正在思考...")
                response = self.send_message(user_input)
                print(f"\n助手: {response}")  # 输出助手回复

                # 询问处理选项
                while True:
                    option = input("\n选择处理方式 [N/R]: ").strip()
                    if option.upper() in ["N", "R"]:
                        self.process_response(response, option)
                        break
                    else:
                        print("请输入有效选项: N, R")

            except KeyboardInterrupt:
                print("\n\n程序被中断,再见！")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")

    def send_message(self, user_input):
        # 构建消息历史
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": user_input})

        data = {
            "model": "qwen2.5-coder",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.7,
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                assistant_reply = result["choices"][0]["message"]["content"]

                # 更新对话历史
                self.chat_history.append({"role": "user", "content": user_input})
                self.chat_history.append(
                    {"role": "assistant", "content": assistant_reply}
                )

                return assistant_reply
            else:
                return f"错误: {response.status_code}, {response.text}"

        except Exception as e:
            return f"连接错误: {str(e)}"

    def process_response(self, response, option):
        """处理回复的不同选项"""
        if option.upper() == "N":
            print("不处理，继续对话...")
        elif option.upper() == "R":
            print("发布到话题...")
            # 创建String消息并发布
            msg = String()
            msg.data = response
            self.command_publisher.publish(msg)
            print(f"已发布消息: {response}")


def main(args=None):
    rclpy.init(args=args)
    chat = LlamaChat()
    try:
        rclpy.spin(chat)  # 保持节点运行
    except KeyboardInterrupt:
        pass
    finally:
        chat.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
