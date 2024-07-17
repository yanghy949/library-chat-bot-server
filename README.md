# 校园图书馆聊天机器人-server

- ### 本项目使用的模型为[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)
- ### 项目仅后端部署，配合前端项目实现聊天机器人功能 [前端项目链接](https://github.com/yanghy949/library-chat-bot-web)
- ### 此项目及其[前端项目](https://github.com/yanghy949/library-chat-bot-web)为学校图书馆项目，仅供学习交流使用

### 功能介绍

- ### 实现了基础的人机对话
- ### 实现了基于知识库的检索问答
- ### 实现了语音/文本转换
- ### 实现了知识库的更新

### 部署方法(server)

- #### 下载语言模型

       请自行下载语言模型到项目目录(此处举例目录为/chatglm-6b)
       配置api.py中的模型路径，如：model_path = "/chatglm-6b"

- #### 部署
   ```shell
     # 安装依赖
     pip install -r env.txt
     # 启动服务
     python api.py
     ```


- ### 完整项目(Docker)

  ```shell
      docker pull registry.cn-chengdu.aliyuncs.com/yanghy949/chatbot:1
    ```