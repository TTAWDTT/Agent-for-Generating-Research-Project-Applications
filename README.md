**该agent用于根据用户的输入，将其转化为标准格式的提纲（该功能开发中），然后根据提纲生成提示词，进而生成科研项目申请书**
*目前的文件为使用训练版，无法接收输入，如果需要修改生成，可以在user_input中修改，在确立config.json的模板后会马上开发交互功能*
<img width="623" height="912" alt="image" src="https://github.com/user-attachments/assets/3e5d5582-8f43-44b6-a5be-eb9064730b50" />
## 部署步骤
### 环境要求
须配置python3.11或借助conda虚拟环境，目前采用Ollama本地部署的qwen3模型（部署过程在此不做说明），下面讲述借助conda的部署步骤
### 配置步骤
1.使用conda创建虚拟环境

    conda create -n my_env python=3.11
2.激活虚拟环境

    conda activate my_env
3.安装所需的库

    pip install -r requirements.txt
4.运行程序

    python main.py
## 运行效果
运行成功后会在终端逐渐打印各步骤的完成情况，随后进行打分以及强化学习。一共三轮，最后输出完整的科研项目申请书。  
全过程大概需要10~12分钟，可通过langsmith的追踪功能检测运行（须自己配置）。  
## 后续开发、优化纲要
### config文件与格式化提取大模型
目前版本只支持用户规范输入前提下的运行，后续会搭载工作节点用于支持用户的不规范输入，并将其转化为config文件中的内容   
### 反馈机制与强化学习 
### 人机交互开发
尽量支持与用户进行频繁的交互，从而更好地切合用户的需求
