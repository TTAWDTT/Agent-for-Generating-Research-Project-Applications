from typing import TypedDict, Annotated, Sequence, List, Dict, Tuple
import re
import numpy as np
from scipy.special import expit
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
# 新增LangSmith追踪相关导入
from langchain_core.callbacks import CallbackManager
from langchain_core.tracers.langchain import LangChainTracer

# 加载环境变量
load_dotenv(override=True)

# 初始化回调管理器（用于LangSmith追踪）
callback_manager = CallbackManager([LangChainTracer()])

# 初始化大语言模型
llm = ChatOllama(model="mixtral", temperature=0.5)

# ===== PPO相关工具函数 =====
class PPOPolicy:
    """PPO策略类，用于优化生成参数"""
    def __init__(self, input_size: int = 4, hidden_size: int = 16, lr: float = 0.001):
        self.weights = np.random.randn(input_size, hidden_size) * 0.1
        self.bias = np.zeros(hidden_size)
        self.output_weights = np.random.randn(hidden_size, 1) * 0.1
        self.output_bias = np.zeros(1)
        self.optimizer = lr  # 简化的优化器
    
    def forward(self, state_features: np.ndarray) -> float:
        """根据状态特征预测最佳temperature参数"""
        hidden = np.tanh(np.dot(state_features, self.weights) + self.bias)
        output = expit(np.dot(hidden, self.output_weights) + self.output_bias)
        return 0.1 + output[0] * 0.9  # 将输出映射到0.1-1.0范围的temperature值
    
    def update(self, advantages: np.ndarray, old_log_probs: np.ndarray, 
               new_log_probs: np.ndarray, states: List[np.ndarray], clip_epsilon: float = 0.2):
        """PPO更新步骤"""
        for state, advantage, old_log_prob, new_log_prob in zip(states, advantages, old_log_probs, new_log_probs):
            ratio = np.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            loss = -np.min([surr1, surr2])  # 负号因为我们要最大化奖励
            
            # 简化的梯度下降更新
            grad = self._compute_gradient(state, loss)
            self.weights -= self.optimizer * grad[0]
            self.bias -= self.optimizer * grad[1]
            self.output_weights -= self.optimizer * grad[2]
            self.output_bias -= self.optimizer * grad[3]
    
    def _compute_gradient(self, state: np.ndarray, loss: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """简化的梯度计算"""
        return (
            np.random.randn(*self.weights.shape) * loss * 0.01,
            np.random.randn(*self.bias.shape) * loss * 0.01,
            np.random.randn(*self.output_weights.shape) * loss * 0.01,
            np.random.randn(*self.output_bias.shape) * loss * 0.01
        )

# 初始化PPO策略
ppo_policy = PPOPolicy()

# ===== 类型定义 =====
class ResearchProposalState(TypedDict):
    user_outline: Annotated[str, "用户输入的提纲（含引文）"]
    extracted_citations: Annotated[List[str], "从提纲中提取的引文列表"]
    requirements: Annotated[str, "申报指南解析结果"]
    sections: Annotated[dict, "申报书各部分内容"]
    draft: Annotated[str, "完整申报书草稿"]
    feedback: Annotated[Sequence[str], "审核反馈记录"]
    final_output: Annotated[str, "最终申报书"]
    scores: Annotated[Dict[str, float], "评分指标"]
    reward: Annotated[float, "PPO奖励值"]
    temperature: Annotated[float, "当前生成温度参数"]
    episode: Annotated[int, "训练轮次"]
    total_steps: Annotated[int, "全局迭代步数（防无限循环）"]

# ===== 评分机制 =====
def score_proposal(state: ResearchProposalState) -> Dict[str, float]:
    """评估申报书质量的评分函数"""
    scores = {
        "citation_completeness": 0.0,
        "citation_relevance": 0.0,
        "structure_quality": 0.0,
        "content_relevance": 0.0
    }
    
    # 1. 引文完整性评分（0-10分）
    total_citations = len(state["extracted_citations"])
    if total_citations > 0:
        missing = [c for c in state["extracted_citations"] if c not in state["draft"]]
        scores["citation_completeness"] = 10.0 * (1 - len(missing)/total_citations)
    
    # 2. 引文相关性评分
    if state["feedback"]:
        last_feedback = state["feedback"][-1]
        if "引文合理" in last_feedback:
            scores["citation_relevance"] = 8.0 + (2.0 if "高度相关" in last_feedback else 0)
        elif "引文不相关" in last_feedback:
            scores["citation_relevance"] = 3.0
        else:
            scores["citation_relevance"] = 5.0
    
    # 3. 结构完整性评分
    required_sections = ["项目名称", "研究背景", "研究内容", "研究目标", 
                        "技术路线", "创新点", "预期成果", "研究基础"]
    completed_sections = sum(1 for sec in required_sections if sec in state["sections"])
    scores["structure_quality"] = 10.0 * (completed_sections / len(required_sections))
    
    # 4. 内容相关性评分
    if state["user_outline"] and state["draft"]:
        outline_keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", state["user_outline"])
        draft_keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", state["draft"])
        overlap = len(set(outline_keywords) & set(draft_keywords)) / len(set(outline_keywords)) if outline_keywords else 0
        scores["content_relevance"] = 10.0 * overlap
    
    return scores

def calculate_reward(scores: Dict[str, float]) -> float:
    """基于评分计算PPO奖励"""
    weights = {
        "citation_completeness": 0.3,
        "citation_relevance": 0.3,
        "structure_quality": 0.2,
        "content_relevance": 0.2
    }
    return sum(scores[key] * weights[key] for key in scores)

# ===== 辅助函数：从提纲提取引文 =====
def extract_citations(outline: str) -> List[str]:
    # 1. 先定位到"引文如下："之后的内容，避免前面文本干扰
    citations_start = outline.find("引文如下：")
    if citations_start == -1:
        print("未找到'引文如下：'标记，无法提取引文")
        return []
    citations_text = outline[citations_start + len("引文如下："):]
    
    # 2. 优化正则：匹配每行开头的序号+标题+作者+要点
    # 增加re.MULTILINE使^匹配每行开头，使用非贪婪匹配处理标题
    pattern = r'^\d+\.\s+(.+?)\n作者：(.+?)\n要点：(.+?)(?=\n\d+\.|$)'
    citations = re.findall(
        pattern, 
        citations_text, 
        re.DOTALL | re.MULTILINE  # 关键：添加多行模式
    )
    
    # 增强调试信息
    print(f"提取到{len(citations)}条引文")
    for i, cit in enumerate(citations, 1):
        print(f"引文{i}：标题={cit[0][:30]}..., 作者={cit[1][:20]}...")
    
    # 清理并去重（合并为字符串便于后续检查）
    cleaned = [f"{num}. {title}\n作者：{author}\n要点：{points}" 
              for num, (title, author, points) in enumerate(citations, 1)]
    return list(set(cleaned))  # 去重

# ===== Agent定义 =====
def requirements_analyzer(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始分析需求...")
    # 提取提纲中的引文并存储
    citations = extract_citations(state["user_outline"])
    print(f"[需求分析] 提取到引文：{citations}")
    
    # 分析申报要求
    prompt = ChatPromptTemplate.from_messages([
        ("system", "您是科研项目申报专家，分析项目类型和要求，注意提纲中已有引文：{citations}"),
        ("human", "用户提纲：{user_outline}\n请总结该类项目的核心要求（格式、内容、常见错误）")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "user_outline": state["user_outline"],
        "citations": citations
    })
    return {
        "requirements": result.content,
        "extracted_citations": citations,
        "total_steps": state["total_steps"] + 1
    }

def content_generator(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始生成内容...")
    sections_to_generate = [
        "项目名称", "研究背景", "研究内容", "研究目标",
        "技术路线", "创新点", "预期成果", "研究基础"
    ]
    
    # 从PPO策略获取当前temperature参数
    state_features = np.array([
        len(state["extracted_citations"]),
        len(state["sections"]),
        len(state["feedback"]),
        state["temperature"] if "temperature" in state else 0.5
    ])
    current_temp = ppo_policy.forward(state_features)
    print(f"[内容生成] PPO调整温度参数为: {current_temp:.2f}")
    
    for section in sections_to_generate:
        if section not in state["sections"]:
            # 使用PPO调整后的temperature
            llm.temperature = current_temp
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""您是科研申报专家，撰写'{section}'部分。必须严格使用提纲中的所有引文：{state['extracted_citations']}，
                不得遗漏，引用时保留原格式（如Smith et al.(2023)），并解释引文与内容的关联。"""),
                ("human", "根据提纲生成内容，确保每个引文都被用到：\n{user_outline}")
            ])
            chain = prompt | llm
            result = chain.invoke({
                "section": section,
                "requirements": state["requirements"],
                "user_outline": state["user_outline"]
            })
            state["sections"][section] = result.content
            print(f"[内容生成] 完成'{section}'部分")
    
    return {
        "sections": state["sections"],
        "temperature": current_temp,
        "total_steps": state["total_steps"] + 1
    }

def proposal_integrator(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始整合提案...")
    # 标准化申报书结构
    draft = "# 科研项目申报书\n\n"
    
    # 1. 项目基本信息表
    draft += "## 一、项目基本信息\n"
    title_match = re.search(r"项目名称：(.*?)\n", state["user_outline"])
    draft += "| 项目名称 | {user_outline_title} |\n".format(
        user_outline_title=title_match.group(1) if title_match else ""
    )
    draft += "|----------|----------------------|\n"
    draft += "| 申报单位 | （请填写）           |\n"
    draft += "| 项目负责人 | （请填写）          |\n"
    draft += "| 研究周期 | （请填写）           |\n"
    draft += "| 申报日期 | （请填写）           |\n\n"
    
    # 2. 立项依据
    draft += "## 二、立项依据\n"
    draft += "### （一）研究背景与意义\n"
    if "研究背景" in state["sections"]:
        draft += state["sections"]["研究背景"] + "\n\n"
    
    draft += "### （二）国内外研究现状\n"
    draft += "1. 相关领域研究进展：\n"
    for citation in state["extracted_citations"]:
        draft += f"- {citation}的研究表明..."
    draft += "\n\n2. 现有研究不足：\n"
    draft += "（结合上述文献分析当前研究空白）\n\n"
    
    # 3. 研究内容与目标
    draft += "## 三、研究内容与目标\n"
    draft += "### （一）主要研究内容\n"
    if "研究内容" in state["sections"]:
        draft += state["sections"]["研究内容"] + "\n\n"
    
    draft += "### （二）研究目标\n"
    if "研究目标" in state["sections"]:
        draft += state["sections"]["研究目标"] + "\n\n"
    
    # 4. 技术路线与研究方法
    draft += "## 四、技术路线与研究方法\n"
    if "技术路线" in state["sections"]:
        draft += state["sections"]["技术路线"] + "\n\n"
    draft += "### （一）研究方法\n"
    draft += "1. 实验设计方案\n"
    draft += "2. 数据采集与分析方法\n"
    draft += "3. 关键技术验证途径\n\n"
    
    # 5. 创新点
    draft += "## 五、项目创新点\n"
    if "创新点" in state["sections"]:
        draft += state["sections"]["创新点"] + "\n\n"
    
    # 6. 研究计划与进度安排
    draft += "## 六、研究计划与进度\n"
    draft += "| 时间阶段 | 主要工作内容 |\n"
    draft += "|----------|--------------|\n"
    draft += "| 第1-3月 | 文献调研与方案设计 |\n"
    draft += "| 第4-9月 | 实验系统开发与数据采集 |\n"
    draft += "| 第10-12月 | 结果分析与报告撰写 |\n\n"
    
    # 7. 预期成果
    draft += "## 七、预期成果与形式\n"
    if "预期成果" in state["sections"]:
        draft += state["sections"]["预期成果"] + "\n\n"
    draft += "成果形式：\n1. 学术论文（预计X篇）\n2. 专利（预计X项）\n3. 系统原型（1套）\n\n"
    
    # 8. 研究基础
    draft += "## 八、研究基础与条件\n"
    if "研究基础" in state["sections"]:
        draft += state["sections"]["研究基础"] + "\n\n"
    draft += "### （一）前期研究积累\n"
    draft += "### （二）实验条件保障\n\n"
    
    # 9. 经费预算
    draft += "## 九、经费预算（单位：元）\n"
    draft += "| 预算科目 | 金额 | 计算依据 |\n"
    draft += "|----------|------|----------|\n"
    draft += "| 文献资料费 | 5000 | 数据库订阅与文献购买 |\n"
    draft += "| 实验材料费 | 20000 | 实验耗材与样本采集 |\n"
    draft += "| 差旅费 | 8000 | 学术交流与数据采集 |\n"
    draft += "| 合计 | 33000 | |\n\n"
    
    # 引文完整性检查补充
    missing = [c for c in state["extracted_citations"] if c not in draft]
    if missing:
        draft += "## 十、引文补充说明\n"
        draft += f"未在正文中规范引用的文献：{', '.join(missing)}\n"
        draft += "补充引用说明：\n"
        for c in missing:
            draft += f"- {c}：[请补充引用场景说明]\n"
    
    print(f"[提案整合] 完成草稿生成，共{len(draft)}字符")
    return {
        "draft": draft,
        "total_steps": state["total_steps"] + 1
    }

def quality_reviewer(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始质量审核...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "审核申报书，重点检查提纲中的引文：{citations}是否全部被正确引用，格式是否保留。"),
        ("human", "申报书草稿：{draft}\n反馈需包含：1. 引文完整性 2. 引用合理性 3. 其他改进建议")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "requirements": state["requirements"],
        "draft": state["draft"],
        "citations": state["extracted_citations"]
    })
    
    # 计算评分和奖励
    new_scores = score_proposal(state)
    new_reward = calculate_reward(new_scores)
    
    feedback = state["feedback"] + [result.content]
    print(f"[质量审核] 完成审核，评分：{new_scores}，奖励：{new_reward:.2f}")
    print(f"[质量审核] 反馈摘要：{result.content[:100]}...")
    return {
        "feedback": feedback,
        "scores": new_scores,
        "reward": new_reward,
        "total_steps": state["total_steps"] + 1
    }

def ppo_learner(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始PPO学习...")
    """PPO学习节点，根据奖励更新策略"""
    if state["episode"] > 0:  # 至少完成一个回合才更新
        # 准备训练数据
        state_features = np.array([
            len(state["extracted_citations"]),
            len(state["sections"]),
            len(state["feedback"]),
            state["temperature"]
        ])
        
        # 计算优势函数
        advantage = state["reward"] - 5.0  # 假设平均奖励为5.0
        
        # 旧策略概率
        old_log_prob = np.log(state["temperature"] / 1.0)
        
        # 新策略概率
        new_temp = ppo_policy.forward(state_features)
        new_log_prob = np.log(new_temp / 1.0)
        
        # 更新PPO策略
        ppo_policy.update(
            advantages=np.array([advantage]),
            old_log_probs=np.array([old_log_prob]),
            new_log_probs=np.array([new_log_prob]),
            states=[state_features]
        )
        print(f"[PPO学习] 策略更新完成，优势值：{advantage:.2f}")
    
    # 准备下一轮
    next_episode = state["episode"] + 1
    print(f"[PPO学习] 完成第{state['episode']}轮，进入第{next_episode}轮")
    return {
        "episode": next_episode,
        "sections": {},  # 重置部分状态用于下一轮训练
        "feedback": [],
        "total_steps": state["total_steps"] + 1
    }

def revision_director(state: ResearchProposalState):
    """审核结果导向器，决定下一步流程"""
    last_feedback = state["feedback"][-1] if state["feedback"] else ""
    
    # 放宽条件：即使有建议修改，超过2轮也强制进入学习阶段
    if len(state["feedback"]) >= 2 and "建议修改" in last_feedback:
        print("[审核导向] 修改次数达2次，强制进入学习阶段")
        return "ppo_learn"  # 直接返回目标节点名称
    
    if "关键缺陷" in last_feedback or "引文遗漏" in last_feedback:
        print("[审核导向] 发现关键缺陷，返回需求分析")
        return "analyze_requirements"
    elif "建议修改" in last_feedback:
        print("[审核导向] 需要 minor 修改，返回内容生成")
        return "generate_content"
    
    print("[审核导向] 审核通过，进入PPO学习")
    return "ppo_learn"  # 直接返回目标节点名称

def check_termination(state: ResearchProposalState):
    """检查是否需要强制终止流程"""
    max_steps = 30  # 最大步骤限制
    if state["total_steps"] >= max_steps:
        print(f"\n[终止检查] 已达到最大步骤({max_steps})，强制终止流程")
        return "force_end"
    return "continue"

# ===== 工作流构建 =====
workflow = StateGraph(ResearchProposalState)

# 添加节点
workflow.add_node("analyze_requirements", requirements_analyzer)
workflow.add_node("generate_content", content_generator)
workflow.add_node("integrate_proposal", proposal_integrator)
workflow.add_node("review_quality", quality_reviewer)
workflow.add_node("ppo_learn", ppo_learner)
workflow.add_node("check_termination", lambda s: {"total_steps": s["total_steps"] + 1})  # 步骤计数节点

# 设置入口
workflow.set_entry_point("analyze_requirements")

# 添加主流程边
workflow.add_edge("analyze_requirements", "generate_content")
workflow.add_edge("generate_content", "integrate_proposal")
workflow.add_edge("integrate_proposal", "review_quality")

# 审核后条件分支 - 修复关键错误点
workflow.add_conditional_edges(
    "review_quality",
    revision_director,  # 该函数直接返回目标节点名称
    {
        "analyze_requirements": "analyze_requirements",
        "generate_content": "generate_content",
        "ppo_learn": "ppo_learn"
    }
)

# PPO学习后循环或结束
workflow.add_conditional_edges(
    "ppo_learn",
    lambda state: "continue" if state["episode"] < 3 else "END",  # 3轮训练
    {"continue": "analyze_requirements", "END": END}
)

# 添加全局步骤检查（防止无限循环）
workflow.add_conditional_edges(
    "check_termination",
    check_termination,
    {"force_end": END, "continue": "integrate_proposal"}
)

# 执行引擎
research_app = workflow.compile()

# ===== 执行示例 =====
if __name__ == "__main__":
    # 用户输入（含明确引文）
    user_outline = """
面向电磁域多模态信息的任务理解及思维推理技术
研究目标:围绕无人化智能化快速发展、AIAgent将用于大幅削减人工成本的趋势，开展面向电磁域多模态信息的任务理解及思维推理技术研究，突破面向多模态信息的多层次决策融合、多AIAgent自演进动态拓扑流等关键技术，面向典型电磁应用任务构建仿真系统并完成验证。
技术需求:
1)支持文本类、图像类、电磁信号类等不少于3种电磁域典型模态作为输入;2)具备长时思维链推理能力，推理Token长度不低于2K;
3)面向“收集可用电磁数据、理解复杂任务需求、融合多模态信息、生成分析报告”为代表的典型电磁应用任务，实现理解任务需求和长时思维推理的正确率不低于90%，单轮对>4个工具的调用准确率不低于90%，多轮工具调用准确率不低于80%。成果形式:技术研究报告2份、模型(含源码)1套、仿真训练与验证系统1套、发表SCIEI检索论文2篇。
研究周期:2年
项目类别:一般项目
拟资助经费:30万
是否对外开放:是
引文如下：
1. 北大团队电磁空间具身智能体 metaAgent
作者：李廉林（北京大学）、崔铁军（东南大学）
要点：多模态融合：构建基于大模型的电磁语义化表征与处理模型，支持电磁信号、文本、图像等多模态信息融合，实现电磁观测信号到自然语言的转化及物理层电磁操控。
长时推理能力：通过多模态大模型（大脑）与语义超表面（小脑）的协同，实现复杂电磁任务的自主规划与动态决策，支持类人推理链的生成与执行。
仿真验证：开发面向智慧家庭、工厂等场景的超材料智能体平台，验证了在动态电磁环境中多模态信息处理和任务执行的稳定性。
技术突破：实现物理域与数字域的一体化设计，解决传统认知雷达响应延迟高、智能水平低的问题，在复杂电磁环境中任务执行准确率显著提升。
2. 基于知识图谱的电磁目标多模态感知方法
作者：X 技术团队（专利）
要点：多模态输入支持：处理雷达探测、射频、光电、文本等多模态数据，通过知识图谱实现多源异构数据的结构化管理与融合。
动态决策机制：提出基于知识图谱的多模态特征提取与冲突消解算法，通过删除不一致探测结果、深度学习模型融合等策略提升目标识别准确率。
任务适应性：支持电磁目标的识别、跟踪与轨迹预测，在典型电磁应用场景中实现目标类别判别和位置预测的高可靠性。
技术优势：减少人工干预，提升多模态数据利用效率，适用于复杂电磁环境下的实时监测与决策。
3. Meta-Transformer 多模态统一框架
作者：香港中文大学联合上海 AI Lab 团队
要点：多模态兼容性：支持 12 种模态（含电磁相关模态如高光谱、IMU），通过统一标记器将原始数据映射到共享 Token 空间，利用冻结参数的 Transformer 编码器提取跨模态语义特征。
长序列处理能力：通过递归计算和隐空间扩展机制处理长序列数据（Token 长度可达数 K），在文本、图像、电磁信号等任务中表现出优异的泛化性。
高准确率验证：在 LAION-2B 数据集上预训练后，在多模态分类、时序预测等任务中超越现有方法，尤其在电磁信号分类和高光谱图像分析中达到行业领先水平。
开源潜力：框架支持模块化扩展，可快速适配电磁域多模态任务，为仿真系统开发提供底层技术支持。
4. 多智能体自主电子干扰系统
作者：浙江大学团队
要点：动态拓扑流：基于信息共享的多智能体协同机制，实现 μs 级干扰参数收敛与动态调整，适应复杂电磁环境中辐射源的快速变化。
长时推理与决策：通过多智能体间的实时态势信息共享，构建分布式决策网络，支持多轮工具调用（如干扰策略选择、参数优化）的高准确率（>80%）。
仿真验证：在电子对抗场景中验证了系统的抗干扰能力，干扰参数更新策略与动态环境变化的匹配度显著提升，任务执行鲁棒性增强。
技术应用：可扩展至认知雷达、频谱管理等领域，解决传统单智能体感知范围有限、决策滞后的问题。
5. 长思维链推理综述（Long CoT）
作者：哈工大团队
要点：推理机制：提出深度推理、广泛探索与可行性反思的三层框架，支持长序列（Token≥2K）的逻辑链生成，解决传统短思维链（Short CoT）推理深度不足的问题。
电磁域适配：通过 “雪球效应” 抑制误差累积，结合强化学习（RL）和过程奖励模型（PRM）优化长链推理路径，在电磁信号分析、频谱预测等任务中实现高稳定性。
技术创新：引入隐空间推理（如 “思维向量”）和多智能体协同校验机制，提升长时推理的可解释性与容错能力，为电磁任务的复杂逻辑分析提供理论支持。
6. 神经符号混合推理引擎
作者：某研究团队（CSDN 博客）
要点：多模态融合架构：通过深度神经网络与符号逻辑的协同，实现 12 种模态的并行处理（延迟 < 3ms），支持电磁信号、文本、图像的动态融合与因果推理。
高准确率验证：在工业物联网、医疗诊断等领域实现 95% 以上的任务准确率，其动态知识图谱更新机制（延迟 < 200ms）可直接迁移至电磁域复杂任务。
可扩展性：支持跨领域知识迁移（如金融→医疗→工业），为电磁仿真系统的多任务适配提供参考。
技术优势：结合量子蒙特卡洛优化算法，在多模态因果推理中提升隐性关联挖掘的准确率，适用于电磁信号异常检测与故障诊断。
7. 开源电磁仿真工具
工具：OpenParEM、openEMS
要点：仿真能力：支持 3D 电磁场全波仿真（OpenParEM）和基于 EC-FDTD 方法的实时电磁信号模拟（openEMS），可构建多模态电磁环境下的任务验证平台。
多智能体集成：与多智能体系统（如 Meta-Transformer、神经符号引擎）结合，可模拟动态电磁环境中多智能体的协同决策过程，验证拓扑流调整机制的有效性。
开源特性：提供模型源码和仿真接口，支持二次开发，满足用户对仿真训练系统的定制化需求。
总结与建议
上述研究成果覆盖了多模态融合、长时推理、多智能体协同、仿真验证等核心技术需求。其中，metaAgent和Meta-Transformer在多模态处理和长序列推理方面具有显著优势，可作为模型开发的基础框架；多智能体电子干扰系统和神经符号混合引擎为动态拓扑流和高准确率任务执行提供了实践方案；OpenParEM/openEMS则为仿真系统构建提供了开源工具链。建议结合具体需求，优先参考metaAgent的工程化实现和Meta-Transformer的泛化能力，同时利用开源仿真工具验证多模态任务流程的可靠性。
"""
    
    # 初始化状态
    initial_state = {
        "user_outline": user_outline,
        "extracted_citations": [],
        "requirements": "",
        "sections": {},
        "draft": "",
        "feedback": [],
        "final_output": "",
        "scores": {},
        "reward": 0.0,
        "temperature": 0.5,
        "episode": 0,
        "total_steps": 0  # 初始化步骤计数
    }
    
    # 执行PPO训练循环（传入追踪回调）
    print("开始PPO训练循环...")
    final_state = research_app.invoke(initial_state, config={"callbacks": callback_manager})
    
    print(f"\n训练完成（{final_state['episode']}轮，{final_state['total_steps']}步）")
    print("最终评分:", final_state["scores"])
    print("最终奖励:", final_state["reward"])
    print("\n=== 最终申报书 ===")
    print(final_state["draft"])