 # -*- coding: utf-8 -*-
from typing import TypedDict, Annotated, Sequence, List, Dict, Tuple, Optional
import re
import numpy as np
from scipy.special import expit
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManager
from langchain_core.tracers.langchain import LangChainTracer

# 加载环境变量
load_dotenv(override=True)

# 初始化回调管理器（用于LangSmith追踪）
callback_manager = CallbackManager([LangChainTracer()])

# 初始化大语言模型
llm = ChatOllama(model="qwen3", temperature=0.5)

# ===== 强化学习策略 (使用REINFORCE算法) =====
class ReinforcePolicy:
    """REINFORCE策略类，用于优化优化多个生成参数"""
    def __init__(self, input_size: int = 3, lr: float = 0.001):
        # 动作空间：仅保留ChatOllama支持的参数
        self.action_dims = 2  # 减少为2个参数
        self.weights = np.random.randn(input_size, self.action_dims) * 0.1
        self.bias = np.zeros(self.action_dims)
        self.optimizer = lr  # 学习率
    
    def forward(self, state_features: np.ndarray) -> Dict[str, float]:
        """根据状态特征预测最佳生成参数组合"""
        output = expit(np.dot(state_features, self.weights) + self.bias)
        
        # 将输出映射到各参数的有效范围（仅保留temperature和top_p）
        return {
            "temperature": 0.1 + output[0] * 0.9,  # 0.1-1.0
            "top_p": 0.5 + output[1] * 0.5,        # 0.5-1.0
        }
    
    def update(self, state: np.ndarray, action: Dict[str, float], reward: float):
        """REINFORCE更新步骤"""
        # 将动作标准化到0-1范围，用于计算策略梯度
        action_values = np.array([
            (action["temperature"] - 0.1) / 0.9,  # 仅处理temperature和top_p
            (action["top_p"] - 0.5) / 0.5,
        ])
        
        # 计算策略梯度 (简化版)
        log_prob = np.sum(np.log(action_values + 1e-8))  # 防止log(0)
        grad_weights = np.outer(state, log_prob * reward)
        grad_bias = log_prob * reward
        
        # 梯度上升更新参数 (最大化奖励)
        self.weights += self.optimizer * grad_weights
        self.bias += self.optimizer * grad_bias

# 初始化REINFORCE策略
rl_policy = ReinforcePolicy()

# ===== 类型定义 =====
class ResearchProposalState(TypedDict):
    user_outline: Annotated[str, "用户输入的提纲"]
    requirements: Annotated[str, "申报指南解析结果"]
    sections: Annotated[dict, "申报书各部分内容"]
    draft: Annotated[str, "完整申报书草稿"]
    feedback: Annotated[Sequence[str], "审核反馈记录"]
    final_output: Annotated[str, "最终申报书"]
    scores: Annotated[Dict[str, float], "评分指标"]
    prev_scores: Annotated[Dict[str, float], "上一轮评分指标"]
    reward: Annotated[float, "强化学习奖励值"]
    generation_params: Annotated[Dict[str, float], "当前生成参数"]
    episode: Annotated[int, "训练轮次"]
    total_steps: Annotated[int, "全局迭代步数（防无限循环）"]
    terminate: Annotated[bool, "是否提前终止"]
    review_focus: Annotated[str, "审核重点"]

# ===== 评分机制 =====
def score_proposal(state: ResearchProposalState) -> Dict[str, float]:
    """评估申报书质量的评分函数"""
    scores = {
        "structure_quality": 0.0,
        "content_relevance": 0.0,
        "language_quality": 0.0
    }
    
    # 1. 结构完整性评分
    required_sections = [
        "国内外研究现状趋势及合作必要性",
        "合作基础、合作互补性",
        "项目合作目标及考核指标",
        "项目研究内容、研究方法及技术路线",
        "任务分解方案",
        "主要创新点",
        "预期外交和经济社会效益",
        "申报单位的已有工作基础、研究成果、研究队伍等",
        "进度安排",
        "项目组织实施、保障措施及风险分析"
    ]
    completed_sections = sum(1 for sec in required_sections if sec in state["sections"])
    scores["structure_quality"] = 10.0 * (completed_sections / len(required_sections))
    
    # 2. 内容相关性评分
    if state["user_outline"] and state["draft"]:
        outline_keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", state["user_outline"])
        draft_keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", state["draft"])
        overlap = len(set(outline_keywords) & set(draft_keywords)) / len(set(outline_keywords)) if outline_keywords else 0
        scores["content_relevance"] = 10.0 * overlap
    
    # 3. 语言质量评分
    if state["draft"]:
        sentences = re.split(r'[。！？,.!?]', state["draft"])
        valid_sentences = [s for s in sentences if s.strip()]
        avg_sentence_length = np.mean([len(s) for s in valid_sentences]) if valid_sentences else 0
        if 15 <= avg_sentence_length <= 30:
            scores["language_quality"] = 8.0
        elif 10 <= avg_sentence_length < 15 or 30 < avg_sentence_length <= 40:
            scores["language_quality"] = 6.0
        else:
            scores["language_quality"] = 3.0
    
    return scores

def calculate_reward(scores: Dict[str, float], prev_scores: Dict[str, float]) -> float:
    """基于评分计算强化学习奖励，包含增量奖励"""
    weights = {
        "structure_quality": 0.4,
        "content_relevance": 0.4,
        "language_quality": 0.2
    }
    
    # 基础奖励
    base_reward = sum(scores[key] * weights[key] for key in scores)
    
    # 增量奖励（仅奖励正向提升）
    if prev_scores and all(key in prev_scores for key in scores):
        delta = sum(max(0, scores[key] - prev_scores[key]) * weights[key] for key in scores)
        return base_reward + delta
    
    return base_reward

# ===== Agent定义 =====
def requirements_analyzer(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始分析需求...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """您是科研项目申报专家，请根据申报指南和用户提纲分析项目要求：
        指南方向: {guideline}
        用户提纲: {user_outline}
        
        请总结该项目的核心要求，包括：
        1. 必须包含的关键部分（根据文件结构）
        2. 项目类型特定要求（如国际合作项目要求）
        3. 格式和字数限制要求"""),
        ("human", "请生成详细的需求分析报告")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "guideline": state.get("requirements", ""),
        "user_outline": state["user_outline"]
    })
    return {
        "requirements": result.content,
        "total_steps": state["total_steps"] + 1
    }

def content_generator(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始生成内容...")
    sections_to_generate = [
        "国内外研究现状趋势及合作必要性",
        "合作基础、合作互补性",
        "项目合作目标及考核指标",
        "项目研究内容、研究方法及技术路线",
        "任务分解方案",
        "主要创新点",
        "预期外交和经济社会效益",
        "申报单位的已有工作基础、研究成果、研究队伍等",
        "进度安排",
        "项目组织实施、保障措施及风险分析"
    ]
    
    # 优先生成未完成或评分低的章节
    if state["scores"]:
        section_quality = {}
        for section in sections_to_generate:
            if section in state["sections"]:
                section_quality[section] = state["scores"].get("structure_quality", 0)
            else:
                section_quality[section] = -1  # 未生成章节优先级最高
        
        sections_to_generate = sorted(sections_to_generate, key=lambda x: section_quality[x])
    
    # 强化学习状态特征
    state_features = np.array([
        len(state["feedback"]),
        state["scores"].get("content_relevance", 0) / 10,
        min(state["total_steps"] / 30, 1.0),  # 归一化到0-1
    ])
    
    current_params = rl_policy.forward(state_features)
    print(f"[内容生成] 强化学习调整参数为: {current_params}")
    
    # 审核重点提示
    focus_prompt = ""
    if state["review_focus"] == "structure":
        focus_prompt = "特别注意章节结构的完整性和逻辑性，严格按照文件结构要求。"
    elif state["review_focus"] == "content":
        focus_prompt = "特别注意内容与用户提纲和指南的相关性和深度。"
    elif state["review_focus"] == "language":
        focus_prompt = "特别注意语言表达的准确性、流畅性和专业性，符合学术规范。"
    
    # 生成或更新章节
    new_sections = state["sections"].copy()  # 保留已有内容
    for section in sections_to_generate:
        # 保留高质量章节
        section_score = state["scores"].get("structure_quality", 0) + \
                state["scores"].get("content_relevance", 0) + \
                state["scores"].get("language_quality", 0)

        if section in new_sections and section_score / 3 > 7:
            print(f"[内容生成] '{section}' 质量综合较高，跳过重新生成")
            continue
            
        # 应用生成参数
        llm.temperature = current_params["temperature"]
        llm.top_p = current_params["top_p"]
        
        # 调整提示词，使内容符合文件结构要求
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""您是科研申报专家，撰写'{section}'部分。{focus_prompt}
            严格按照文件结构要求生成内容，确保格式规范。参考以下信息：
            项目指南: {state['requirements']}
            用户提纲: {state['user_outline']}
            
            注意字数限制和专业术语使用。"""),
            ("human", "请生成符合要求的'{section}'部分内容")
        ])
        chain = prompt | llm
        result = chain.invoke({
            "section": section,
            "requirements": state["requirements"],
            "user_outline": state["user_outline"]
        })
        new_sections[section] = result.content
        print(f"[内容生成] 完成'{section}'部分")
    
    return {
        "sections": new_sections,
        "generation_params": current_params,
        "total_steps": state["total_steps"] + 1
    }

def proposal_integrator(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始整合提案...")
    draft = "# 国际合作科研项目申报书\n\n"
    
    # 按照文件结构组织内容
    draft += "## 第一部分 国内外现状趋势分析及合作理由\n"
    draft += "### 一、国内外研究现状趋势及合作必要性\n"
    draft += state["sections"].get("国内外研究现状趋势及合作必要性", "（待补充内容）") + "\n\n"
    
    draft += "### 二、合作基础、合作互补性\n"
    draft += state["sections"].get("合作基础、合作互补性", "（待补充内容）") + "\n\n"
    
    draft += "## 第二部分 研究目标及内容\n"
    draft += "### 一、项目合作目标及考核指标\n"
    draft += state["sections"].get("项目合作目标及考核指标", "（待补充内容）") + "\n\n"
    
    draft += "### 二、项目研究内容、研究方法及技术路线\n"
    draft += state["sections"].get("项目研究内容、研究方法及技术路线", "（待补充内容）") + "\n\n"
    
    draft += "### 三、任务分解方案\n"
    draft += state["sections"].get("任务分解方案", "（待补充内容）") + "\n\n"
    
    draft += "### 四、主要创新点\n"
    draft += state["sections"].get("主要创新点", "（待补充内容）") + "\n\n"
    
    draft += "### 五、预期外交和经济社会效益\n"
    draft += state["sections"].get("预期外交和经济社会效益", "（待补充内容）") + "\n\n"
    
    draft += "## 第三部分 申报单位及参与单位研究基础\n"
    draft += "### 一、申报单位的已有工作基础、研究成果、研究队伍等\n"
    draft += state["sections"].get("申报单位的已有工作基础、研究成果、研究队伍等", "（待补充内容）") + "\n\n"
    
    draft += "## 第四部分 进度安排\n"
    draft += state["sections"].get("进度安排", "（待补充内容）") + "\n\n"
    
    draft += "## 第五部分 项目组织实施、保障措施及风险分析\n"
    draft += "### 一、项目组织实施机制\n"
    draft += "### 二、保障措施\n"
    draft += "### 三、知识产权对策、成果管理及合作权益分配\n"
    draft += "### 四、风险分析及对策\n"
    draft += state["sections"].get("项目组织实施、保障措施及风险分析", "（待补充内容）") + "\n"
    
    print(f"[提案整合] 完成草稿生成，共{len(draft)}字符")
    return {
        "draft": draft,
        "total_steps": state["total_steps"] + 1
    }

def quality_reviewer(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始质量审核...")
    
    # 动态调整审核重点
    if state["scores"]:
        lowest_score_dim = min(state["scores"], key=state["scores"].get)
        if lowest_score_dim == "structure_quality":
            state["review_focus"] = "structure"
        elif lowest_score_dim == "content_relevance":
            state["review_focus"] = "content"
        elif lowest_score_dim == "language_quality":
            state["review_focus"] = "language"
        else:
            state["review_focus"] = "all"
    else:
        state["review_focus"] = "all"
    
    print(f"[质量审核] 本次审核重点：{state['review_focus']}")
    
    # 审核提示词
    focus_instructions = ""
    if state["review_focus"] == "structure":
        focus_instructions = "重点检查申报书是否严格遵循文件结构要求，章节是否完整，格式是否正确。"
    elif state["review_focus"] == "content":
        focus_instructions = "重点检查申报书内容是否符合指南要求，是否完整覆盖用户提纲内容。"
    elif state["review_focus"] == "language":
        focus_instructions = "重点检查申报书的语言表达质量，包括准确性、流畅性、专业性和学术规范性。"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""您是科研项目评审专家，{focus_instructions}请基于以下要求进行审核：
        1. 是否符合文件结构要求
        2. 是否满足指南具体要求
        3. 是否完整反映用户提纲内容
        4. 语言表达是否专业规范
        
        请给出具体修改建议。"""),
        ("human", "申报书草稿：{draft}\n指南要求：{requirements}\n用户提纲：{user_outline}\n请给出详细审核意见：")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "draft": state["draft"],
        "requirements": state["requirements"],
        "user_outline": state["user_outline"]
    })
    
    # 计算评分和奖励
    new_scores = score_proposal(state)
    new_reward = calculate_reward(new_scores, state["scores"])
    
    feedback = state["feedback"] + [result.content]
    print(f"[质量审核] 完成审核，评分：{new_scores}，奖励：{new_reward:.2f}")
    print(f"[质量审核] 反馈摘要：{result.content[:100]}...")
    return {
        "feedback": feedback,
        "scores": new_scores,
        "prev_scores": state["scores"],
        "reward": new_reward,
        "total_steps": state["total_steps"] + 1
    }

def rl_learner(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始强化学习更新...")
    """强化学习学习节点，根据奖励更新策略"""
    if state["episode"] > 0:  # 至少完成一个回合才更新
        state_features = np.array([
            len(state["feedback"]),
            state["scores"].get("content_relevance", 0) / 10,
            min(state["total_steps"] / 30, 1.0),
        ])
        
        # 更新REINFORCE策略
        rl_policy.update(
            state=state_features,
            action=state["generation_params"],
            reward=state["reward"]
        )
        print(f"[强化学习] 策略更新完成，奖励值：{state['reward']:.2f}")
    
    # 检查是否达到质量标准
    if state["scores"]:
        avg_score = sum(state["scores"].values()) / len(state["scores"])
        if avg_score > 6.0:
            print(f"[强化学习] 质量达标（平均分{avg_score:.2f}），提前终止训练")
            return {
                "episode": state["episode"] + 1,
                "feedback": [],
                "terminate": True,
                "total_steps": state["total_steps"] + 1
            }
    
    # 准备下一轮（保留高质量章节）
    next_episode = state["episode"] + 1
    print(f"[强化学习] 完成第{state['episode']}轮，进入第{next_episode}轮")
    return {
        "episode": next_episode,
        "feedback": [],
        "terminate": False,
        "total_steps": state["total_steps"] + 1
    }

def revision_director(state: ResearchProposalState):
    """审核结果导向器，决定下一步流程"""
    if state.get("terminate", False):
        return "END"
    
    last_feedback = state["feedback"][-1] if state["feedback"] else ""
    
    # 限制修改次数
    if len(state["feedback"]) >= 2:  # 修改次数达2次即进入学习阶段
        print("[审核导向] 修改次数达2次，强制进入学习阶段")
        return "rl_learn"
    
    # 检查是否有明显问题
    if "不完整" in last_feedback or "格式" in last_feedback:
        print("[审核导向] 发现结构问题，返回内容生成")
        return "generate_content"
    elif "不相关" in last_feedback or "指南要求" in last_feedback:
        print("[审核导向] 内容相关性不足，返回内容生成")
        return "generate_content"
    elif "语言问题" in last_feedback or "表达" in last_feedback:
        print("[审核导向] 语言表达需改进，返回内容生成")
        return "generate_content"
    
    print("[审核导向] 审核通过，进入强化学习")
    return "rl_learn"

def check_termination(state: ResearchProposalState):
    """检查是否需要强制终止流程"""
    max_steps = 30
    max_episodes = 3
    
    # 检查步数限制
    if state["total_steps"] >= max_steps:
        print(f"\n[终止检查] 已达到最大步骤({max_steps})，强制终止流程")
        return "force_end"
    
    # 检查轮次限制
    if state["episode"] >= max_episodes:
        print(f"\n[终止检查] 已达到最大训练轮次({max_episodes})，强制终止流程")
        return "force_end"
    
    return "continue"

# ===== 工作流构建 =====
workflow = StateGraph(ResearchProposalState)

# 添加节点
workflow.add_node("analyze_requirements", requirements_analyzer)
workflow.add_node("generate_content", content_generator)
workflow.add_node("integrate_proposal", proposal_integrator)
workflow.add_node("check_termination", lambda s: {"total_steps": s["total_steps"] + 1})  # 步骤计数
workflow.add_node("review_quality", quality_reviewer)
workflow.add_node("rl_learn", rl_learner)

# 设置入口
workflow.set_entry_point("analyze_requirements")

# 主流程：加入终止检查节点
workflow.add_edge("analyze_requirements", "generate_content")
workflow.add_edge("generate_content", "integrate_proposal")
workflow.add_edge("integrate_proposal", "check_termination")  # 先检查是否终止

# 终止检查后分支
workflow.add_conditional_edges(
    "check_termination",
    check_termination,
    {"force_end": END, "continue": "review_quality"}
)

# 审核后条件分支
workflow.add_conditional_edges(
    "review_quality",
    revision_director,
    {
        "analyze_requirements": "analyze_requirements",
        "generate_content": "generate_content",
        "rl_learn": "rl_learn",
        "END": END
    }
)

# 强化学习后循环或结束
workflow.add_conditional_edges(
    "rl_learn",
    lambda state: "END" if state["terminate"] or state["episode"] >= 3 else "analyze_requirements",
    {"END": END, "analyze_requirements": "analyze_requirements"}
)

# 执行引擎
research_app = workflow.compile()

# ===== 执行示例 =====
if __name__ == "__main__":
    user_outline = """
中国和韩国政府间能源技术联合研究项目
研究目标:围绕氢能技术发展需求，开展海水制氢与消纳关键技术研究，突破海水制氢设备研制和产品化技术，开发水陆空氢电混动智能无人交通运载工具，构建绿电海水制氢消纳一体化管理平台。

技术需求:
1)海水制氢效率不低于60%
2)氢电混动系统能量转换效率不低于85%
3)一体化管理平台支持实时监控和智能调度

成果形式:技术研究报告2份、专利3项、产品样机2套、管理平台1套
研究周期:3年
项目类别:中韩政府间合作项目
拟资助经费:750万元人民币
"""

    guideline = """
中国和韩国政府间能源技术联合研究项目
合作协议:《中国科技部与韩国产业通商资源部关于开展2025 年能源技术联合研发项目合作的备忘录》。
领域方向:氢能;光伏。
拟支持项目数:不超过2个
共拟支持经费:1500万元人民币(每个项目不超过750万元)其他要求:
(1)项目执行期为3年。
(2)中方项目申报单位必须为企业(韩方有关要求参见韩方指南)
"""
    
    # 初始化状态
    initial_state = {
        "user_outline": user_outline,
        "requirements": guideline,
        "sections": {},
        "draft": "",
        "feedback": [],
        "final_output": "",
        "scores": {},
        "prev_scores": {},
        "reward": 0.0,
        "generation_params": {
            "temperature": 0.5,
            "top_p": 0.7,
        },
        "episode": 0,
        "total_steps": 0,
        "terminate": False,
        "review_focus": "all"
    }
    
    # 执行流程
    print("开始强化学习训练循环...")
    final_state = research_app.invoke(initial_state, config={"callbacks": callback_manager})
    
    print(f"\n训练完成（{final_state['episode']}轮，{final_state['total_steps']}步）")
    print("最终评分:", final_state["scores"])
    print("最终奖励:", final_state["reward"])
    print("\n=== 最终申报书 ===")
    print(final_state["draft"])