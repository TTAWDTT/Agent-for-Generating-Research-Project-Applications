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
    user_outline: Annotated[str, "用户输入的提纲（含引文）"]
    extracted_citations: Annotated[List[str], "从提纲中提取的引文列表"]
    used_citations: Annotated[List[str], "已使用的引文列表"]
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
    required_sections = ["项目名称", "研究背景", "研究内容", "研究目标", 
                        "技术路线", "创新点", "预期成果", "研究基础"]
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

# ===== 辅助函数 =====
def extract_citations(outline: str) -> List[str]:
    """从提纲中提取引文"""
    citations_start = outline.find("引文如下：")
    if citations_start == -1:
        print("未找到'引文如下：'标记，无法提取引文")
        return []
    citations_text = outline[citations_start + len("引文如下："):]
    
    # 优化正则：兼容数字+点/括号格式
    citation_blocks = re.split(r"\n\s*(\d+\.|\(\d+\))\s*", citations_text.strip())
    
    citations = []
    for i in range(1, len(citation_blocks), 2):
        if i + 1 >= len(citation_blocks):
            continue
            
        num = citation_blocks[i].strip()
        content = citation_blocks[i+1].strip()
        
        if "作者：" in content and "要点：" in content:
            citations.append(f"{num} {content}")
    
    # 去重并保留顺序
    seen = set()
    unique_citations = []
    for cit in citations:
        if cit not in seen:
            seen.add(cit)
            unique_citations.append(cit)
    
    print(f"提取到{len(unique_citations)}条引文")
    for i, cit in enumerate(unique_citations, 1):
        truncated = cit[:60] + "..." if len(cit) > 60 else cit
        print(f"引文{i}：{truncated}")
    
    return unique_citations

def check_used_citations(draft: str, all_citations: List[str]) -> List[str]:
    """检查哪些引文已在草稿中使用（更健壮的匹配）"""
    used = []
    for citation in all_citations:
        # 提取引文唯一标识（序号+作者）
        id_match = re.match(r"^(\d+\.|\(\d+\))\s*作者：([^，,]+)", citation)
        if id_match:
            citation_id = id_match.group(1) + id_match.group(2)
        else:
            citation_id = citation.split('\n')[0]  #  fallback
        
        if citation_id in draft:
            used.append(citation)
    return used

# ===== Agent定义 =====
def requirements_analyzer(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始分析需求...")
    citations = extract_citations(state["user_outline"])
    print(f"[需求分析] 提取到{len(citations)}条引文")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "您是科研项目申报专家，分析项目类型和要求，可参考提纲中提供的引文：{citations}"),
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
        "used_citations": [],
        "total_steps": state["total_steps"] + 1
    }

def content_generator(state: ResearchProposalState):
    print(f"\n[步骤{state['total_steps']}] 开始生成内容...")
    sections_to_generate = [
        "项目名称", "研究背景", "研究内容", "研究目标",
        "技术路线", "创新点", "预期成果", "研究基础"
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
    
    # 强化学习状态特征（移除引文相关特征）
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
        focus_prompt = "特别注意章节结构的完整性和逻辑性。"
    elif state["review_focus"] == "content":
        focus_prompt = "特别注意内容与用户提纲的相关性和深度。"
    elif state["review_focus"] == "language":
        focus_prompt = "特别注意语言表达的准确性、流畅性和专业性。"
    
    # 章节长度配置
    section_lengths = {
        "研究背景": 800, "研究内容": 1200, "技术路线": 1000,
        "创新点": 600, "预期成果": 600, "研究基础": 800,
        "项目名称": 100, "研究目标": 500
    }
    
    # 生成或更新章节
    new_sections = state["sections"].copy()  # 保留已有内容
    for section in sections_to_generate:
        # 保留高质量章节
        if section in new_sections and state["scores"].get("structure_quality", 0) > 7:
            print(f"[内容生成] '{section}' 质量较高，跳过重新生成")
            continue
            
        # 应用生成参数
        llm.temperature = current_params["temperature"]
        llm.top_p = current_params["top_p"]
        
        max_tokens = section_lengths.get(section, 800)
        # 调整提示词，使引文作为辅助参考
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""您是科研申报专家，撰写'{section}'部分。{focus_prompt}
            可参考提纲中的引文作为辅助：{state['extracted_citations']}，
            引用时保留原格式，解释引文与内容的关联（如适用）。控制在{max_tokens}字左右。"""),
            ("human", "根据提纲生成内容：\n{user_outline}")
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
    # 调整引文使用方式，作为参考信息
    if state["extracted_citations"]:
        draft += "参考相关研究成果：\n"
        for citation in state["extracted_citations"]:
            draft += f"- {citation}的研究表明..."
    else:
        draft += "（请补充相关领域研究进展）\n"
    draft += "\n\n2. 现有研究不足：\n"
    draft += "（结合相关研究分析当前研究空白）\n\n"
    
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
    
     # 6. 研究计划与进度安排（动态版本）
    draft += "## 六、研究计划与进度\n"
    draft += "| 时间阶段 | 主要工作内容 | 阶段目标 |\n"
    draft += "|----------|--------------|----------|\n"
    
    # 从用户提纲提取研究周期（默认2年）
    period_match = re.search(r"研究周期：(\d+)年", state["user_outline"])
    total_years = int(period_match.group(1)) if period_match else 2
    total_months = total_years * 12  # 总月数
    
    # 按研究阶段比例分配时间（可根据总周期自适应）
    stages = [
        {
            "ratio": 0.2,  # 占总周期的20%
            "title": "文献调研与方案设计",
            "goal": "完成文献综述、确定技术路线"
        },
        {
            "ratio": 0.5,  # 50%
            "title": "核心技术研发与实验验证",
            "goal": "实现关键技术突破、完成原型系统开发"
        },
        {
            "ratio": 0.3,  # 30%
            "title": "系统优化与成果整理",
            "goal": "完成系统测试、撰写研究报告与论文"
        }
    ]
    
    # 计算各阶段时间范围
    current_month = 1
    for i, stage in enumerate(stages, 1):
        stage_months = int(round(total_months * stage["ratio"], 0))
        end_month = current_month + stage_months - 1
        
        # 格式化时间显示（如"第1-3月"、"第4-10月"）
        if current_month == end_month:
            time_range = f"第{current_month}月"
        else:
            time_range = f"第{current_month}-{end_month}月"
        
        draft += f"| {time_range} | {stage['title']} | {stage['goal']} |\n"
        current_month = end_month + 1  # 进入下一阶段
    
    draft += "\n"  # 换行分隔
    
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
    
    # 9. 经费预算（动态计算版本）
    draft += "## 九、经费预算（单位：元）\n"
    draft += "| 预算科目 | 金额 | 计算依据 |\n"
    draft += "|----------|------|----------|\n"
    
    # 从用户提纲提取拟资助经费（如果存在）
    fund_match = re.search(r"拟资助经费：(\d+)万", state["user_outline"])
    total_fund = int(fund_match.group(1)) * 10000 if fund_match else 300000  # 默认30万
    
    # 按比例分配预算
    budget_items = {
        "文献资料费": 0.05,    # 5%
        "实验材料费": 0.4,     # 40%
        "差旅费": 0.15,        # 15%
        "设备使用费": 0.2,     # 20%
        "劳务费": 0.15,        # 15%
        "其他费用": 0.05       # 5%
    }
    
    for item, ratio in budget_items.items():
        amount = int(total_fund * ratio)
        # 根据科目设置计算依据
        basis = {
            "文献资料费": "数据库订阅与文献购买",
            "实验材料费": "实验耗材与样本采集",
            "差旅费": "学术交流与数据采集",
            "设备使用费": "仪器租赁与维护",
            "劳务费": "参与人员补助",
            "其他费用": "不可预见支出"
        }[item]
        draft += f"| {item} | {amount} | {basis} |\n"
    
    draft += f"| 合计 | {total_fund} | |\n\n"
    
    # 检查已使用引文（作为辅助信息）
    used_citations = check_used_citations(draft, state["extracted_citations"])
    missing = [c for c in state["extracted_citations"] if c not in used_citations]
    
    if missing and len(missing) < len(state["extracted_citations"])/2:
        draft += "## 十、参考资料说明\n"
        draft += f"部分参考资料未在正文中直接引用：{', '.join([c.split(chr(10))[0] for c in missing])}\n"
    
    print(f"[提案整合] 完成草稿生成，共{len(draft)}字符，参考了{len(used_citations)}/{len(state['extracted_citations'])}条引文")
    return {
        "draft": draft,
        "used_citations": used_citations,
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
        focus_instructions = "重点检查申报书的章节结构是否完整，逻辑是否连贯，各部分内容是否符合学术规范。"
    elif state["review_focus"] == "content":
        focus_instructions = "重点检查申报书内容与用户提纲的相关性，以及内容的深度和科学性。"
    elif state["review_focus"] == "language":
        focus_instructions = "重点检查申报书的语言表达质量，包括准确性、流畅性、专业性和可读性。"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"您是科研项目评审专家，{focus_instructions}请基于以下要求进行审核并给出具体修改建议。"),
        ("human", "申报书草稿：{draft}\n需求要求：{requirements}\n请给出详细审核意见：")
    ])
    chain = prompt | llm
    result = chain.invoke({
        "draft": state["draft"],
        "requirements": state["requirements"]
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
        if avg_score > 8.0:
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
    if len(state["feedback"]) >= 2 and "建议修改" in last_feedback:
        print("[审核导向] 修改次数达2次，强制进入学习阶段")
        return "rl_learn"
    
    if "结构不完整" in last_feedback:
        print("[审核导向] 发现结构问题，返回内容生成")
        return "generate_content"
    elif "内容不相关" in last_feedback:
        print("[审核导向] 内容相关性不足，返回内容生成")
        return "generate_content"
    elif "语言问题" in last_feedback:
        print("[审核导向] 语言表达需改进，返回内容生成")
        return "generate_content"
    
    print("[审核导向] 审核通过，进入强化学习")
    return "rl_learn"

def check_termination(state: ResearchProposalState):
    """检查是否需要强制终止流程"""
    max_steps = 30
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
        "used_citations": [],
        "requirements": "",
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
