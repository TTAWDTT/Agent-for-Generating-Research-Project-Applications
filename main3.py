import os
import re
import json
import logging
import yaml
import time
import psutil
import subprocess
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import langchain

# 修复导入警告
try:
    from langchain_community.cache import SQLiteCache
except ImportError:
    from langchain.cache import SQLiteCache

# ========== 日志配置 ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ========== 环境变量与缓存 ==========
load_dotenv(override=True)
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
logger.info("LLM缓存已启用 (SQLiteCache at ./.langchain.db)")

# ---------------------------- 状态定义 ----------------------------
class State(TypedDict):
    tree: Annotated[List[Dict[str, Any]], "模板树结构，带内容"]
    flat: Annotated[List[Dict[str, Any]], "扁平节点列表引用"]
    template_content: Annotated[str, "原始模板内容"]
    logical_analysis: Annotated[str, "LLM对模板逻辑的分析"]
    writing_order: Annotated[List[int], "智能排序后的写作顺序"]
    current_index: Annotated[int, "当前写作进度索引"]
    completed_sections: Annotated[str, "已完成的章节内容"]

# ---------------------------- 系统监控函数 ----------------------------
def check_system_resources():
    """检查系统资源状态"""
    try:
        memory = psutil.virtual_memory()
        logger.info(f"内存使用率: {memory.percent}% (可用: {memory.available // (1024**3)}GB)")
        
        if memory.percent > 90:
            logger.warning("系统内存使用率过高，可能影响程序稳定性")
            return False
        return True
    except Exception as e:
        logger.warning(f"无法检查系统资源: {e}")
        return True

def check_ollama_status():
    """检查Ollama服务状态"""
    try:
        # 检查Ollama进程
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("Ollama服务正常运行")
            return True
        else:
            logger.error("Ollama服务异常")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama命令超时")
        return False
    except Exception as e:
        logger.error(f"检查Ollama状态失败: {e}")
        return False

def restart_ollama_if_needed():
    """必要时重启Ollama服务"""
    try:
        if not check_ollama_status():
            logger.info("尝试重启Ollama服务...")
            subprocess.run(['ollama', 'serve'], timeout=5)
            time.sleep(5)
            return check_ollama_status()
        return True
    except Exception as e:
        logger.error(f"重启Ollama失败: {e}")
        return False

# ---------------------------- 工具函数 ----------------------------
def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_checkpoint(state: State, checkpoint_file: str = "checkpoint.json"):
    """保存检查点"""
    try:
        checkpoint_data = {
            "current_index": state.get("current_index", 0),
            "completed_sections": state.get("completed_sections", ""),
            "logical_analysis": state.get("logical_analysis", ""),
            "writing_order": state.get("writing_order", []),
            "tree_content": {n["id"]: n["content"] for n in state.get("flat", []) if n["content"]},
            "timestamp": time.time()
        }
        
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        logger.info(f"检查点已保存: {checkpoint_file}")
    except Exception as e:
        logger.warning(f"保存检查点失败: {e}")

def load_checkpoint(state: State, checkpoint_file: str = "checkpoint.json") -> bool:
    """加载检查点"""
    try:
        if not os.path.exists(checkpoint_file):
            return False
            
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        
        # 恢复状态
        state["current_index"] = checkpoint_data.get("current_index", 0)
        state["completed_sections"] = checkpoint_data.get("completed_sections", "")
        state["logical_analysis"] = checkpoint_data.get("logical_analysis", "")
        state["writing_order"] = checkpoint_data.get("writing_order", [])
        
        # 恢复树中的内容
        if "flat" in state and state["flat"]:
            id_to_node = {n["id"]: n for n in state["flat"]}
            for node_id, content in checkpoint_data.get("tree_content", {}).items():
                if int(node_id) in id_to_node:
                    id_to_node[int(node_id)]["content"] = content
        
        timestamp = checkpoint_data.get("timestamp", 0)
        logger.info(f"检查点已加载 (保存时间: {time.ctime(timestamp)})")
        logger.info(f"从第 {state['current_index'] + 1} 个章节继续")
        return True
    except Exception as e:
        logger.warning(f"加载检查点失败: {e}")
        return False

def safe_llm_invoke(prompt_chain, inputs: Dict[str, Any], max_retries: int = 5, base_wait: int = 5) -> str:
    """安全的LLM调用，包含增强的重试机制和超时控制"""
    global llm
    
    for attempt in range(max_retries):
        try:
            # 每次调用前检查系统状态
            if not check_system_resources():
                logger.warning("系统资源不足，等待30秒...")
                time.sleep(30)
            
            if attempt > 0:  # 第一次重试前检查Ollama
                if not restart_ollama_if_needed():
                    logger.error("无法恢复Ollama服务")
                    time.sleep(base_wait * 2)
                    continue
            
            logger.info(f"LLM调用尝试 {attempt + 1}/{max_retries}")
            start_time = time.time()
            
            # Windows兼容的超时控制
            import threading
            import queue
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def llm_call_worker():
                try:
                    chain = prompt_chain | llm
                    result = chain.invoke(inputs)
                    result_queue.put(result)
                except Exception as e:
                    exception_queue.put(e)
            
            # 启动工作线程
            worker_thread = threading.Thread(target=llm_call_worker)
            worker_thread.daemon = True
            worker_thread.start()
            
            # 等待结果，最多等待60秒
            worker_thread.join(timeout=60.0)
            
            if worker_thread.is_alive():
                # 线程还在运行，说明超时了
                logger.error("LLM调用超时 (60秒)")
                raise TimeoutError("LLM调用超时")
            
            # 检查是否有异常
            if not exception_queue.empty():
                raise exception_queue.get()
            
            # 获取结果
            if result_queue.empty():
                raise Exception("LLM调用没有返回结果")
                
            result = result_queue.get()
            
            elapsed_time = time.time() - start_time
            
            # 处理返回结果
            if hasattr(result, 'content'):
                content = result.content
                logger.debug(f"LLM返回AIMessage类型, content长度: {len(content) if content else 0}")
            elif hasattr(result, 'text'):
                content = result.text
                logger.debug(f"LLM返回带text属性的对象，长度: {len(content) if content else 0}")
            elif isinstance(result, str):
                content = result
                logger.debug(f"LLM返回字符串类型，长度: {len(content)}")
            else:
                # 如果是其他类型，尝试转换但排除系统消息
                result_str = str(result)
                if "SystemMessage" in result_str or "HumanMessage" in result_str:
                    raise Exception("LLM返回了原始消息对象而不是生成的内容")
                content = result_str
                logger.debug(f"LLM返回其他类型: {type(result)}, 转换后长度: {len(content)}")
            
            logger.info(f"LLM调用成功，耗时: {elapsed_time:.2f}秒，返回内容长度: {len(content) if content else 0}")
            
            # 基础质量检查
            if not content:
                raise Exception("LLM返回内容为空")
            
            content = content.strip()
            if len(content) < 20:
                raise Exception(f"LLM返回内容过短: '{content[:50]}...'")
                
            return content
            
        except TimeoutError:
            logger.error(f"LLM调用超时 (尝试 {attempt + 1}/{max_retries})")
        except Exception as e:
            elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            logger.error(f"失败时间: {elapsed_time:.2f}秒")
        
        if attempt < max_retries - 1:
            wait_time = base_wait * (2 ** attempt)  # 指数退避
            wait_time = min(wait_time, 120)  # 最大等待2分钟
            logger.info(f"等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
            
            # 重新初始化LLM连接
            try:
                logger.info("重新初始化LLM连接...")
                llm = ChatOllama(
                    model=CONFIG["模型"], 
                    temperature=0.3,
                    model_kwargs={
                        "num_ctx": 8192,
                        "num_predict": -1,
                        "repeat_penalty": 1.1
                    }
                )
                logger.info("LLM连接重新初始化成功")
            except Exception as init_error:
                logger.error(f"LLM重新初始化失败: {init_error}")
                time.sleep(10)
        else:
            logger.error(f"LLM调用彻底失败，已重试{max_retries}次")
    
    raise Exception("LLM调用失败")

# ---------------------------- Markdown解析函数 ----------------------------
def parse_markdown_template(md: str) -> List[Dict[str, Any]]:
    """解析markdown模板，改进多级嵌套标题处理"""
    lines = md.splitlines()
    tree: List[Dict[str, Any]] = []
    stack: List[Tuple[int, Dict[str, Any]]] = []
    in_code = False
    current: Optional[Dict[str, Any]] = None

    def new_node(level: int, title: str) -> Dict[str, Any]:
        return {
            "id": None, "level": level, "title": title.strip(),
            "instructions": "", "placeholder": "", "content": "", "children": []
        }

    for line in lines:
        # 检测代码块
        if line.strip().startswith("```"):
            in_code = not in_code
            if current: 
                current["placeholder"] += line + "\n"
            continue
        
        # 处理标题行 - 改进嵌套结构识别
        if re.match(r"^#{1,6}\s+", line) and not in_code:
            # 准确计算标题级别
            level = len(line) - len(line.lstrip('#'))
            title = line.strip('#').strip()
            
            # 跳过系统级标题
            skip_titles = ["申请书", "填报说明", "项目申请简表", "申请经费预算表", "审查意见", "推荐意见"]
            if any(skip in title for skip in skip_titles):
                continue
            
            node = new_node(level, title)
            
            # 改进的树结构维护逻辑
            while stack and stack[-1][0] >= level:
                stack.pop()
                
            if not stack:
                tree.append(node)
            else:
                stack[-1][1]["children"].append(node)
                
            stack.append((level, node))
            current = node
            continue

        if current is None: 
            continue

        # 处理表格和代码块作为占位符
        is_table_like = line.strip().startswith("|") or re.match(r"^\s*[-|:+]+\s*$", line)
        if in_code or is_table_like:
            current["placeholder"] += line + "\n"
        else:
            # 其他内容作为指导信息
            current["instructions"] += line + "\n"

    # 分配ID
    def assign_ids(nodes: List[Dict[str, Any]], start_id: int = 0) -> int:
        cid = start_id
        for n in nodes:
            n["id"] = cid
            cid += 1
            cid = assign_ids(n["children"], cid)
        return cid
    assign_ids(tree, 0)
    
    return tree

def flatten_tree(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """扁平化树结构"""
    out: List[Dict[str, Any]] = []
    def walk(nodes: List[Dict[str, Any]]):
        for n in nodes:
            out.append(n)
            if n["children"]: 
                walk(n["children"])
    walk(tree)
    return out

def parent_chain_titles(tree: List[Dict[str, Any]], target_id: int) -> str:
    """获取父链标题"""
    chain: List[str] = []
    found = False
    def dfs(nodes: List[Dict[str, Any]], path: List[str]):
        nonlocal chain, found
        for n in nodes:
            new_path = path + [n["title"]]
            if n["id"] == target_id:
                chain = new_path[:-1]
                found = True
                return
            dfs(n["children"], new_path)
            if found: return
    dfs(tree, [])
    return " > ".join(chain) if chain else "根目录"

def assemble_markdown(tree: List[Dict[str, Any]], include_empty: bool = False) -> str:
    """组装最终的Markdown文档"""
    out: List[str] = []
    def walk(nodes: List[Dict[str, Any]]):
        for n in nodes:
            out.append(f"{'#' * n['level']} {n['title']}")
            content = (n["content"] or "").strip()
            if content or include_empty:
                out.append(content if content else "[待完成]")
                out.append("")
            if n["children"]: 
                walk(n["children"])
    walk(tree)
    return "\n".join(out).strip() + "\n"

def word_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))

# ---------------------------- 层次化写作顺序确定 ----------------------------
def determine_hierarchical_writing_order(tree: List[Dict[str, Any]], flat_nodes: List[Dict[str, Any]]) -> List[int]:
    """层次化确定写作顺序：只写叶子节点，但按层级递归排序"""
    
    def collect_leaf_nodes_recursively(nodes: List[Dict[str, Any]]) -> List[int]:
        """递归收集叶子节点，保持层级顺序"""
        leaf_ids = []
        
        for node in nodes:
            if len(node["children"]) == 0:
                # 这是叶子节点，加入写作列表
                # 排除系统性章节
                if not any(keyword in node["title"] for keyword in ["简表", "预算表", "审查意见", "推荐意见"]):
                    leaf_ids.append(node["id"])
                    logger.info(f"加入写作队列: {node['title']} (Level {node['level']})")
            else:
                # 这是父节点，递归处理其子节点
                child_leaves = collect_leaf_nodes_recursively(node["children"])
                leaf_ids.extend(child_leaves)
        
        return leaf_ids
    
    # 从根节点开始递归收集
    writing_order = collect_leaf_nodes_recursively(tree)
    
    logger.info(f"层次化写作顺序确定完成，共 {len(writing_order)} 个叶子节点")
    return writing_order

# ---------------------------- 配置与 LLM ----------------------------
PROJECT_ROOT = Path(__file__).parent
CONFIG = load_yaml(PROJECT_ROOT / "config.yaml")

CONFIG.setdefault("输出路径", "output")
CONFIG.setdefault("单章字数", 0)
CONFIG.setdefault("领域", ["通用技术"])

if isinstance(CONFIG.get("格式"), str):
    CONFIG["格式"] = CONFIG["格式"].replace("\\", os.sep).replace("/", os.sep)

required_fields = ["模型", "项目名称", "格式"]
for field in required_fields:
    if field not in CONFIG:
        raise ValueError(f"config.yaml 中缺少必需字段: {field}")

PROMPTS = load_yaml(PROJECT_ROOT / "prompts_template.yaml")

llm = ChatOllama(
    model=CONFIG["模型"], 
    temperature=0.3,
    model_kwargs={
        "num_ctx": 8192,
        "num_predict": -1,
        "repeat_penalty": 1.1
    }
)

# ---------------------------- 工作流节点 ----------------------------
def init_node(state: State) -> Dict[str, Any]:
    """初始化：加载并解析模板"""
    logger.info("初始化：加载模板并解析")
    
    # 系统检查
    check_system_resources()
    check_ollama_status()
    
    tpath = PROJECT_ROOT / CONFIG["格式"]
    if not tpath.exists(): 
        raise FileNotFoundError(f"模板文件未找到: {tpath}")
    
    template_content = tpath.read_text(encoding="utf-8")
    tree = parse_markdown_template(template_content)
    flat = flatten_tree(tree)
    
    logger.info(f"模板解析：总节点 {len(flat)}")
    for node in flat:
        logger.debug(f"节点 {node['id']}: Level {node['level']} - {node['title']} (子节点: {len(node['children'])})")
    
    new_state = {
        "tree": tree, 
        "flat": flat, 
        "template_content": template_content,
        "current_index": 0,
        "completed_sections": "",
        "logical_analysis": "",
        "writing_order": []
    }
    
    if load_checkpoint(new_state):
        logger.info("从检查点恢复状态")
        return new_state
    else:
        return {
            "tree": tree, 
            "flat": flat, 
            "template_content": template_content,
            "current_index": 0,
            "completed_sections": ""
        }

def analyze_logic_node(state: State) -> Dict[str, Any]:
    """分析模板逻辑并确定层次化写作顺序"""
    logger.info("分析模板逻辑并制定层次化写作策略")
    
    if state.get("logical_analysis") and state.get("writing_order"):
        logger.info("逻辑分析已存在，跳过分析步骤")
        return {
            "logical_analysis": state["logical_analysis"],
            "writing_order": state["writing_order"]
        }
    
    domain_str = "、".join(CONFIG.get("领域", []))
    config_str = yaml.dump(CONFIG, allow_unicode=True, sort_keys=False)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS["system_prompts"]["analyze_logic"]),
        ("human", PROMPTS["human_prompts"]["analyze_logic"]),
    ])
    
    raw_response = safe_llm_invoke(prompt, {
        "template_content": state["template_content"][:2000],
        "领域": domain_str,
        "config_str": config_str[:1000]
    })
    
    logical_analysis = raw_response
    
    # 使用层次化方式确定写作顺序
    writing_order = determine_hierarchical_writing_order(state["tree"], state["flat"])
    
    logger.info(f"层次化写作顺序确定：共 {len(writing_order)} 个章节")
    
    result = {
        "logical_analysis": logical_analysis,
        "writing_order": writing_order
    }
    
    # 立即保存检查点
    save_checkpoint({**state, **result})
    
    return result

def sequential_write_node(state: State) -> Dict[str, Any]:
    """顺序写作单个章节"""
    if state["current_index"] >= len(state["writing_order"]):
        logger.info("所有章节已完成写作")
        return {"current_index": state["current_index"]}
    
    current_section_id = state["writing_order"][state["current_index"]]
    id_to_node = {n["id"]: n for n in state["flat"]}
    node = id_to_node[current_section_id]
    
    logger.info(f"正在写作章节: '{node['title']}' (进度: {state['current_index'] + 1}/{len(state['writing_order'])})")
    
    if node["content"].strip():
        logger.info(f"章节 '{node['title']}' 已存在内容，跳过")
        return {
            "tree": state["tree"],
            "completed_sections": state["completed_sections"],
            "current_index": state["current_index"] + 1
        }
    
    domain_str = "、".join(CONFIG.get("领域", []))
    config_str = yaml.dump(CONFIG, allow_unicode=True, sort_keys=False)
    min_words = int(CONFIG.get("单章字数", 0) or 0)
    length_clause = f"建议不少于{min_words}字" if min_words > 0 else "内容充实具体"
    
    # 构建章节级别信息
    level_info = "一级" if node["level"] == 1 else f"{node['level']}级"
    parent_context = parent_chain_titles(state["tree"], current_section_id)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS["system_prompts"]["sequential_write"]),
        ("human", PROMPTS["human_prompts"]["sequential_write"]),
    ])
    
    try:
        logger.info("开始LLM调用生成章节内容...")
        content = safe_llm_invoke(prompt, {
            "title": node["title"],
            "parent_chain": parent_context,
            "level": level_info,
            "instructions": (node["instructions"] or "").strip(),
            "placeholder": (node["placeholder"] or "").strip(),
            "logical_analysis": state["logical_analysis"][:500] if state["logical_analysis"] else "",
            "completed_sections": state["completed_sections"][-800:] if state["completed_sections"] else "",  
            "length_clause": length_clause,
            "config_str": config_str[:300],  
            "领域": domain_str,
        })
        
        logger.info(f"章节内容长度: {len(content)}")
        
        if not content or len(content) < 50:
            raise Exception(f"章节内容过短: {len(content)} 字符")
        
        node["content"] = content
        
        # 更新已完成章节
        section_markdown = f"## {node['title']}\n{content}\n\n"
        updated_completed = state["completed_sections"] + section_markdown
        
        logger.info(f"章节 '{node['title']}' 写作完成，内容长度: {len(content)} 字符")
        
        updated_state = {
            **state,
            "tree": state["tree"],
            "completed_sections": updated_completed,
            "current_index": state["current_index"] + 1
        }
        save_checkpoint(updated_state)
        
        return {
            "tree": state["tree"],
            "completed_sections": updated_completed,
            "current_index": state["current_index"] + 1
        }
        
    except Exception as e:
        logger.error(f"章节 '{node['title']}' 写作失败: {e}")
        save_checkpoint(state)
        raise e

def check_completion(state: State) -> str:
    """检查是否完成所有章节"""
    if state["current_index"] >= len(state["writing_order"]):
        return "finish"
    else:
        return "continue_writing"

def finish_node(state: State) -> Dict[str, Any]:
    """完成写作，保存文件"""
    logger.info("所有章节写作完成，正在保存文件")
    
    final_md = assemble_markdown(state["tree"])
    out_dir = PROJECT_ROOT / CONFIG["输出路径"]
    out_dir.mkdir(exist_ok=True)
    
    safe_project_name = re.sub(r'[\\/*?:"<>|]', "", CONFIG['项目名称'])
    output_file = out_dir / f"{safe_project_name}_最终申报书.md"
    
    output_file.write_text(final_md, encoding="utf-8")
    logger.info(f"申报书已保存至: {output_file.absolute()}")
    
    checkpoint_file = "checkpoint.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("检查点文件已清理")
    
    print("\n" + "="*50)
    print("生成完成！文件保存在:", output_file.absolute())
    print("="*50)
    
    return {}

# ---------------------------- 图编排 ----------------------------
workflow = StateGraph(State)
workflow.add_node("init", init_node)
workflow.add_node("analyze_logic", analyze_logic_node)
workflow.add_node("sequential_write", sequential_write_node)
workflow.add_node("finish", finish_node)

workflow.set_entry_point("init")
workflow.add_edge("init", "analyze_logic")
workflow.add_edge("analyze_logic", "sequential_write")
workflow.add_conditional_edges("sequential_write", check_completion, {
    "continue_writing": "sequential_write",
    "finish": "finish"
})
workflow.add_edge("finish", END)

app = workflow.compile()

# ---------------------------- 主入口 ----------------------------
if __name__ == "__main__":
    try:
        # 添加psutil依赖检查
        try:
            import psutil
        except ImportError:
            logger.warning("psutil未安装，无法监控系统资源。运行: pip install psutil")
        
        logger.info(f"开始生成《{CONFIG['项目名称']}》申报书")
        logger.info(f"模板: {CONFIG['格式']}，模型: {CONFIG['模型']}")
        
        initial_state: State = {
            "tree": [], "flat": [], "template_content": "",
            "logical_analysis": "", "writing_order": [],
            "current_index": 0, "completed_sections": ""
        }
        
        recursion_limit = int(os.getenv("GRAPH_RECURSION_LIMIT", "2000"))
        logger.info(f"设置 LangGraph recursion_limit = {recursion_limit}")
        
        app.invoke(initial_state, config={"recursion_limit": recursion_limit})
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.info("可以重新运行程序，将自动从检查点恢复")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("程序结束")