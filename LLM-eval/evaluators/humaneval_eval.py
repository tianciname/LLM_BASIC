# evaluators/humaneval_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re
import sys
import io
import signal
import contextlib
from config import CONFIG
def evaluate(pipeline, max_samples):
    """
    评估模型的代码生成能力 (HumanEval)
    参照 MMLU 评估器的结构，使用对话模板、放宽长度限制、剔除思维链草稿。
    """
    # 加载 HumanEval 数据集（测试集）
    ds = load_dataset('openai_humaneval', split='test')
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))
    
    correct = 0
    total = 0
    
    for idx, item in enumerate(tqdm(ds, desc="HumanEval")):
        prompt_text = item['prompt']          # 原始问题提示，包含函数签名和文档字符串
        test_code = item['test']              # 单元测试代码
        entry_point = item['entry_point']     # 要测试的函数名
        
        # 构造用户消息（遵循 Chat 模版）
        user_content = f"Please complete the following Python function. Return ONLY the function definition and its body, without any extra explanation or test code.\n\n{prompt_text}"
        messages = [{"role": "user", "content": user_content}]
        
        # 生成代码，允许完整输出（最大1024 token）
        out = pipeline(messages, max_new_tokens=CONFIG["max_new_token_length"], max_length=None, do_sample=False)[0]['generated_text']
        raw_response = out[-1]['content'].strip()
        
        # 剥离思维链草稿 <think>...</think>
        if "</think>" in raw_response:
            response = raw_response.split("</think>")[-1].strip()
        else:
            response = raw_response
        
        # 从生成内容中提取完整的函数定义
        generated_code = extract_function(response, entry_point, prompt_text)
        
        # 运行单元测试判定是否正确
        is_correct = check_correctness(generated_code, test_code, entry_point)
        
        if is_correct:
            correct += 1
        total += 1
        
        # 可选：打印错误样例（用于调试）
        # if not is_correct:
        #     print(f"\n--- Failed sample {idx} ---\n{generated_code}\n---\n")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy


def extract_function(generated: str, entry_point: str, original_prompt: str) -> str:
    """
    从模型生成的完整回答中抽取出目标函数的代码。
    策略：
        1. 如果生成内容中包含与入口点匹配的 'def {entry_point}('，则从该处截取到函数结束（通过缩进检测）。
        2. 否则回退到原始 prompt + 模型生成的内容拼接作为候选。
    """
    # 查找第一个匹配的函数定义
    pattern = rf'(def {re.escape(entry_point)}\s*\([^)]*\)\s*->?\s*[^:]*:\s*\n(?:[ \t]+.*\n?)*)'
    match = re.search(pattern, generated, re.MULTILINE)
    if match:
        func_code = match.group(1).rstrip()
        # 确保函数体不为空（至少包含一个pass或实际代码）
        if func_code.strip().endswith(':'):
            # 未提取到函数体，尝试补一个pass
            func_code += "\n    pass"
        return func_code
    else:
        # 如果没有找到完整函数，尝试用原始prompt + 生成内容组合成可执行代码
        # 很多模型会直接延续prompt内容
        combined = original_prompt + "\n" + generated
        # 再次尝试提取
        match2 = re.search(pattern, combined, re.MULTILINE)
        if match2:
            return match2.group(1)
        else:
            # 实在提取不到，返回生成内容本身（让测试时出错）
            return generated


def check_correctness(generated_code: str, test_code: str, entry_point: str) -> bool:
    """
    动态执行生成的函数定义和测试代码，判断是否正确。
    使用超时和重定向输出来防止阻塞和污染。
    """
    # 构造完整的执行环境：先定义函数，再运行测试
    full_code = f"{generated_code}\n\n{test_code}\n\ncheck({entry_point})"
    
    # 设置超时（3秒）和捕获异常
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    # 保存原有信号处理（仅对 SIGALRM 有效，Windows 需用其他方式，这里假设 Unix 环境）
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)  # 3秒超时
    
    stdout = io.StringIO()
    stderr = io.StringIO()
    result = False
    
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            # 在单独的字典中执行代码，避免污染全局
            exec_globals = {}
            exec(full_code, exec_globals)
            # 如果执行完成没有抛出异常，测试函数 check 内部会断言，若断言失败会抛出 AssertionError
            result = True  # 没有异常即为通过
    except AssertionError:
        # 测试失败（check 函数断言未通过）
        result = False
    except Exception as e:
        # 其他错误（语法错误、超时等）
        result = False
    finally:
        signal.alarm(0)  # 取消超时
        signal.signal(signal.SIGALRM, old_handler)
    
    return result