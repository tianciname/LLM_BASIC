from .arithmetic import ArithmeticTask
from .commonsense import CommonsenseTask

# 任务注册表
TASK_REGISTRY = {
    "arithmetic": ArithmeticTask,
    "commonsense": CommonsenseTask,
}

def get_task(name):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {name}")
    return TASK_REGISTRY[name]()