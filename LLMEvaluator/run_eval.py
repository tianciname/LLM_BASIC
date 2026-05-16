import argparse
from evaluator import LLMEvaluator
from tasks.glue_task import MNLITask
from tasks.squad_task import SQuADTask
from tasks.hellaswag_task import HellaSwagTask
from tasks.gsm8k_task import GSM8KTask
from tasks.humaneval_task import HumanEvalTask
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--tasks", type=str, nargs="+", default=["mnli", "squad", "hellaswag", "gsm8k", "humaneval"])
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per task")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    evaluator = LLMEvaluator(args.model, device=args.device)

    task_map = {
        "mnli": MNLITask(num_samples=args.num_samples),
        "squad": SQuADTask(num_samples=args.num_samples),
        "hellaswag": HellaSwagTask(num_samples=args.num_samples),
        "gsm8k": GSM8KTask(num_samples=args.num_samples),
        "humaneval": HumanEvalTask(num_samples=args.num_samples)
    }

    selected_tasks = [task_map[t] for t in args.tasks if t in task_map]
    results = evaluator.run_all(selected_tasks, num_samples=args.num_samples, batch_size=args.batch_size)

    for task, metrics in results.items():
        print(f"{task}: {metrics}")

    evaluator.save_results(args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()