"""Main experiment entry point.

Usage:
    python scripts/run_experiment.py --model gpt2 --task ioi --method acdc
"""
import argparse
import os
import sys

# Make sure the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuits.model import load_model
from circuits.tasks.ioi import IOITask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gpt2")
    parser.add_argument("--task",   default="ioi")
    parser.add_argument("--method", default="acdc")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"=== Experiment ===")
    print(f"Model:  {args.model}")
    print(f"Task:   {args.task}")
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")

    # Load model
    model = load_model(args.model, device=args.device)

    # Load task
    task = IOITask()
    data = task.get_data(n_examples=3)
    print(f"\nTask data loaded: {len(data.clean_prompts)} examples")
    print(f"Example clean prompt:     '{data.clean_prompts[0]}'")
    print(f"Example corrupted prompt: '{data.corrupted_prompts[0]}'")

    # Quick sanity check: run model on clean prompt
    tokens = model.to_tokens(data.clean_prompts[0])
    logits = model(tokens)
    top_token = logits[0, -1].argmax()
    print(f"\nModel prediction for clean prompt: '{model.to_string(top_token)}'")
    print(f"Expected answer: '{data.answer_tokens[0]}'")

    print("\n✓ Environment verified — model loads and runs correctly")
    print("  Next: implement scoring methods in circuits/scoring/")


if __name__ == "__main__":
    main()
