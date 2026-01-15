#!/usr/bin/env python
"""
Evaluation script using vLLM for math datasets.
Computes pass@1 and pass@k metrics.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Optional, List, Dict, Any

import datasets
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Import grading functions from math_haechan (same logic, copied here to be self-contained)
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser


# ============================================================
# Grading functions (from math_haechan.py)
# ============================================================

def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string

    def _remove_right_units(string):
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    if expr is None:
        return None
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if len(given_normalized) == 0:
        return False
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break
    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) \
        or grade_answer_sympy(given_answer, ground_truth)


def compute_score(solution_str, ground_truth):
    """Compute score for a single solution against ground truth."""
    retval = 0.0
    try:
        if grade_answer_verl(solution_str, ground_truth):
            retval = 1.0
    except Exception as e:
        print(f"Error in grading: {e}")
    return retval


# ============================================================
# Dataset loading and preprocessing
# ============================================================

def load_dataset_from_source(data_source: str, split: str = "test"):
    """Load dataset from HuggingFace or local source."""
    if data_source == "opencompass/AIME2025":
        dataset1 = datasets.load_dataset(data_source, "AIME2025-I", trust_remote_code=True)['test']
        dataset2 = datasets.load_dataset(data_source, "AIME2025-II", trust_remote_code=True)['test']
        dataset = datasets.concatenate_datasets([dataset1, dataset2])
    else:
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)[split]
    return dataset


def preprocess_dataset(
    dataset,
    instruction: str = "Let's think step by step and output the final answer within \\boxed{}.\n\n"
):
    """
    Preprocess dataset to the expected format (similar to dataset_aime2025.py).
    
    Output format:
    {
        "data_source": str,
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": split, "index": idx},
    }
    """
    processed_data = []
    
    for idx, example in enumerate(dataset):
        # Handle different dataset formats
        question = example.get("question", example.get("problem", ""))
        answer = example.get("answer", example.get("solution", ""))
        
        question_with_instruction = instruction + question
        
        data = {
            "data_source": "evaluation",
            "prompt": [{"role": "user", "content": question_with_instruction}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "test", "index": idx},
        }
        processed_data.append(data)
    
    return processed_data


def load_preprocessed_parquet(parquet_path: str) -> List[Dict]:
    """Load already preprocessed parquet file."""
    dataset = datasets.load_dataset("parquet", data_files=parquet_path)["train"]
    processed_data = []
    for idx, example in enumerate(dataset):
        processed_data.append({
            "prompt": example["prompt"],
            "reward_model": example["reward_model"],
            "extra_info": example.get("extra_info", {"index": idx}),
            "data_source": example.get("data_source", "parquet"),
        })
    return processed_data


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt_for_model(prompt_messages: List[Dict], tokenizer) -> str:
    """Format prompt messages for the model using chat template if available."""
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            formatted = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
    
    # Fallback: simple concatenation
    return "\n".join([msg["content"] for msg in prompt_messages])


# ============================================================
# Metrics computation
# ============================================================

def compute_pass_at_k(results: List[List[Dict]], k: int) -> float:
    """
    Compute pass@k metric.
    For each problem, check if any of the first k samples is correct.
    
    Args:
        results: List of lists, where each inner list contains sample results for one problem
        k: Number of samples to consider
    
    Returns:
        Pass@k accuracy
    """
    correct = 0
    total = len(results)
    
    for problem_results in results:
        samples = problem_results[:k]
        if any(sample["score"] > 0 for sample in samples):
            correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_majority_vote_at_k(results: List[List[Dict]], k: int) -> float:
    """
    Compute majority vote accuracy at k.
    For each problem, take the majority answer among k samples.
    
    Args:
        results: List of lists, where each inner list contains sample results for one problem
        k: Number of samples to consider
    
    Returns:
        Majority vote accuracy
    """
    correct = 0
    total = len(results)
    
    for problem_results in results:
        samples = problem_results[:k]
        # Count correct vs incorrect
        num_correct = sum(1 for s in samples if s["score"] > 0)
        num_incorrect = len(samples) - num_correct
        
        # Majority vote
        if num_correct > num_incorrect:
            correct += 1
    
    return correct / total if total > 0 else 0.0


# ============================================================
# Main evaluation function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate math models using vLLM")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model (local path or HuggingFace model name)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                        help="Tensor parallel size for distributed inference")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, 
                        help="GPU memory utilization (0.0 to 1.0)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Data type for model weights")
    
    # Dataset arguments
    parser.add_argument("--data_source", type=str, default="opencompass/AIME2025", 
                        help="Dataset source (HuggingFace dataset name)")
    parser.add_argument("--data_path", type=str, default=None, 
                        help="Local parquet file path (overrides data_source if provided)")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use")
    
    # Sampling arguments
    parser.add_argument("--n_samples", type=int, default=8, 
                        help="Number of samples to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=0.95, 
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1, 
                        help="Top-k sampling parameter (-1 to disable)")
    parser.add_argument("--max_tokens", type=int, default=4096, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, 
                        help="Repetition penalty (1.0 for no penalty)")
    parser.add_argument("--presence_penalty", type=float, default=0.0,
                        help="Presence penalty")
    parser.add_argument("--frequency_penalty", type=float, default=0.0,
                        help="Frequency penalty")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./eval_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--output_prefix", type=str, default="eval", 
                        help="Prefix for output files")
    
    # Instruction
    parser.add_argument("--instruction", type=str, 
                        default="Let's think step by step and output the final answer within \\boxed{}.\n\n",
                        help="Instruction prefix to prepend to each prompt")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for generation (None for all at once)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, f"{args.output_prefix}_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model from: {args.model_path}")
    print(f"{'='*60}")
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype=args.dtype,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading dataset...")
    print(f"{'='*60}")
    
    if args.data_path:
        print(f"Loading from local parquet: {args.data_path}")
        processed_data = load_preprocessed_parquet(args.data_path)
    else:
        print(f"Loading from HuggingFace: {args.data_source}")
        raw_dataset = load_dataset_from_source(args.data_source, args.split)
        processed_data = preprocess_dataset(raw_dataset, args.instruction)
    
    print(f"Loaded {len(processed_data)} examples")
    
    # Prepare prompts
    prompts = []
    for item in processed_data:
        formatted_prompt = format_prompt_for_model(item["prompt"], tokenizer)
        prompts.append(formatted_prompt)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
    )
    
    print(f"\n{'='*60}")
    print("Sampling Parameters:")
    print(f"{'='*60}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  max_tokens: {args.max_tokens}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print(f"  presence_penalty: {args.presence_penalty}")
    print(f"  frequency_penalty: {args.frequency_penalty}")
    print(f"{'='*60}")
    
    # Generate responses
    print(f"\nGenerating {args.n_samples} responses for {len(prompts)} prompts...")
    
    if args.batch_size is not None and args.batch_size < len(prompts):
        # Batch generation
        all_outputs = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(batch_outputs)
        outputs = all_outputs
    else:
        # Generate all at once
        outputs = llm.generate(prompts, sampling_params)
    
    print("Generation complete!")
    
    # Process results
    print("\nScoring responses...")
    all_results = []  # List of lists, each inner list contains results for one problem
    all_generations = []  # For saving all generations
    
    for idx, (item, output) in enumerate(tqdm(zip(processed_data, outputs), 
                                               total=len(processed_data), 
                                               desc="Scoring")):
        ground_truth = item["reward_model"]["ground_truth"]
        prompt_text = prompts[idx]
        original_prompt = item["prompt"][0]["content"] if item["prompt"] else ""
        
        problem_results = []
        problem_generations = {
            "index": item.get("extra_info", {}).get("index", idx),
            "prompt": original_prompt,
            "formatted_prompt": prompt_text,
            "ground_truth": ground_truth,
            "data_source": item.get("data_source", "unknown"),
            "generations": []
        }
        
        for sample_idx, completion in enumerate(output.outputs):
            response_text = completion.text
            score = compute_score(response_text, ground_truth)
            extracted_answer = extract_answer(response_text)
            
            problem_results.append({
                "response": response_text,
                "score": score,
                "extracted_answer": extracted_answer,
            })
            
            problem_generations["generations"].append({
                "sample_idx": sample_idx,
                "response": response_text,
                "extracted_answer": extracted_answer,
                "score": score,
            })
        
        all_results.append(problem_results)
        all_generations.append(problem_generations)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    pass_at_1 = compute_pass_at_k(all_results, 1)
    pass_at_8 = compute_pass_at_k(all_results, min(8, args.n_samples))
    maj_at_8 = compute_majority_vote_at_k(all_results, min(8, args.n_samples))
    
    # Compute for all k values up to n_samples
    pass_at_k_results = {}
    for k in range(1, args.n_samples + 1):
        pass_at_k_results[f"pass@{k}"] = compute_pass_at_k(all_results, k)
        pass_at_k_results[f"maj@{k}"] = compute_majority_vote_at_k(all_results, k)
    
    # Compute average accuracy across all samples
    total_samples = sum(len(pr) for pr in all_results)
    total_correct = sum(sum(1 for s in pr if s["score"] > 0) for pr in all_results)
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Prepare metrics dictionary
    metrics = {
        "model_path": args.model_path,
        "data_source": args.data_source if not args.data_path else args.data_path,
        "num_problems": len(all_results),
        "n_samples": args.n_samples,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
        "pass@1": pass_at_1,
        "pass@8": pass_at_8,
        "maj@8": maj_at_8,
        "avg_accuracy": avg_accuracy,
        "all_pass_at_k": pass_at_k_results,
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_source if not args.data_path else args.data_path}")
    print(f"Number of problems: {len(all_results)}")
    print(f"Samples per problem: {args.n_samples}")
    print(f"{'='*60}")
    print(f"Pass@1:  {pass_at_1:.4f} ({int(pass_at_1 * len(all_results))}/{len(all_results)})")
    print(f"Pass@8:  {pass_at_8:.4f} ({int(pass_at_8 * len(all_results))}/{len(all_results)})")
    print(f"Maj@8:   {maj_at_8:.4f} ({int(maj_at_8 * len(all_results))}/{len(all_results)})")
    print(f"Avg Acc: {avg_accuracy:.4f}")
    print(f"{'='*60}")
    
    # Print all pass@k values
    print("\nDetailed pass@k results:")
    for k in range(1, min(args.n_samples + 1, 17)):  # Show up to pass@16
        if f"pass@{k}" in pass_at_k_results:
            print(f"  pass@{k}: {pass_at_k_results[f'pass@{k}']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{args.output_prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save generations
    generations_path = os.path.join(args.output_dir, f"{args.output_prefix}_generations.jsonl")
    with open(generations_path, "w") as f:
        for gen in all_generations:
            f.write(json.dumps(gen, ensure_ascii=False) + "\n")
    print(f"Generations saved to: {generations_path}")
    
    # Also save a summary file with just the key metrics
    summary_path = os.path.join(args.output_dir, f"{args.output_prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.data_source if not args.data_path else args.data_path}\n")
        f.write(f"Number of problems: {len(all_results)}\n")
        f.write(f"Samples per problem: {args.n_samples}\n")
        f.write(f"\n")
        f.write(f"Pass@1: {pass_at_1:.4f}\n")
        f.write(f"Pass@8: {pass_at_8:.4f}\n")
        f.write(f"Maj@8: {maj_at_8:.4f}\n")
        f.write(f"Avg Accuracy: {avg_accuracy:.4f}\n")
    print(f"Summary saved to: {summary_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

