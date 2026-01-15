# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/workspace/MLILAB-GRPO/data/DAPO")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "BytedTsinghua-SIA/DAPO-Math-17k"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)


    train_dataset = dataset["train"]
    # train_dataset = dataset["train"].select(range(7431))
    # test_dataset = dataset["test"].select(range(33))

    original_opening_instruction = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    original_closing_instruction = '\n\nRemember to put your answer on its own line after \"Answer:\".'
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}.\n\n"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt")[0]["content"]
            if original_opening_instruction in question and original_closing_instruction in question:
                question = question.replace(original_opening_instruction, "")
                question = question.replace(original_closing_instruction, "")
            else:
                assert False, "Instruction following is not supported"
            
            question = instruction_following + question

            solution = example.pop("reward_model")["ground_truth"]
            answer = solution
            
            if idx < 10:
                print("===================")
                print(question)
                # print("===================")
                # print(solution)
                print("===================")
                print(answer)
            
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    print(f"Before deduplication: {len(train_dataset)}")
    
    # Deduplicate based on question (prompt content)
    def get_question(example):
        return example["prompt"][0]["content"]
    
    seen_questions = set()
    unique_indices = []
    for idx, example in enumerate(train_dataset):
        question = get_question(example)
        if question not in seen_questions:
            seen_questions.add(question)
            unique_indices.append(idx)
    
    train_dataset = train_dataset.select(unique_indices)
    print(f"After deduplication: {len(train_dataset)}")
    
    train_dataset = train_dataset.select(range(len(train_dataset) - len(train_dataset) % 128 + 8))
    print(f"After filtering to 128-batch: {len(train_dataset)}")
    
    # assert len(train_dataset) % 128 == 0, "The number of examples must be divisible by 128"
    
    # test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    # print(len(train_dataset))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f"train_{len(train_dataset)}.parquet"))
    # test_dataset.to_parquet(os.path.join(local_dir, "test_example_33.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
