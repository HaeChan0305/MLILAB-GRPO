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
    parser.add_argument("--local_dir", default="/workspace/GRPO/data/AIME2024")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    test_dataset = datasets.load_dataset(data_source)['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}.\n\n"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("Problem")

            question = instruction_following + question

            answer = str(example.pop("Answer"))
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    test_dataset = test_dataset.remove_columns(['ID', 'Solution'])

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
