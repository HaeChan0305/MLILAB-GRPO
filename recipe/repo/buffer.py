import os 
import pickle
from verl import DataProto
from tensordict import TensorDict


class Buffer:
    def __init__(self, default_local_dir: str, global_step: int, save_freq: int):
        self.default_local_dir = default_local_dir
        self.global_step = global_step
        self.save_freq = save_freq
        self.buffer_dir = os.path.join(self.default_local_dir, "buffer")

        if not os.path.exists(self.buffer_dir):
            assert self.global_step == 0, f"Buffer directory does not exist, but global_step is {self.global_step}"
            os.makedirs(self.buffer_dir)
            self.buffer = []
        elif not os.listdir(self.buffer_dir):
            assert self.global_step == 0, f"Buffer directory is empty, but global_step is {self.global_step}"
            self.buffer = []
        else:
            self._load(self)
    
    def update(self, epoch: int, step: int, batch: DataProto):
        self.global_step = step
        self.buffer.append({"epoch": epoch, "step": step, "batch": batch})
    
    def _get_buffer_path(self):
        return os.path.join(self.buffer_dir, f"buffer_step_{self.global_step}.pkl")
    
    def _load(self):
        buffer_path = self._get_buffer_path()
        with open(buffer_path, "rb") as f:
            self.buffer = pickle.load(f)
            print(f"Loaded buffer : {buffer_path}")
    
    def save(self):
        assert self.global_step > 0 and self.global_step % self.save_freq == 0, f"Can only save buffer at every {self.save_freq} steps. Current step: {self.global_step}"
        
        buffer_path = self._get_buffer_path()
        if os.path.exists(buffer_path):
            print(f"Warning: Overwriting existing buffer file at {buffer_path}")

        with open(buffer_path, "wb") as f:
            pickle.dump(self.buffer, f)
            print(f"Saved buffer : {buffer_path}")

        # 2 step에 35MB


    def make_off_policy_batch(self, batch: DataProto) -> DataProto:
        off_policy_batch = DataProto()
        for query_idx in set(batch.non_tensor_batch['index']):
            for entry in self.buffer:
                subbatch = self._extract_subbatch(entry['batch'], query_idx)
                if subbatch is None:
                    continue
                else:
                    if not off_policy_batch.batch:
                        off_policy_batch.batch = subbatch.batch
                        off_policy_batch.non_tensor_batch = subbatch.non_tensor_batch
                        off_policy_batch.meta_info = subbatch.meta_info
                    else:
                        off_policy_batch.batch = TensorDict.cat([off_policy_batch.batch, subbatch.batch], dim=0)
                        for k in off_policy_batch.non_tensor_batch.keys():
                            off_policy_batch.non_tensor_batch[k].extend(subbatch.non_tensor_batch[k])
                        for k in off_policy_batch.meta_info.keys():
                            if k == 'global_token_num':
                                off_policy_batch.meta_info[k].extend(subbatch.meta_info[k])
                            else:
                                if off_policy_batch.meta_info[k] != subbatch.meta_info[k]:
                                    print(f"Warning: meta_info key '{k}' has different values across subbatches.")
        if off_policy_batch.batch:
            return off_policy_batch
    
    def _extract_subbatch(self, batch: DataProto, query_idx: int) -> DataProto:
        indices = []
        for i, idx in enumerate(batch.non_tensor_batch['index']):
            if idx == query_idx:
                indices.append(i)
            
        if indices == []:
            return None
        
        subbatch = DataProto()
        subbatch.batch = TensorDict(
                                    {k: v[indices] for k, v in batch.batch.items()},
                                    batch_size=len(indices),
                                    )
        subbatch.non_tensor_batch = {k: [v[i] for i in indices] for k, v in batch.non_tensor_batch.items()}
        subbatch.meta_info = {k: [v[i] for i in indices] if k == 'global_token_num' else v for k, v in batch.meta_info.items()}
        return subbatch

