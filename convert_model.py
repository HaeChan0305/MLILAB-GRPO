import os
import glob
import argparse


def convert_to_safetensors_format(actor_path):
    """
    Convert FSDP sharded checkpoint (.pt files) to HuggingFace safetensors format.
    Uses veRL's model_merger for proper handling of DTensor and sharded checkpoints.
    
    Expected structure:
    - actor_path/model_world_size_X_rank_Y.pt (FSDP sharded model weights)
    - actor_path/huggingface/ (tokenizer and config files)
    
    Output:
    - actor_path/huggingface/*.safetensors (converted model weights)
    """
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    
    huggingface_path = os.path.join(actor_path, "huggingface")
    
    # Check if already converted
    safetensor_files = glob.glob(os.path.join(huggingface_path, "*.safetensors"))
    if safetensor_files:
        print(f"Model at {huggingface_path} is already in safetensors format.")
        return huggingface_path
    
    # Check for FSDP sharded checkpoint files
    pt_files = sorted(glob.glob(os.path.join(actor_path, "model_world_size_*_rank_*.pt")))
    
    if not pt_files:
        print(f"No FSDP checkpoint files found at {actor_path}. Skipping conversion.")
        return None
    
    print(f"Found {len(pt_files)} FSDP sharded checkpoint files.")
    print(f"Converting FSDP checkpoint to safetensors format using veRL model_merger...")
    
    # Temporarily allow loading old pickle format (PyTorch 2.6+ compatibility)
    old_env = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
    
    try:
        # Use veRL's FSDPModelMerger
        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=actor_path,
            target_dir=huggingface_path,
            hf_model_config_path=huggingface_path,  # tokenizer/config files location
            trust_remote_code=True,
        )
        
        merger = FSDPModelMerger(config)
        merger.merge_and_save()
        merger.cleanup()
        
        print(f"✅ Model successfully converted to safetensors format: {huggingface_path}")
        return huggingface_path
    
    finally:
        # Restore original environment
        if old_env is None:
            os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = old_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace safetensors format")
    parser.add_argument(
        "--actor_path",
        type=str,
        required=True,
        help="Path to actor folder containing FSDP sharded checkpoints (model_world_size_*_rank_*.pt)"
    )
    args = parser.parse_args()
    
    convert_to_safetensors_format(args.actor_path)
