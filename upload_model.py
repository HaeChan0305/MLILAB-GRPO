import os
import re
import glob
import shutil
import subprocess
from huggingface_hub import HfApi, upload_folder, snapshot_download


# Hugging Face에 로그인된 사용자 토큰 확인
api = HfApi()


def upload_model(experiment_name, repo_id, token, step, inference_only=True):
    """
    모델 체크포인트를 Hugging Face에 업로드
    
    Args:
        experiment_name: 실험 이름
        repo_id: HuggingFace 저장소 ID
        token: HuggingFace 토큰
        step: 업로드할 step
        inference_only: True면 inference에 필요한 파일만 업로드 (기본값 True)
                       - huggingface/ 폴더
                       - model_world_size_*.pt 파일
                       False면 모든 파일 업로드 (optim, extra_state 등 포함)
    """
    # Inference에 불필요한 파일 패턴 (training resume에만 필요)
    ignore_patterns = None
    if inference_only:
        ignore_patterns = [
            "actor/optim_world_size_*",       # 옵티마이저 상태
            "actor/extra_state_world_size_*", # 학습 스텝, 스케줄러 상태
            "fsdp_config.json",         # FSDP 설정
            "data.pt",                  # 기타 training 데이터
        ]
        print(f"📦 Inference-only mode: excluding {ignore_patterns}")
    
    # 리포지토리 생성 (이미 존재하면 무시됨)
    api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

    # 모델 폴더 업로드
    folder_path = f"./models/{experiment_name}/global_step_{step}"

    # chmod 777 적용
    print(f"Applying chmod 777 to {folder_path}")
    subprocess.run(["sudo", "chmod", "-R", "777", folder_path], check=True)

    print(f"Uploading step {step}")
    upload_folder(
        folder_path=folder_path,      # 업로드할 로컬 모델 폴더 경로
        repo_id=repo_id,             # 업로드할 Hugging Face repo 이름
        repo_type="model",           # model / dataset / space 중 선택
        commit_message=f"Upload global_step_{step}",
        token=token,
        ignore_patterns=ignore_patterns,
    )
    print(f"Uploaded step {step} (inference-only: {inference_only})")




def get_step_from_commit_message(message: str) -> int | None:
    """
    커밋 메시지에서 step 번호 추출
    지원하는 형식:
    - 'Upload step 105' -> 105
    - 'Upload global_step_105' -> 105
    - 'Upload global_step_105 (inference-only)' -> 105
    """
    patterns = [
        r'Upload global_step_(\d+)',  # Upload global_step_105
        r'Upload step (\d+)',          # Upload step 105
        r'global_step_(\d+)',          # global_step_105
        r'step[_\-\s]?(\d+)',          # step_105, step-105, step 105
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def list_checkpoint_commits(repo_id: str, token: str = None, max_commits: int = 200):
    """저장소의 모든 체크포인트 커밋 목록 조회 (Git clone 방식)"""
    import tempfile
    import shutil
    
    all_commits = []
    
    # Git을 사용하여 커밋 히스토리 조회
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_url = f"https://huggingface.co/{repo_id}"
        if token:
            # 토큰이 있으면 인증 URL 사용
            repo_url = f"https://USER:{token}@huggingface.co/{repo_id}"
        
        clone_dir = os.path.join(temp_dir, "repo")
        
        try:
            # Shallow clone with only git history (no files)
            print(f"📥 Fetching commit history from {repo_id}...")
            result = subprocess.run(
                ["git", "clone", "--bare", "--filter=blob:none", repo_url, clone_dir],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"⚠️  Git clone failed: {result.stderr}")
                # Fallback to huggingface_hub
                return _list_commits_fallback(repo_id, token, max_commits)
            
            # Git log로 커밋 히스토리 조회
            result = subprocess.run(
                ["git", "-C", clone_dir, "log", "--format=%H|%s|%aI", f"-{max_commits}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"⚠️  Git log failed: {result.stderr}")
                return _list_commits_fallback(repo_id, token, max_commits)
            
            # 커밋 파싱
            class CommitInfo:
                def __init__(self, commit_id, title, created_at):
                    self.commit_id = commit_id
                    self.title = title
                    self.created_at = created_at
            
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    commit_id = parts[0]
                    title = parts[1]
                    created_at = parts[2] if len(parts) > 2 else ""
                    all_commits.append(CommitInfo(commit_id, title, created_at))
            
        except subprocess.TimeoutExpired:
            print("⚠️  Git clone timeout, using fallback...")
            return _list_commits_fallback(repo_id, token, max_commits)
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return _list_commits_fallback(repo_id, token, max_commits)
    
    print(f"📋 Total commits fetched: {len(all_commits)}")
    
    checkpoint_commits = []
    for commit in all_commits:
        step = get_step_from_commit_message(commit.title)
        checkpoint_commits.append({
            'commit_id': commit.commit_id,
            'title': commit.title,
            'step': step,
            'created_at': commit.created_at
        })
    
    return checkpoint_commits


def _list_commits_fallback(repo_id: str, token: str = None, max_commits: int = 200):
    """Fallback: huggingface_hub 라이브러리 사용"""
    from huggingface_hub import list_repo_commits
    
    all_commits = []
    
    try:
        for commit in list_repo_commits(repo_id, token=token, repo_type="model"):
            all_commits.append(commit)
            if len(all_commits) >= max_commits:
                break
    except Exception as e:
        print(f"⚠️  Fallback also failed: {e}")
    
    print(f"📋 Total commits fetched (fallback): {len(all_commits)}")
    
    checkpoint_commits = []
    for commit in all_commits:
        step = get_step_from_commit_message(commit.title)
        checkpoint_commits.append({
            'commit_id': commit.commit_id,
            'title': commit.title,
            'step': step,
            'created_at': commit.created_at
        })
    
    return checkpoint_commits

def download_model(experiment_name, repo_id, step, token, convert_to_safetensors=True):
    # list_checkpoint_commits를 사용하여 모든 커밋 조회 (페이지네이션 처리됨)
    commits = list_checkpoint_commits(repo_id, token)
    commit_message = None
    commit_id = None
    
    for commit in commits:
        if commit['step'] == step:
            commit_message = commit['title']
            commit_id = commit['commit_id']
            break
    
    if commit_message is None:
        available_steps = sorted([c['step'] for c in commits if c['step'] is not None])
        raise ValueError(f"Step {step} not found in repo {repo_id}. Available steps: {available_steps}")
    
    print(f"Commit message: {commit_message}")
    print(f"Commit id: {commit_id}")
    
    
    folder_path = f"./models/{experiment_name}/global_step_{step}"
    os.makedirs(folder_path, exist_ok=True)
    
    # repo 이름과 commit을 주면 huggingface hub에서 해당 commit의 모델을 다운로드 받는다.
    snapshot_download(
        repo_id=repo_id,
        revision=commit_id,
        token=token,
        local_dir=folder_path
    )
    
    print(f"✅ Model successfully downloaded to {folder_path}")
    
    # Convert to safetensors format
    if convert_to_safetensors:
        actor_path = os.path.join(folder_path, "actor")
        convert_to_safetensors_format(actor_path)
    
    return folder_path


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


def move_id2score_json_files(experiment_name):
    base_dir = f"/wbl/post-training/haechan_workspace/MLILAB-GRPO/models/{experiment_name}"

    # id2score_step_*.json 파일들 찾기
    json_files = glob.glob(os.path.join(base_dir, "id2score_step_*.json"))

    moved_count = 0
    skipped_count = 0

    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        
        # step 번호 추출: id2score_step_100.json -> 100
        step = filename.replace("id2score_step_", "").replace(".json", "")
        
        # 대상 폴더
        target_dir = os.path.join(base_dir, f"global_step_{step}")
        
        if os.path.isdir(target_dir):
            # 폴더가 존재하면 이동
            target_path = os.path.join(target_dir, filename)
            shutil.move(json_file, target_path)
            print(f"✓ Moved: {filename} -> global_step_{step}/")
            moved_count += 1
        else:
            # 폴더가 없으면 그대로 유지
            print(f"⏭ Skipped: {filename} (global_step_{step}/ not found)")
            skipped_count += 1

    print(f"\n📊 Summary: {moved_count} moved, {skipped_count} skipped")


if __name__ == "__main__":
    token = None
    
    # ============================================================
    # 예시 1: 모델 다운로드
    # ============================================================
    # experiment_name = "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-again"
    # repo_id = f"HaeChan0305/{experiment_name}"
    # download_model(experiment_name, repo_id, 405, token)
    
    # ============================================================
    # 예시 2: 체크포인트 정리 미리보기 (삭제 없음)
    # ============================================================
    
    # experiment_names = [
    #     # ("qwen3-grpohistbeta-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6", 200),
    #     # ("qwen3-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6", 345),
    #     # ("qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-again", 595),
    #     # ("qwen3-grpo-paper-dapo17k-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6-again", 635),
    #     # ("qwen3-grpo-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6", 345),
    # ]    
    
    # for experiment_name, max_step in experiment_names:
    #     repo_id = f"HaeChan0305/{experiment_name}"
        
    #     for step in range(5, max_step + 1, 5):
    #         try:
    #             download_model(experiment_name, repo_id, step, token, convert_to_safetensors=False)
    #         except Exception as e:
    #             print(f"Error downloading model for step {step}: {e}")
    #             continue

    experiment_names = [
        "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-again", # 5, 595
        "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_75", # 5, 540
        "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_5", # 5, 540
        "qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_25", # 5, 540
        "qwen3-grpo-paper-dapo17k-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6-again", # 5, 635
    ]    

    for experiment_name in experiment_names:
        repo_id = f"HaeChan0305/{experiment_name}"
        
        inference_only_interval = 135 # 100 step마다 모든 파일 업로드
        for step in range(5, 540 + 1, 5):
            try:
                if step % inference_only_interval == 0:
                    upload_model(experiment_name, repo_id, token, step, inference_only=False)
                else:
                    upload_model(experiment_name, repo_id, token, step, inference_only=True)
            except Exception as e:
                print(f"Error uploading model for step {step}: {e}")
                continue

        