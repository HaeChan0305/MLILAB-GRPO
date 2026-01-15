"""
W&B 로그를 불러오고 처리하는 스크립트
"""

import wandb
import pandas as pd


def get_runs_from_project(entity: str, project: str):
    """
    특정 프로젝트의 모든 run을 가져옵니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
    
    Returns:
        runs: wandb.Api.runs 객체
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    return runs


def get_run_history(entity: str, project: str, run_id: str) -> pd.DataFrame:
    """
    특정 run의 로그 히스토리를 DataFrame으로 반환합니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
        run_id: run ID
    
    Returns:
        pd.DataFrame: 로그 히스토리
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()
    return history


def get_run_summary(entity: str, project: str, run_id: str) -> dict:
    """
    특정 run의 summary (최종 메트릭)를 반환합니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
        run_id: run ID
    
    Returns:
        dict: summary 메트릭
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return dict(run.summary)


def get_run_config(entity: str, project: str, run_id: str) -> dict:
    """
    특정 run의 config (하이퍼파라미터)를 반환합니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
        run_id: run ID
    
    Returns:
        dict: config
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return dict(run.config)


def get_run_id_name_pairs(entity: str, project: str) -> dict:
    """
    프로젝트의 모든 run에 대해 run_name -> run_id 매핑을 반환합니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
    
    Returns:
        dict: {run_name: run_id} 딕셔너리
    """
    runs = get_runs_from_project(entity, project)
    return {run.name: run.id for run in runs}


def get_run_id_from_name(entity: str, project: str, run_name: str) -> str | None:
    """
    run_name으로 run_id를 찾습니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
        run_name: run 이름
    
    Returns:
        str | None: run_id (없으면 None)
    """
    pairs = get_run_id_name_pairs(entity, project)
    return pairs.get(run_name)


def resolve_run_identifier(entity: str, project: str, run_identifier: str) -> str:
    """
    run_id 또는 run_name을 받아서 run_id를 반환합니다.
    먼저 run_id로 직접 접근을 시도하고, 실패하면 run_name으로 검색합니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
        run_identifier: run_id 또는 run_name
    
    Returns:
        str: run_id
    
    Raises:
        ValueError: run을 찾을 수 없는 경우
    """
    api = wandb.Api()
    
    # 먼저 run_id로 직접 접근 시도
    try:
        run = api.run(f"{entity}/{project}/{run_identifier}")
        return run.id
    except wandb.errors.CommError:
        pass
    
    # run_name으로 검색
    run_id = get_run_id_from_name(entity, project, run_identifier)
    if run_id is not None:
        return run_id
    
    raise ValueError(f"run_id 또는 run_name '{run_identifier}'을(를) 찾을 수 없습니다.")


def list_all_runs_info(entity: str, project: str) -> pd.DataFrame:
    """
    프로젝트의 모든 run 정보를 DataFrame으로 반환합니다.
    
    Args:
        entity: W&B 사용자 이름 또는 팀 이름
        project: 프로젝트 이름
    
    Returns:
        pd.DataFrame: run 정보 (name, id, state, created_at 등)
    """
    runs = get_runs_from_project(entity, project)
    
    run_info = []
    for run in runs:
        run_info.append({
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "created_at": run.created_at,
            "tags": run.tags,
        })
    
    return pd.DataFrame(run_info)

def filter_by_max_step(df: pd.DataFrame, max_step: int | None) -> pd.DataFrame:
    """
    특정 step 이전의 데이터만 필터링합니다.
    
    Args:
        df: 히스토리 DataFrame
        max_step: 최대 step (None이면 전체 데이터 반환)
    
    Returns:
        pd.DataFrame: 필터링된 DataFrame
    """
    if max_step is None:
        return df
    if "_step" in df.columns:
        return df[df["_step"] <= max_step].copy()
    return df


if __name__ == "__main__":
    # 사용 예시 - 아래 값들을 본인의 entity, project, run_id로 변경하세요
    ENTITY = "haechan-kaist"  # W&B 사용자 이름 또는 팀 이름
    PROJECT = "verl_grpo_prev_epoch_qwen2_5_1_5b_MATH"  # 프로젝트 이름
    
    # RUN_ID_1 = "5kbeoly1" # qwen3-grpo-paper-batch128-cliph0_2-clipl0_28-clipc3-nokl-lr1e-6
    # RUN_ID_2 = "1j5vy957" # qwen3-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6
    
    # RUN_ID_1 = "wsoy74nw" # qwen3-grpo-paper-batch128-cliph0_2-clipl0_28-clipc3-nokl-lr2_3e-6
    # RUN_ID_2 = "d911mqwe" # qwen3-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr2_3e-6

    # RUN_ID_1 = "ym2iscu6" # qwen3-grpo-paper-dapo17k-batch128-cliph0_2-clipl0_28-clipc3-nokl-lr1e-6
    # RUN_ID_2 = "876a13g6" # qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6

    # RUN_ID_1 = "75h60ujw" # qwen3-grpo-paper-dapo17k-batch128-cliph0_2-clipl0_28-clipc3-nokl-lr1e-6-again
    # RUN_ID_2 = "42x4ce9e" # qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-again
    # RUN_ID_3 = "7n9h9rr0" # qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_5
    # RUN_ID_4 = "aygjkyjs" # qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_75
    # RUN_ID_5 = "crk7a4k7" # qwen3-grpohistbeta-paper-dapo17k-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6-df0_25

    # 특정 step 이전의 데이터만 볼 때 사용 (None이면 전체 데이터 사용)
    MAX_STEP = 10000  # 예: 100으로 설정하면 step 100 이전 데이터만 사용

    

    # run_name_1 = "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6"
    run_name_1 = "qwen3-8b-grpo-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6-again"
    run_name_2 = "qwen3-8b-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6"

    run_id_1 = get_run_id_from_name(ENTITY, PROJECT, run_name_1)
    run_id_2 = get_run_id_from_name(ENTITY, PROJECT, run_name_2)

    # 히스토리 불러오기
    history_df_1 = get_run_history(ENTITY, PROJECT, run_id_1)
    history_df_2 = get_run_history(ENTITY, PROJECT, run_id_2)
    
    # MAX_STEP 이전 데이터만 필터링
    history_df_1 = filter_by_max_step(history_df_1, MAX_STEP)
    history_df_2 = filter_by_max_step(history_df_2, MAX_STEP)

    # metrics
    METRIC = "mean@8"
    
    # "best@8/mean"으로 끝나는 컬럼만 필터링
    def filter_columns(df: pd.DataFrame, metric: str) -> list:
        return [col for col in df.columns if col.endswith(metric) and "OlympiadBench" not in col and "amc23" not in col]

    metric_cols = filter_columns(history_df_1, METRIC)
    print(f"=== {METRIC} 컬럼들 ===")
    for col in metric_cols:
        print(f"  - {col}")

    WEIGHTS = {
        f"val-core/HuggingFaceH4/MATH-500/reward/{METRIC}" : 500,
        f"val-core/math-ai/minervamath/reward/{METRIC}" : 272,
        f"val-core/Maxwell-Jia/AIME_2024/reward/{METRIC}" : 30,
        f"val-core/opencompass/AIME2025/reward/{METRIC}" : 30,
        f"val-core/rawsh/2024_AMC12/reward/{METRIC}" : 45,
        # f"val-core/math-ai/amc23/reward/{METRIC}" : 40,
        # f"val-core/Hothan/OlympiadBench/reward/{METRIC}" : 674,
    }


    def compute_weighted_average(df: pd.DataFrame, weights: dict) -> pd.Series:
        """
        best@8/mean 컬럼들의 가중평균을 계산합니다.
        
        Args:
            df: 히스토리 DataFrame
            weights: {컬럼명: 가중치} 딕셔너리
        
        Returns:
            pd.Series: 각 step별 가중평균 값
        """
        # 가중치가 설정된 컬럼만 사용
        cols_with_weights = [col for col in weights.keys() if col in df.columns]
        
        if not cols_with_weights:
            raise ValueError("가중치가 설정된 컬럼이 없습니다. WEIGHTS를 채워주세요.")
        
        # 가중합 계산
        total_weight = sum(weights[col] for col in cols_with_weights)
        weighted_sum = sum(df[col] * weights[col] for col in cols_with_weights)
        
        return weighted_sum / total_weight

    
    # Run 1
    weighted_avg_1 = compute_weighted_average(history_df_1, WEIGHTS)
    best_idx_1 = weighted_avg_1.idxmax()
    best_step_1 = history_df_1.loc[best_idx_1, "_step"] if "_step" in history_df_1.columns else best_idx_1
    best_value_1 = weighted_avg_1[best_idx_1]
    
    print(f"\n=== Run 1 ({run_id_1}) 최고 weighted average ===")
    print(f"  Step: {best_step_1}")
    print(f"  Weighted Avg: {best_value_1:.4f}")
    print(f"  각 메트릭 값:")
    for col in WEIGHTS.keys():
        if col in history_df_1.columns:
            print(f"    - {col}: {history_df_1.loc[best_idx_1, col]:.4f}")

    # Run 2
    weighted_avg_2 = compute_weighted_average(history_df_2, WEIGHTS)
    best_idx_2 = weighted_avg_2.idxmax()
    best_step_2 = history_df_2.loc[best_idx_2, "_step"] if "_step" in history_df_2.columns else best_idx_2
    best_value_2 = weighted_avg_2[best_idx_2]
    
    print(f"\n=== Run 2 ({run_id_2}) 최고 weighted average ===")
    print(f"  Step: {best_step_2}")
    print(f"  Weighted Avg: {best_value_2:.4f}")
    print(f"  각 메트릭 값:")
    for col in WEIGHTS.keys():
        if col in history_df_2.columns:
            print(f"    - {col}: {history_df_2.loc[best_idx_2, col]:.4f}")

    # Run1 - Run2 차이 비교 (각 run의 best step 기준)
    print(f"\n=== Run1 - Run2 차이 (각 best step 기준) ===")
    print(f"  Weighted Avg 차이: {best_value_1 - best_value_2:+.4f}")
    print(f"  각 메트릭 차이:")
    for col in metric_cols:
        val_1 = history_df_1.loc[best_idx_1, col] if col in history_df_1.columns else None
        val_2 = history_df_2.loc[best_idx_2, col] if col in history_df_2.columns else None
        if val_1 is not None and val_2 is not None:
            diff = val_1 - val_2
            print(f"    - {col}: {diff:+.4f}  (Run1: {val_1:.4f}, Run2: {val_2:.4f})")
        else:
            print(f"    - {col}: N/A")


