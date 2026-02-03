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

    run_name_1 = "qwen3-dr-grpohistbeta-paper-batch128-cliph1_0-clipl1_0-clipc10-nokl-lr1e-6"
    run_name_2 = "qwen3-dr-grpo-paper-batch128-cliph0_28-clipl0_2-clipc3-nokl-lr1e-6"

    # 각 실험별로 참조할 특정 step (None이면 best step 자동 탐색)
    FIXED_STEP_1 = None  # Run 1에서 볼 step (None이면 best step 탐색)
    FIXED_STEP_2 = None  # Run 2에서 볼 step (None이면 best step 탐색)
    
    # best step 탐색 시 최대 step 범위 (None이면 전체 범위)
    MAX_STEP_1 = 540  # Run 1의 탐색 범위 최대 step
    MAX_STEP_2 = 540  # Run 2의 탐색 범위 최대 step

    run_id_1 = get_run_id_from_name(ENTITY, PROJECT, run_name_1)
    run_id_2 = get_run_id_from_name(ENTITY, PROJECT, run_name_2)

    # 히스토리 불러오기
    history_df_1 = get_run_history(ENTITY, PROJECT, run_id_1)
    history_df_2 = get_run_history(ENTITY, PROJECT, run_id_2)
    
    # MAX_STEP 이전 데이터만 필터링 (FIXED_STEP이 None일 때 탐색 범위 제한)
    history_df_1_filtered = filter_by_max_step(history_df_1, MAX_STEP_1)
    history_df_2_filtered = filter_by_max_step(history_df_2, MAX_STEP_2)
    
    print(f"Run 1: {run_name_1} (fixed_step: {FIXED_STEP_1}, max_step: {MAX_STEP_1})")
    print(f"Run 2: {run_name_2} (fixed_step: {FIXED_STEP_2}, max_step: {MAX_STEP_2})")

    # metrics
    METRIC = "mean@8"
    
    # metric으로 끝나는 컬럼만 필터링
    def filter_columns(df: pd.DataFrame, metric: str) -> list:
        return [col for col in df.columns if col.endswith(metric) and "OlympiadBench" not in col and "amc23" not in col]

    # val-core 또는 val-aux 컬럼 필터링 (단순 평균용)
    def filter_val_columns(df: pd.DataFrame, metric: str, prefix: str = None) -> list:
        """
        prefix: "val-core/", "val-aux/", 또는 None (둘 다 포함)
        """
        cols = []
        for col in df.columns:
            if not col.endswith(metric):
                continue
            if "OlympiadBench" in col or "amc23" in col:
                continue
            if prefix is None:
                if col.startswith("val-core/") or col.startswith("val-aux/"):
                    cols.append(col)
            elif col.startswith(prefix):
                cols.append(col)
        return cols

    metric_cols = filter_columns(history_df_1, METRIC)
    val_core_cols = filter_val_columns(history_df_1, METRIC, "val-core/")
    val_aux_cols = filter_val_columns(history_df_1, METRIC, "val-aux/")
    
    # val-core가 없으면 val-aux 사용
    val_cols_for_avg = val_core_cols if val_core_cols else val_aux_cols
    
    print(f"=== {METRIC} 컬럼들 ===")
    for col in metric_cols:
        print(f"  - {col}")
    
    print(f"\n=== val-core 컬럼들 ===")
    for col in val_core_cols:
        print(f"  - {col}")
    
    print(f"\n=== val-aux 컬럼들 ===")
    for col in val_aux_cols:
        print(f"  - {col}")
    
    print(f"\n=== 단순 평균에 사용할 컬럼들 ({'val-core' if val_core_cols else 'val-aux'}) ===")
    for col in val_cols_for_avg:
        print(f"  - {col}")

    # 가중치 정의 (dataset별 샘플 수 기준)
    DATASET_WEIGHTS = {
        "HuggingFaceH4/MATH-500": 500,
        "math-ai/minervamath": 272,
        "Maxwell-Jia/AIME_2024": 30,
        "opencompass/AIME2025": 30,
        "rawsh/2024_AMC12": 45,
    }
    
    # 동적으로 WEIGHTS 생성 (컬럼 이름에서 dataset 매칭)
    def build_weights(df: pd.DataFrame, metric: str, dataset_weights: dict, prefix: str = None) -> dict:
        """
        prefix: "val-core/", "val-aux/", 또는 None (둘 다 시도, val-core 우선)
        """
        weights = {}
        prefixes_to_try = [prefix] if prefix else ["val-core/", "val-aux/"]
        
        for pfx in prefixes_to_try:
            for col in df.columns:
                if col.startswith(pfx) and col.endswith(metric):
                    for dataset, weight in dataset_weights.items():
                        if dataset in col:
                            weights[col] = weight
                            break
            if weights:  # val-core에서 찾았으면 val-aux는 스킵
                break
        return weights
    
    WEIGHTS = build_weights(history_df_1, METRIC, DATASET_WEIGHTS)


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

    def compute_simple_average(df: pd.DataFrame, metric_columns: list) -> pd.Series:
        """
        메트릭 컬럼들의 단순평균을 계산합니다.
        
        Args:
            df: 히스토리 DataFrame
            metric_columns: 평균을 낼 컬럼명 리스트
        
        Returns:
            pd.Series: 각 step별 단순평균 값
        """
        cols_available = [col for col in metric_columns if col in df.columns]
        
        if not cols_available:
            raise ValueError("사용 가능한 메트릭 컬럼이 없습니다.")
        
        return df[cols_available].mean(axis=1)

    
    # Weighted Average (가중치 매칭 컬럼이 있는 경우에만)
    if WEIGHTS:
        print(f"\n=== 가중 평균에 사용될 컬럼 ({len(WEIGHTS)}개) ===")
        for col, w in WEIGHTS.items():
            print(f"  - {col} (weight: {w})")
        
        # Run 1
        weighted_avg_1 = compute_weighted_average(history_df_1, WEIGHTS)
        best_idx_1 = weighted_avg_1.idxmax()
        best_step_1 = history_df_1.loc[best_idx_1, "_step"] if "_step" in history_df_1.columns else best_idx_1
        best_value_1 = weighted_avg_1[best_idx_1]
        
        print(f"\n=== Run 1 ({run_name_1}) 최고 weighted average ===")
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
        
        print(f"\n=== Run 2 ({run_name_2}) 최고 weighted average ===")
        print(f"  Step: {best_step_2}")
        print(f"  Weighted Avg: {best_value_2:.4f}")
        print(f"  각 메트릭 값:")
        for col in WEIGHTS.keys():
            if col in history_df_2.columns:
                print(f"    - {col}: {history_df_2.loc[best_idx_2, col]:.4f}")

        # Run1 - Run2 차이 비교 (각 run의 best step 기준)
        print(f"\n=== Run1 - Run2 차이 (가중 평균 best step 기준) ===")
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
    else:
        print(f"\n⚠️  가중 평균에 매칭되는 컬럼이 없습니다. 단순 평균만 계산합니다.")

    # ============================================================
    # 특정 Step 또는 Best Step의 벤치마크별 성능 보기
    # ============================================================
    SIMPLE_AVG_METRICS = val_cols_for_avg  # val-core 또는 val-aux 메트릭 사용 (동적으로 찾음)
    
    def get_row_by_step(df: pd.DataFrame, step: int):
        """특정 step의 row index를 반환"""
        if "_step" in df.columns:
            matches = df[df["_step"] == step]
            if len(matches) > 0:
                return matches.index[0]
        return None
    
    def get_metrics_at_step(df: pd.DataFrame, idx, metric_cols: list) -> dict:
        """특정 index의 메트릭 값들을 반환"""
        result = {}
        for col in metric_cols:
            if col in df.columns:
                result[col] = df.loc[idx, col]
        return result
    
    print(f"\n{'='*60}")
    print(f"=== 벤치마크별 성능 비교 ===")
    print(f"{'='*60}")
    print(f"사용 메트릭: {len(SIMPLE_AVG_METRICS)}개")
    for col in SIMPLE_AVG_METRICS:
        print(f"  - {col}")
    
    # Run 1 - 고정 step 또는 best step
    if FIXED_STEP_1 is not None:
        idx_1 = get_row_by_step(history_df_1, FIXED_STEP_1)
        if idx_1 is None:
            print(f"\n⚠️  Run 1: Step {FIXED_STEP_1}을 찾을 수 없습니다.")
            available_steps = sorted(history_df_1["_step"].dropna().unique().tolist())
            print(f"    사용 가능한 steps: {available_steps[:10]}... (총 {len(available_steps)}개)")
        else:
            step_1 = FIXED_STEP_1
    else:
        # best step 찾기 (MAX_STEP으로 필터링된 범위 내에서)
        simple_avg_1 = compute_simple_average(history_df_1_filtered, SIMPLE_AVG_METRICS)
        idx_1 = simple_avg_1.idxmax()
        step_1 = history_df_1_filtered.loc[idx_1, "_step"] if "_step" in history_df_1_filtered.columns else idx_1
    
    # Run 2 - 고정 step 또는 best step
    if FIXED_STEP_2 is not None:
        idx_2 = get_row_by_step(history_df_2, FIXED_STEP_2)
        if idx_2 is None:
            print(f"\n⚠️  Run 2: Step {FIXED_STEP_2}을 찾을 수 없습니다.")
            available_steps = sorted(history_df_2["_step"].dropna().unique().tolist())
            print(f"    사용 가능한 steps: {available_steps[:10]}... (총 {len(available_steps)}개)")
        else:
            step_2 = FIXED_STEP_2
    else:
        # best step 찾기 (MAX_STEP으로 필터링된 범위 내에서)
        simple_avg_2 = compute_simple_average(history_df_2_filtered, SIMPLE_AVG_METRICS)
        idx_2 = simple_avg_2.idxmax()
        step_2 = history_df_2_filtered.loc[idx_2, "_step"] if "_step" in history_df_2_filtered.columns else idx_2
    
    # 결과 출력
    if idx_1 is not None:
        # FIXED_STEP이면 원본 df, 아니면 filtered df 사용
        df_1_for_metrics = history_df_1 if FIXED_STEP_1 is not None else history_df_1_filtered
        metrics_1 = get_metrics_at_step(df_1_for_metrics, idx_1, SIMPLE_AVG_METRICS)
        avg_1 = sum(metrics_1.values()) / len(metrics_1) if metrics_1 else 0
        
        mode_1 = "고정" if FIXED_STEP_1 is not None else f"Best (≤{MAX_STEP_1})"
        print(f"\n=== Run 1 ({run_name_1}) - {mode_1} Step: {int(step_1)} ===")
        print(f"  Simple Avg: {avg_1:.4f}")
        print(f"  각 메트릭 값:")
        for col, val in metrics_1.items():
            print(f"    - {col}: {val:.4f}")
    
    if idx_2 is not None:
        # FIXED_STEP이면 원본 df, 아니면 filtered df 사용
        df_2_for_metrics = history_df_2 if FIXED_STEP_2 is not None else history_df_2_filtered
        metrics_2 = get_metrics_at_step(df_2_for_metrics, idx_2, SIMPLE_AVG_METRICS)
        avg_2 = sum(metrics_2.values()) / len(metrics_2) if metrics_2 else 0
        
        mode_2 = "고정" if FIXED_STEP_2 is not None else f"Best (≤{MAX_STEP_2})"
        print(f"\n=== Run 2 ({run_name_2}) - {mode_2} Step: {int(step_2)} ===")
        print(f"  Simple Avg: {avg_2:.4f}")
        print(f"  각 메트릭 값:")
        for col, val in metrics_2.items():
            print(f"    - {col}: {val:.4f}")
    
    # 비교
    if idx_1 is not None and idx_2 is not None:
        print(f"\n=== Run1 vs Run2 비교 ===")
        print(f"  Run1 Step: {int(step_1)}, Simple Avg: {avg_1:.4f}")
        print(f"  Run2 Step: {int(step_2)}, Simple Avg: {avg_2:.4f}")
        print(f"  차이 (Run1 - Run2): {avg_1 - avg_2:+.4f}")
        print(f"\n  메트릭별 차이:")
        for col in SIMPLE_AVG_METRICS:
            val_1 = metrics_1.get(col)
            val_2 = metrics_2.get(col)
            if val_1 is not None and val_2 is not None:
                diff = val_1 - val_2
                print(f"    - {col}: {diff:+.4f}  (Run1: {val_1:.4f}, Run2: {val_2:.4f})")
