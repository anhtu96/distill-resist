#!/usr/bin/env python
"""HuggingFace sync helpers: push models to HF Hub and remove local copies."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def push_to_hf(local_dir: str, repo_id: str, path_in_repo: str | None = None) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    subdir = path_in_repo or Path(local_dir).name
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model", path_in_repo=subdir)
    print(f"Pushed {local_dir} -> {repo_id}/{subdir}")


def remove_local(model_dir: str) -> None:
    p = Path(model_dir)
    if p.exists():
        shutil.rmtree(str(p))
        print(f"Removed {p}")


def delete_remote(repo_id: str, folder_path: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.delete_folder(repo_id=repo_id, path_in_repo=folder_path, repo_type="model")
        print(f"Deleted remote folder: {repo_id}/{folder_path}")
    except Exception as e:
        print(f"Failed to delete {repo_id}/{folder_path}: {e}")


def clean_remote_results(repo_id: str, repo_type: str = "model", extensions: tuple = (".jsonl", ".json")) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    target_files = [f for f in files if f.endswith(extensions)]
    if not target_files:
        print(f"No result files ({extensions}) found in {repo_id} ({repo_type})")
        return
    print(f"Found {len(target_files)} result files to delete in {repo_id} ({repo_type}):")
    for f in target_files:
        print(f"Deleting: {f}")
        try:
            api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type=repo_type)
        except Exception as e:
            print(f"  Failed to delete {f}: {e}")
    print("Cleanup completed successfully.")


def download_from_hf(repo_id: str, adapter_dir: str) -> None:
    from huggingface_hub import snapshot_download
    subdir = Path(adapter_dir).name
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{subdir}/*",
        local_dir=str(Path(adapter_dir).parent),
    )
    print(f"Downloaded {repo_id}/{subdir} -> {adapter_dir}")


def get_student_adapter_dir(
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
    normalized: bool = False,
    filter_loops: bool = True,
    with_doge: bool = False,
    doge_coef: float = 1e-4,
    doge_temp: float = 0.7,
    with_ads: bool = False,
    tau: float = 0.7,
    lam: float = 0.01,
    eps: float = 0.01,
) -> str:
    from src.config import Settings
    s = Settings()
    model_name = Path(s.student_model_id).name
    teacher_name = Path(s.teacher_model).name
    dataset_name = s.dataset_name
    base = Path(s.student_output_dir)

    if model_name not in base.name:
        out = base.parent / f"student_{model_name}_{dataset_name}_base_tch_{teacher_name}"
    else:
        out = base

    if name_suffix:
        out = Path(str(out) + name_suffix)

    if paraphrased and "_paraphrased" not in out.name:
        out = Path(str(out) + "_paraphrased")

    if with_caa:
        from src.config import get_caa_param_suffix
        param_suffix = get_caa_param_suffix(s, style=caa_style or None)
        suffix = f"_caa_{caa_style}{param_suffix}" if (caa_style and caa_style != "mixed") else f"_caa{param_suffix}"
        out = Path(str(out) + suffix)

    if with_doge:
        doge_sfx = f"_doge_coef{doge_coef}"
        if not out.name.endswith(doge_sfx):
            out = Path(str(out) + doge_sfx)

    if with_ads:
        ads_sfx = f"_ads_tau{tau}_lam{lam}_eps{eps}"
        if not out.name.endswith(ads_sfx):
            out = Path(str(out) + ads_sfx)

    if normalized and "_normalized" not in out.name:
        out = Path(str(out) + "_normalized")

    if not filter_loops and "_no_filter" not in out.name:
        out = Path(str(out) + "_no_filter")

    return str(out)



def get_student_trace_name(
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
    normalized: bool = False,
    filter_loops: bool = True,
    with_doge: bool = False,
    doge_coef: float = 1e-4,
    doge_temp: float = 0.7,
    with_ads: bool = False,
    tau: float = 0.7,
    lam: float = 0.01,
    eps: float = 0.01,
) -> str:
    from src.config import Settings
    s = Settings()
    student_model_name = Path(s.student_model_id).name
    dataset_name = s.dataset_name
    
    # Base: student_{student_model_name}_base_tch_{dataset_name}
    base_name = f"student_{student_model_name}_base_tch_{dataset_name}"
    
    # CAA parameters
    if with_caa:
        from src.config import get_caa_param_suffix
        param_suffix = get_caa_param_suffix(s, style=caa_style or None)
        suffix = f"_caa_{caa_style}{param_suffix}" if (caa_style and caa_style != "mixed") else f"_caa{param_suffix}"
        base_name += suffix
        
    # Name suffix
    if name_suffix:
        base_name += name_suffix

    if with_doge:
        doge_sfx = f"_doge_coef{doge_coef}"
        if not base_name.endswith(doge_sfx):
            base_name += doge_sfx

    if with_ads:
        ads_sfx = f"_ads_tau{tau}_lam{lam}_eps{eps}"
        if not base_name.endswith(ads_sfx):
            base_name += ads_sfx
        
    if paraphrased and "_paraphrased" not in base_name:
        base_name += "_paraphrased"

    if normalized and "_normalized" not in base_name:
        base_name += "_normalized"

    if not filter_loops and "_no_filter" not in base_name:
        base_name += "_no_filter"
        
    return base_name



def push_results_to_hf(
    repo_id: str,
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
    normalized: bool = False,
    filter_loops: bool = True,
    with_doge: bool = False,
    doge_coef: float = 1e-4,
    doge_temp: float = 0.7,
    with_ads: bool = False,
    tau: float = 0.7,
    lam: float = 0.01,
    eps: float = 0.01,
    repo_type: str = "dataset",
) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
    
    # 1. Get the subfolder name under which to upload the files in the repo
    # This should be the adapter name so that the repo structure remains consistent.
    adapter_dir = get_student_adapter_dir(
        with_caa, caa_style, name_suffix, paraphrased, normalized, filter_loops,
        with_doge, doge_coef, doge_temp, with_ads, tau, lam, eps
    )
    adapter_name = Path(adapter_dir).name
    
    # 2. Get the trace name to locate local JSON/JSONL files
    trace_name = get_student_trace_name(
        with_caa, caa_style, name_suffix, paraphrased, normalized, filter_loops,
        with_doge, doge_coef, doge_temp, with_ads, tau, lam, eps
    )

    
    files_to_upload = [
        Path("data/test") / f"{trace_name}.jsonl",
        Path("data/test") / f"{trace_name}_af.jsonl",
        Path("results") / f"{trace_name}.json",
        Path("results") / f"{trace_name}_af.json",
    ]
    
    uploaded_any = False
    for path in files_to_upload:
        if not path.exists():
            # Try fallback pattern if trace_name varies by formatting (e.g. 3e-5 vs 3e-05)
            candidates = sorted(path.parent.glob(f"{path.stem}*"))
            if candidates:
                path = candidates[0]
        if path.exists():
            # Keep the adapter directory name as the parent folder in the HuggingFace repo
            path_in_repo = f"{adapter_name}/{path.name}"
            print(f"Uploading file: {path} -> {repo_id}/{path_in_repo} (repo_type={repo_type})")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            uploaded_any = True
            
    if not uploaded_any:
        print(f"Warning: No generated JSON/JSONL files found for trace name: {trace_name}")


def push_teacher_test_to_hf(
    repo_id: str,
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
    normalized: bool = False,
    filter_loops: bool = True,
    with_doge: bool = False,
    doge_coef: float = 1e-4,
    doge_temp: float = 0.7,
    with_ads: bool = False,
    tau: float = 0.7,
    lam: float = 0.01,
    eps: float = 0.01,
    repo_type: str = "dataset",
) -> None:
    from huggingface_hub import HfApi
    from src.config import Settings
    s = Settings()
    teacher_tag = Path(s.teacher_model).name
    trace_name = f"teacher_{teacher_tag}_{s.dataset_name}"

    if with_caa:
        style = caa_style or "mixed"
        from src.config import get_caa_param_suffix
        param_suffix = get_caa_param_suffix(s, style=style)
        trace_name += f"_caa_{style}{param_suffix}"

    if name_suffix:
        trace_name += name_suffix

    if with_doge:
        trace_name += f"_doge_coef{doge_coef}"

    if with_ads:
        trace_name += f"_ads_tau{tau}_lam{lam}_eps{eps}"

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)

    path = Path("data/test") / f"{trace_name}.jsonl"
    if not path.exists():
        candidates = sorted(Path("data/test").glob(f"{trace_name}*.jsonl"))
        if candidates:
            path = candidates[0]

    if path.exists():
        path_in_repo = f"teacher_test/{path.name}"
        print(f"Uploading file: {path} -> {repo_id}/{path_in_repo} (repo_type={repo_type})")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    else:
        print(f"Warning: Teacher test file not found at {path}")


def push_teacher_holdout_to_hf(
    repo_id: str,
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
    normalized: bool = False,
    filter_loops: bool = True,
    with_doge: bool = False,
    doge_coef: float = 1e-4,
    doge_temp: float = 0.7,
    with_ads: bool = False,
    tau: float = 0.7,
    lam: float = 0.01,
    eps: float = 0.01,
    repo_type: str = "dataset",
) -> None:
    from huggingface_hub import HfApi
    from src.config import Settings
    s = Settings()
    teacher_tag = Path(s.teacher_model).name
    trace_name = f"teacher_{teacher_tag}_{s.dataset_name}_base"

    subfolder = "base"
    if with_caa:
        subfolder = "caa"
        style = caa_style or "mixed"
        from src.config import get_caa_param_suffix
        param_suffix = get_caa_param_suffix(s, style=style)
        trace_name += f"_caa_{style}{param_suffix}"
    elif with_doge:
        subfolder = "doge"
        trace_name += f"_doge_coef{doge_coef}_temp{doge_temp}"
    elif with_ads:
        subfolder = "ads"
        trace_name += f"_ads_tau{tau}_lam{lam}_eps{eps}"

    if name_suffix:
        trace_name += name_suffix

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)

    path = Path(f"data/teacher_holdout/{subfolder}") / f"{trace_name}.jsonl"
    if not path.exists():
        candidates = sorted(Path(f"data/teacher_holdout/{subfolder}").glob(f"teacher_{teacher_tag}_{s.dataset_name}_base_*.jsonl"))
        if candidates:
            path = candidates[0]
        elif subfolder != "base":
            path = Path("data/teacher_holdout/base") / f"{trace_name}.jsonl"

    if path.exists():
        path_in_repo = f"teacher_holdout/{path.name}"
        print(f"Uploading file: {path} -> {repo_id}/{path_in_repo} (repo_type={repo_type})")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )
    else:
        print(f"Warning: Teacher holdout file not found at {path}")


def main():
    parser = argparse.ArgumentParser(description="HuggingFace model sync utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Push a local dir to HuggingFace")
    p_push.add_argument("local_dir")
    p_push.add_argument("repo_id")
    p_push.add_argument("--path-in-repo", default=None)

    p_rm = sub.add_parser("remove", help="Remove a local model dir")
    p_rm.add_argument("model_dir")

    p_del_remote = sub.add_parser("delete-remote", help="Delete a folder from HuggingFace")
    p_del_remote.add_argument("repo_id")
    p_del_remote.add_argument("folder_path")

    p_dl = sub.add_parser("download", help="Download a model subfolder from HuggingFace")
    p_dl.add_argument("repo_id")
    p_dl.add_argument("adapter_dir")

    def _add_common_flags(p):
        p.add_argument("--with-caa", action="store_true")
        p.add_argument("--caa-style", default="")
        p.add_argument("--name-suffix", default="")
        p.add_argument("--paraphrased", action="store_true")
        p.add_argument("--normalized", action="store_true")
        p.add_argument("--no-filter-loops", action="store_false", dest="filter_loops")
        p.add_argument("--with-doge", action="store_true")
        p.add_argument("--doge-coef", type=float, default=1e-4)
        p.add_argument("--doge-temp", type=float, default=0.7)
        p.add_argument("--with-ads", action="store_true")
        p.add_argument("--tau", type=float, default=0.7)
        p.add_argument("--lam", type=float, default=0.01)
        p.add_argument("--eps", type=float, default=0.01)

    p_dir = sub.add_parser("adapter-dir", help="Print the expected student LoRA adapter path")
    _add_common_flags(p_dir)

    p_res = sub.add_parser("push-results", help="Push generated JSON/JSONL files to HuggingFace")
    p_res.add_argument("repo_id")
    _add_common_flags(p_res)
    p_res.add_argument("--repo-type", default="dataset")

    p_ttest = sub.add_parser("push-teacher-test", help="Push teacher test set to HuggingFace")
    p_ttest.add_argument("repo_id")
    _add_common_flags(p_ttest)
    p_ttest.add_argument("--repo-type", default="dataset")

    p_thold = sub.add_parser("push-teacher-holdout", help="Push teacher holdout dataset to HuggingFace")
    p_thold.add_argument("repo_id")
    _add_common_flags(p_thold)
    p_thold.add_argument("--repo-type", default="dataset")

    p_clean = sub.add_parser("clean-remote-results", help="Delete all .json and .jsonl files from a remote HF repository")
    p_clean.add_argument("repo_id")
    p_clean.add_argument("--repo-type", default="model")


    args = parser.parse_args()

    if args.cmd == "push":
        push_to_hf(args.local_dir, args.repo_id, args.path_in_repo)
    elif args.cmd == "remove":
        remove_local(args.model_dir)
    elif args.cmd == "delete-remote":
        delete_remote(args.repo_id, args.folder_path)
    elif args.cmd == "clean-remote-results":
        clean_remote_results(args.repo_id, repo_type=args.repo_type)
    elif args.cmd == "download":
        download_from_hf(args.repo_id, args.adapter_dir)
    elif args.cmd == "adapter-dir":
        print(get_student_adapter_dir(
            args.with_caa, args.caa_style, args.name_suffix, args.paraphrased, args.normalized, args.filter_loops,
            args.with_doge, args.doge_coef, args.doge_temp, args.with_ads, args.tau, args.lam, args.eps
        ))
    elif args.cmd == "push-results":
        push_results_to_hf(
            repo_id=args.repo_id,
            with_caa=args.with_caa,
            caa_style=args.caa_style,
            name_suffix=args.name_suffix,
            paraphrased=args.paraphrased,
            normalized=args.normalized,
            filter_loops=args.filter_loops,
            with_doge=args.with_doge,
            doge_coef=args.doge_coef,
            doge_temp=args.doge_temp,
            with_ads=args.with_ads,
            tau=args.tau,
            lam=args.lam,
            eps=args.eps,
            repo_type=args.repo_type,
        )
    elif args.cmd == "push-teacher-test":
        push_teacher_test_to_hf(
            repo_id=args.repo_id,
            with_caa=args.with_caa,
            caa_style=args.caa_style,
            name_suffix=args.name_suffix,
            paraphrased=args.paraphrased,
            normalized=args.normalized,
            filter_loops=args.filter_loops,
            with_doge=args.with_doge,
            doge_coef=args.doge_coef,
            doge_temp=args.doge_temp,
            with_ads=args.with_ads,
            tau=args.tau,
            lam=args.lam,
            eps=args.eps,
            repo_type=args.repo_type,
        )
    elif args.cmd == "push-teacher-holdout":
        push_teacher_holdout_to_hf(
            repo_id=args.repo_id,
            with_caa=args.with_caa,
            caa_style=args.caa_style,
            name_suffix=args.name_suffix,
            paraphrased=args.paraphrased,
            normalized=args.normalized,
            filter_loops=args.filter_loops,
            with_doge=args.with_doge,
            doge_coef=args.doge_coef,
            doge_temp=args.doge_temp,
            with_ads=args.with_ads,
            tau=args.tau,
            lam=args.lam,
            eps=args.eps,
            repo_type=args.repo_type,
        )



if __name__ == "__main__":
    main()
