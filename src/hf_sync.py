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
        from src.config import format_caa_strength_tag, format_caa_max_inj_tag
        s_str = format_caa_strength_tag(s.effective_caa_strength)
        t_str = format_caa_max_inj_tag(s.caa_max_injection_tokens)
        suffix = f"_caa_{caa_style}_s{s_str}_{t_str}" if caa_style else f"_caa_s{s_str}_{t_str}"
        out = Path(str(out) + suffix)

    return str(out)


def get_student_trace_name(
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
) -> str:
    from src.config import Settings
    s = Settings()
    student_model_name = Path(s.student_model_id).name
    dataset_name = s.dataset_name
    
    # Base: student_{student_model_name}_base_tch_{dataset_name}
    base_name = f"student_{student_model_name}_base_tch_{dataset_name}"
    
    # CAA parameters
    if with_caa:
        from src.config import format_caa_strength_tag, format_caa_max_inj_tag
        s_str = format_caa_strength_tag(s.effective_caa_strength)
        t_str = format_caa_max_inj_tag(s.caa_max_injection_tokens)
        suffix = f"_caa_{caa_style}_s{s_str}_{t_str}" if caa_style else f"_caa_s{s_str}_{t_str}"
        base_name += suffix
        
    # Name suffix
    if name_suffix:
        base_name += name_suffix
        
    if paraphrased and "_paraphrased" not in base_name:
        base_name += "_paraphrased"
        
    return base_name


def push_results_to_hf(
    repo_id: str,
    with_caa: bool = False,
    caa_style: str = "",
    name_suffix: str = "",
    paraphrased: bool = False,
) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    
    # 1. Get the subfolder name under which to upload the files in the repo
    # This should be the adapter name so that the repo structure remains consistent.
    adapter_dir = get_student_adapter_dir(with_caa, caa_style, name_suffix, paraphrased)
    adapter_name = Path(adapter_dir).name
    
    # 2. Get the trace name to locate local JSON/JSONL files
    trace_name = get_student_trace_name(with_caa, caa_style, name_suffix, paraphrased)
    
    files_to_upload = [
        Path("data/test") / f"{trace_name}.jsonl",
        Path("data/test") / f"{trace_name}_af.jsonl",
        Path("results") / f"{trace_name}.json",
        Path("results") / f"{trace_name}_af.json",
    ]
    
    uploaded_any = False
    for path in files_to_upload:
        if path.exists():
            # Keep the adapter directory name as the parent folder in the HuggingFace repo
            path_in_repo = f"{adapter_name}/{path.name}"
            print(f"Uploading file: {path} -> {repo_id}/{path_in_repo}")
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            uploaded_any = True
            
    if not uploaded_any:
        print(f"Warning: No generated JSON/JSONL files found for trace name: {trace_name}")


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

    p_dir = sub.add_parser("adapter-dir", help="Print the expected student LoRA adapter path")
    p_dir.add_argument("--with-caa", action="store_true")
    p_dir.add_argument("--caa-style", default="")
    p_dir.add_argument("--name-suffix", default="")
    p_dir.add_argument("--paraphrased", action="store_true")

    p_res = sub.add_parser("push-results", help="Push generated JSON/JSONL files to HuggingFace")
    p_res.add_argument("repo_id")
    p_res.add_argument("--with-caa", action="store_true")
    p_res.add_argument("--caa-style", default="")
    p_res.add_argument("--name-suffix", default="")
    p_res.add_argument("--paraphrased", action="store_true")

    args = parser.parse_args()

    if args.cmd == "push":
        push_to_hf(args.local_dir, args.repo_id, args.path_in_repo)
    elif args.cmd == "remove":
        remove_local(args.model_dir)
    elif args.cmd == "delete-remote":
        delete_remote(args.repo_id, args.folder_path)
    elif args.cmd == "download":
        download_from_hf(args.repo_id, args.adapter_dir)
    elif args.cmd == "adapter-dir":
        print(get_student_adapter_dir(args.with_caa, args.caa_style, args.name_suffix, args.paraphrased))
    elif args.cmd == "push-results":
        push_results_to_hf(
            repo_id=args.repo_id,
            with_caa=args.with_caa,
            caa_style=args.caa_style,
            name_suffix=args.name_suffix,
            paraphrased=args.paraphrased,
        )


if __name__ == "__main__":
    main()
