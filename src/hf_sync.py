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

    if with_caa:
        from src.config import format_caa_strength_tag, format_caa_max_inj_tag
        s_str = format_caa_strength_tag(s.caa_strength)
        t_str = format_caa_max_inj_tag(s.caa_max_injection_tokens)
        suffix = f"_caa_{caa_style}_s{s_str}_{t_str}" if caa_style else f"_caa_s{s_str}_{t_str}"
        out = Path(str(out) + suffix)

    return str(out)


def main():
    parser = argparse.ArgumentParser(description="HuggingFace model sync utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Push a local dir to HuggingFace")
    p_push.add_argument("local_dir")
    p_push.add_argument("repo_id")
    p_push.add_argument("--path-in-repo", default=None)

    p_rm = sub.add_parser("remove", help="Remove a local model dir")
    p_rm.add_argument("model_dir")

    p_dl = sub.add_parser("download", help="Download a model subfolder from HuggingFace")
    p_dl.add_argument("repo_id")
    p_dl.add_argument("adapter_dir")

    p_dir = sub.add_parser("adapter-dir", help="Print the expected student LoRA adapter path")
    p_dir.add_argument("--with-caa", action="store_true")
    p_dir.add_argument("--caa-style", default="")
    p_dir.add_argument("--name-suffix", default="")

    args = parser.parse_args()

    if args.cmd == "push":
        push_to_hf(args.local_dir, args.repo_id, args.path_in_repo)
    elif args.cmd == "remove":
        remove_local(args.model_dir)
    elif args.cmd == "download":
        download_from_hf(args.repo_id, args.adapter_dir)
    elif args.cmd == "adapter-dir":
        print(get_student_adapter_dir(args.with_caa, args.caa_style, args.name_suffix))


if __name__ == "__main__":
    main()
