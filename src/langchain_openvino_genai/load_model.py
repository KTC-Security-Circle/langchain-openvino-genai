from pathlib import Path

from huggingface_hub import snapshot_download

# LOCAL_DIR = Path(__file__).parent.parent.parent / "llms"
# 実行しているカレントディレクトリに llms フォルダを作成する
LOCAL_DIR = Path.cwd() / "llms"


def _make_gitignore(path: Path) -> None:
    gitignore_path = path / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.open("w").write("*\n!.gitignore\n")


def load_model(
    repo_id: str,
    *,
    revision: str = "main",
    download_path: str | Path = LOCAL_DIR,
    cache_dir: str | None = None,
    add_gitignore: bool = True,
) -> str:
    """Download the model from Hugging Face Hub and return the local path.

    Args:
        repo_id (str): The repository ID of the model on Hugging Face Hub.
        revision (str, optional): The specific model version to download. Defaults to "main".
        download_path (str | Path, optional): The local directory to download the model to. Defaults to LOCAL_DIR.
        cache_dir (str | None, optional): The directory to cache the downloaded model. Defaults to None.
        add_gitignore (bool, optional): Whether to add a .gitignore file to the download directory. Defaults to True.

    Returns:
        str: The local path to the downloaded model.
    """
    dp = Path(download_path)
    dp.mkdir(parents=True, exist_ok=True)
    if add_gitignore:
        _make_gitignore(dp)

    model_name = repo_id.split("/")[-1]
    model_dir = dp / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        revision=revision,
        cache_dir=cache_dir,
    )
