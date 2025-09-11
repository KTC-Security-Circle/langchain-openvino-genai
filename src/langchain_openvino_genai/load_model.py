from pathlib import Path

from huggingface_hub import snapshot_download

# LOCAL_DIR = Path(__file__).parent.parent.parent / "llms"
# 実行しているカレントディレクトリに llms フォルダを作成する
LOCAL_DIR = Path.cwd() / "llms"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

def load_model(repo_id: str, revision: str = "main", download_path: str | Path = LOCAL_DIR, cache_dir: str | None = None) -> str:
    """Download the model from Hugging Face Hub and return the local path.

    Args:
        repo_id (str): The repository ID of the model on Hugging Face Hub.
        revision (str, optional): The specific model version to download. Defaults to "main".
        download_path (str | Path, optional): The local directory to download the model to. Defaults to LOCAL_DIR.
        cache_dir (str | None, optional): The directory to cache the downloaded model. Defaults to None.

    Returns:
        str: The local path to the downloaded model.
    """
    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(download_path),
        revision=revision,
        cache_dir=cache_dir,
    )
    return local_path