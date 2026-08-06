from pathlib import Path


def read_model_name(model, path="best_model_name.txt"):
    model_name_path = Path(path)
    if model_name_path.exists():
        name = model_name_path.read_text(encoding="utf-8").strip()
        if name:
            return name

    return model.__class__.__name__
