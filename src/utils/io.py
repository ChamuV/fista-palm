# src/utils/io.py

def make_result_name(name, params=None, ext=None):
    parts = [name]

    if params:
        for k, v in params.items():
            parts.append(f"{k}{v}")

    filename = "_".join(parts)

    if ext is not None:
        filename += f".{ext}"

    return filename


def make_fig_path(root, name, params=None, ext="png"):
    filename = make_result_name(name, params, ext)

    save_dir = root / "results" / "figs"
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / filename


def make_table_path(root, name, params=None, ext="csv"):
    filename = make_result_name(name, params, ext)

    save_dir = root / "results" / "tables"
    save_dir.mkdir(parents=True, exist_ok=True)

    return save_dir / filename


def save_table(df, root, name, params=None, ext="csv", index=False):
    save_path = make_table_path(root, name, params, ext)

    if ext == "csv":
        df.to_csv(save_path, index=index)
    elif ext == "tex":
        df.to_latex(save_path, index=index)
    else:
        raise ValueError("Supported table formats: csv, tex")

    return save_path