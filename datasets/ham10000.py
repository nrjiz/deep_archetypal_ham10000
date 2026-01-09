from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional: used only for numpy-side image loading for plots/interpolation
try:  # pragma: no cover
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio  # type: ignore

try:  # pragma: no cover
    from skimage.transform import resize
except Exception:  # pragma: no cover
    resize = None


DX_CLASSES: List[str] = ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"]
DX_TO_INDEX: Dict[str, int] = {dx: i for i, dx in enumerate(DX_CLASSES)}


def _auto_find_metadata_csv(data_dir: Path) -> Path:
    # Common Kaggle filename
    candidates = [
        data_dir / "HAM10000_metadata.csv",
        data_dir / "metadata.csv",
        data_dir / "ham10000_metadata.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: any csv containing "metadata" in the name
    meta_like = sorted([p for p in data_dir.glob("*.csv") if "meta" in p.name.lower()])
    if meta_like:
        return meta_like[0]

    # Last resort: any csv in root
    any_csv = sorted(list(data_dir.glob("*.csv")))
    if any_csv:
        return any_csv[0]

    raise FileNotFoundError(
        f"Could not find a metadata CSV in {data_dir}. "
        "Place HAM10000_metadata.csv in the folder, or pass --metadata-csv."
    )


def _find_image_dirs(data_dir: Path) -> List[Path]:
    """
    Returns plausible image directories under a HAM10000 root.
    Handles common Kaggle layout: HAM10000_images_part_1 / _part_2.
    """
    # Common Kaggle names (case-insensitive)
    known = []
    for name in [
        "HAM10000_images_part_1",
        "HAM10000_images_part_2",
        "ham10000_images_part_1",
        "ham10000_images_part_2",
        "images",
        "imgs",
    ]:
        p = data_dir / name
        if p.exists() and p.is_dir():
            known.append(p)

    # If not found, search for any subdir containing jpg files
    if not known:
        for p in data_dir.rglob("*"):
            if p.is_dir():
                # quick check: any jpg within 1 level
                jpgs = list(p.glob("*.jpg"))
                if jpgs:
                    known.append(p)

    # de-duplicate
    out = []
    seen = set()
    for p in known:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    if not out:
        raise FileNotFoundError(
            f"Could not find any image directories under {data_dir}. "
            "Expected e.g. HAM10000_images_part_1/ and HAM10000_images_part_2/."
        )
    return out


def build_image_id_to_path(data_dir: Path, exts: Sequence[str] = ("jpg", "jpeg", "png")) -> Dict[str, Path]:
    """
    Map image_id (stem, e.g. 'ISIC_0027419') -> full file path.
    """
    data_dir = Path(data_dir)
    image_dirs = _find_image_dirs(data_dir)

    id_to_path: Dict[str, Path] = {}
    exts_l = {e.lower().lstrip(".") for e in exts}

    for img_dir in image_dirs:
        for p in img_dir.glob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower().lstrip(".")
            if suf not in exts_l:
                continue
            image_id = p.stem
            # Keep the first occurrence; if duplicates exist, prefer the first found
            if image_id not in id_to_path:
                id_to_path[image_id] = p
    return id_to_path


def load_metadata(data_dir: Path, metadata_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Load HAM10000 metadata and attach a 'filepath' column.
    """
    data_dir = Path(data_dir)
    csv_path = Path(metadata_csv) if metadata_csv is not None else _auto_find_metadata_csv(data_dir)

    df = pd.read_csv(csv_path)

    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata file {csv_path} missing columns: {sorted(missing)}")

    id_to_path = build_image_id_to_path(data_dir)
    df["filepath"] = df["image_id"].map(lambda x: str(id_to_path.get(str(x), "")))

    # Drop rows without a matching image file
    before = len(df)
    df = df[df["filepath"].astype(str).str.len() > 0].copy()
    after = len(df)
    if after < before:
        print(f"[HAM10000] Warning: dropped {before-after} rows because image files were not found under {data_dir}.")

    # Normalize common categorical columns
    for col in ["dx", "dx_type", "sex", "localization"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def apply_filters(
    df: pd.DataFrame,
    dx: Optional[Sequence[str]] = None,
    localization: Optional[Sequence[str]] = None,
    dx_type: Optional[Sequence[str]] = None,
    sex: Optional[Sequence[str]] = None,
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()

    def _norm_list(xs: Optional[Sequence[str]]) -> Optional[List[str]]:
        if xs is None:
            return None
        normed = []
        for x in xs:
            if x is None:
                continue
            for part in str(x).split(","):
                part = part.strip().lower()
                if part:
                    normed.append(part)
        return normed if normed else None

    dx = _norm_list(dx)
    localization = _norm_list(localization)
    dx_type = _norm_list(dx_type)
    sex = _norm_list(sex)

    if dx is not None:
        out = out[out["dx"].isin(dx)]
    if localization is not None and "localization" in out.columns:
        out = out[out["localization"].isin(localization)]
    if dx_type is not None and "dx_type" in out.columns:
        out = out[out["dx_type"].isin(dx_type)]
    if sex is not None and "sex" in out.columns:
        out = out[out["sex"].isin(sex)]
    if age_min is not None and "age" in out.columns:
        out = out[out["age"].astype(float) >= float(age_min)]
    if age_max is not None and "age" in out.columns:
        out = out[out["age"].astype(float) <= float(age_max)]

    out = out.reset_index(drop=True)
    return out


def split_train_test_by_lesion(
    df: pd.DataFrame,
    test_size: float = 0.1,
    seed: int = 42,
    group_col: str = "lesion_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group split: ensures no lesion_id appears in both train and test.
    """
    if group_col not in df.columns:
        raise ValueError(f"Group split requested but '{group_col}' not in df columns.")

    lesions = df[group_col].astype(str).unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(lesions)

    n_test = max(1, int(round(len(lesions) * test_size)))
    test_lesions = set(lesions[:n_test])

    test_df = df[df[group_col].astype(str).isin(test_lesions)].copy()
    train_df = df[~df[group_col].astype(str).isin(test_lesions)].copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def make_soft_labels(
    df: pd.DataFrame,
    num_labels: int = 7,
    off_value: float = 0.03,
    on_value: float = 0.82,
    dx_classes: Sequence[str] = DX_CLASSES,
) -> np.ndarray:
    """
    Return soft label vectors (N, num_labels).
    For each sample, label vector has 'on_value' at true class and 'off_value' elsewhere.
    """
    if num_labels < 1 or num_labels > len(dx_classes):
        raise ValueError(f"num_labels must be in [1, {len(dx_classes)}], got {num_labels}")

    dx_to_index = {dx: i for i, dx in enumerate(dx_classes)}

    y = np.ones((len(df), len(dx_classes)), dtype=np.float32) * float(off_value)
    for i, dx in enumerate(df["dx"].astype(str).tolist()):
        if dx not in dx_to_index:
            raise ValueError(f"Unknown dx='{dx}'. Expected one of: {dx_classes}")
        y[i, dx_to_index[dx]] = float(on_value)

    return y[:, :num_labels].astype(np.float32)


def load_image_np(filepath: str, img_size: int = 64) -> np.ndarray:
    """
    Numpy-side image loading for plotting/inference helpers.
    Returns float32 image in [0, 1] with shape (img_size, img_size, 3).
    """
    if resize is None:
        raise ImportError("scikit-image is required for numpy-side resizing (skimage.transform.resize).")

    img = imageio.imread(filepath)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = resize(img, (img_size, img_size, 3), anti_aliasing=True, preserve_range=True).astype(np.float32)
    # normalize to [0,1]
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img
