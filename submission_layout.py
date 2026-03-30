from pathlib import Path


def submission_output_path(name: str | Path, base_dir: str | Path | None = None) -> Path:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    raw = Path(name)

    if raw.name == "submission.csv" and raw.parent not in (Path("."), Path("")):
        target_dir = base / raw.parent
    elif raw.suffix.lower() == ".csv":
        target_dir = base / raw.stem
    else:
        target_dir = base / raw

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / "submission.csv"


def resolve_submission_path(name: str | Path, base_dir: str | Path | None = None) -> Path:
    base = Path.cwd() if base_dir is None else Path(base_dir)
    raw = Path(name)

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        if raw.is_dir():
            candidates.append(raw / "submission.csv")
    else:
        candidates.append(base / raw)
        if raw.name == "submission.csv" and raw.parent not in (Path("."), Path("")):
            candidates.append(base / raw.parent / "submission.csv")
        elif raw.suffix.lower() == ".csv":
            candidates.append(base / raw.stem / "submission.csv")
        else:
            candidates.append(base / raw / "submission.csv")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    if raw.suffix.lower() == ".csv":
        return base / raw.stem / "submission.csv"
    return base / raw / "submission.csv"
