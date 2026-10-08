"""Load and inspect a CSV dataset."""
from pathlib import Path
import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV and attach basic inspection details in ``DataFrame.attrs``.

    Accepts a CSV path; returns its rows as a DataFrame. Raises FileNotFoundError
    when absent and pandas.errors.ParserError for malformed CSV input.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    frame = pd.read_csv(path)
    frame.attrs["inspection"] = {
        "shape": [int(x) for x in frame.shape],
        "columns": list(frame.columns),
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicate_records": int(frame.duplicated().sum()),
    }
    return frame

