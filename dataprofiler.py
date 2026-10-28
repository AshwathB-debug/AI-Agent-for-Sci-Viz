# ==========================================
# BLOCK 1: DATA PROFILER
# ==========================================

from config import Union, Dict, Any, Optional, Tuple, List
from pathlib import Path
import pandas as pd
import json
import numpy as np
import openpyxl, xlrd


class DataProfiler:
    
    """
    Data profiler for CSV, TXT, XLSX, JSON.
    Provides column analysis, statistics, data quality, correlations,
    and an LLM-friendly compact summary.
    """

    SUPPORTED_FORMATS = {'.csv', '.txt', '.xlsx', '.xls', '.json'}
    MAX_NULL_RATE_FOR_COERCION = 0.20  # reject silent type-coercion if >20% NaN

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.df: Optional[pd.DataFrame] = None
        self.format: Optional[str] = None
        self._profile: Dict[str, Any] = {}
        self._coercion_log: Dict[str, str] = {}
        self._validate_file()

    def _validate_file(self) -> None:
        if not self.filepath.exists():
            raise FileNotFoundError(f'File not found: {self.filepath}')
        self.format = self.filepath.suffix.lower()
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f'Unsupported format: {self.format}\n'
                f'Supported: {", ".join(sorted(self.SUPPORTED_FORMATS))}'
            )

    def load_data(self, **read_kwargs) -> pd.DataFrame:
        try:
            if self.format == '.csv':
                self.df = pd.read_csv(self.filepath, **read_kwargs)
            elif self.format == '.json':
                try:
                    self.df = pd.read_json(self.filepath, **read_kwargs)
                except Exception:
                    # Fallback for nested JSON, dict-of-dicts, or non-standard orientations
                    with open(self.filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.df = pd.json_normalize(data)
                    elif isinstance(data, dict):
                        # Check if dict values are lists or records
                        self.df = pd.DataFrame.from_dict(data, orient='index' if not any(isinstance(v, list) for v in data.values()) else 'columns')
                    else:
                        self.df = pd.DataFrame([data])
            elif self.format in {'.xlsx', '.xls'}:
                sheet = read_kwargs.pop('sheet_name', 0)
                self.df = pd.read_excel(self.filepath, sheet_name=sheet, **read_kwargs, engine = ("openpyxl" if self.format == '.xlsx' else "xlrd"))
            elif self.format == '.txt':
                delimiter = read_kwargs.pop('delimiter', None)
                if delimiter is None:
                    delimiter = self._infer_delimiter()
                self.df = pd.read_csv(self.filepath, delimiter=delimiter, **read_kwargs)

            self._coerce_types()
            return self.df

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}") from e

    def _coerce_types(self) -> None:
        """Safely coerce object columns to datetime or numeric, both guarded by null-rate."""
        for col in self.df.columns:
            if self.df[col].dtype != 'object':
                continue

            # Try datetime with null-rate guard (fix for silent NaT corruption)
            try:
                converted = pd.to_datetime(
                    self.df[col], format='mixed', errors='coerce'
                )
                null_rate = converted.isna().sum() / max(len(converted), 1)
                if null_rate < self.MAX_NULL_RATE_FOR_COERCION:
                    # Also require that this column wasn't mostly-NaN to start
                    original_non_null = self.df[col].notna().sum()
                    converted_non_null = converted.notna().sum()
                    if original_non_null > 0 and converted_non_null / original_non_null > 0.8:
                        self.df[col] = converted
                        self._coercion_log[col] = f"object → {converted.dtype}"
                        continue
            except Exception:
                pass

            # Try numeric with same guard
            try:
                converted = pd.to_numeric(self.df[col], errors='coerce')
                null_rate = converted.isna().sum() / max(len(converted), 1)
                if null_rate < self.MAX_NULL_RATE_FOR_COERCION:
                    self.df[col] = converted
                    self._coercion_log[col] = f"object → {converted.dtype}"
            except Exception:
                pass

    def _infer_delimiter(self) -> str:
        try:
            with open(self.filepath, 'r') as f:
                sample = f.read(1000)
            delimiters = [',', '\t', '|', ';']
            counts = {d: sample.count(d) for d in delimiters}
            return max(counts, key=counts.get)
        except Exception:
            return ','

    def profile(self, sample_size: Optional[int] = None) -> Dict[str, Any]:
        if self.df is None:
            raise ValueError('Data not loaded. Call load_data() first.')
        df = self.df.head(sample_size) if sample_size else self.df
        self._profile = {
            'metadata': self._profile_metadata(),
            'columns': self._profile_columns(df),
            'statistics': self._profile_statistics(df),
            'quality': self._profile_data_quality(df),
            'correlations': self._profile_correlations(df),
            'sample_data': self._get_sample_data(),
        }
        return self._profile

    def _profile_metadata(self):
        return {
            'filepath': str(self.filepath),
            'format': self.format,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'size_mb': round(self.filepath.stat().st_size / (1024**2), 2),
        }

    def _profile_columns(self, df):
        return {
            col: {
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].notna().sum()),
                'null_pct': round(df[col].isna().sum() / max(len(df), 1) * 100, 2),
                'unique': int(df[col].nunique()),
                'unique_pct': round(df[col].nunique() / max(len(df), 1) * 100, 2),
            }
            for col in df.columns
        }

    def _profile_statistics(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s) > 0:
                stats[col] = {
                    'mean': round(float(s.mean()), 4),
                    'median': round(float(s.median()), 4),
                    'std': round(float(s.std()), 4),
                    'min': round(float(s.min()), 4),
                    'max': round(float(s.max()), 4),
                    'q25': round(float(s.quantile(0.25)), 4),
                    'q75': round(float(s.quantile(0.75)), 4),
                }
        return stats

    def _profile_data_quality(self, df):
        total = df.shape[0] * df.shape[1]
        missing = int(df.isna().sum().sum())
        return {
            'completeness_pct': round((1 - missing / max(total, 1)) * 100, 2),
            'missing_cells': missing,
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_rows_pct': round(df.duplicated().sum() / max(len(df), 1) * 100, 2),
        }

    def _profile_correlations(self, df):
        numeric_df = df.select_dtypes(include=[np.number])
        correlations = {}
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i+1:]:
                    v = corr.loc[c1, c2]
                    if pd.notna(v) and abs(v) > 0.7:
                        correlations[f'{c1} ↔ {c2}'] = round(float(v), 4)
        return correlations

    def _get_sample_data(self):
        sample_df = self.df.head(5)
        return {
            'head_5_rows': sample_df.to_dict(orient='records'),
            'columns': list(self.df.columns),
            'dtypes': {col: str(dt) for col, dt in self.df.dtypes.items()},
            'coerced_columns': self._coercion_log,
        }

    def compact_summary(self, max_cols: int = 50, max_sample_chars: int = 80) -> str:
        """
        LLM-friendly text summary. Much smaller than full JSON dump.
        Scales to wide datasets by truncating columns and samples.
        """
        if self.df is None:
            return "(data not loaded)"

        lines = [f"Dataset: {len(self.df):,} rows × {len(self.df.columns)} columns"]
        cols = list(self.df.columns)
        if len(cols) > max_cols:
            lines.append(f"(showing first {max_cols} of {len(cols)} columns)")
            cols = cols[:max_cols]

        lines.append("\nColumns (name, dtype, nulls, sample values):")
        for col in cols:
            dtype = self.df[col].dtype
            nulls = int(self.df[col].isna().sum())
            samples = self.df[col].dropna().head(3).tolist()
            sample_str = str(samples)
            if len(sample_str) > max_sample_chars:
                sample_str = sample_str[:max_sample_chars] + "...]"
            lines.append(f"  - {col!r} ({dtype}, {nulls} nulls): {sample_str}")

        numeric = self.df.select_dtypes(include=[np.number])
        if len(numeric.columns) > 0:
            lines.append("\nNumeric stats:")
            for col in list(numeric.columns)[:max_cols]:
                s = numeric[col].dropna()
                if len(s) > 0:
                    lines.append(
                        f"  {col}: min={s.min():.4g}, max={s.max():.4g}, "
                        f"mean={s.mean():.4g}, std={s.std():.4g}"
                    )

        if len(numeric.columns) > 1:
            corr = numeric.corr()
            strong = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i+1:]:
                    v = corr.loc[c1, c2]
                    if pd.notna(v) and abs(v) > 0.7:
                        strong.append(f"{c1}↔{c2}: {v:.2f}")
            if strong:
                lines.append(f"\nStrong correlations: {', '.join(strong)}")

        if self._coercion_log:
            lines.append(f"\nType coercions applied: {self._coercion_log}")

        return "\n".join(lines)