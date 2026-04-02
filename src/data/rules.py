import random
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(slots=True)
class Rule:
    # Target column rules
    unique_value_rate: float = 0.2
    absent_value_rate: float = 0.5

    # Numerical row rules
    min_rows: int = 150

    # Textual length threshold
    text_length_threshold: int = 100


def is_tables_with_few_rows(df: pd.DataFrame, rules: Rule) -> bool:
    if len(df) < rules.min_rows:
        return True

    return False


def _is_long_text(series: pd.Series, rules: Rule) -> bool:
    str_mask = series.dropna().apply(lambda x: isinstance(x, str))
    str_vals = series.dropna()[str_mask]
    if len(str_vals) == 0:
        return False

    return str_vals.str.len().gt(rules.text_length_threshold).any()


def _is_datetime_column(series: pd.Series) -> bool:
    # 根据dtype判断是否为日期时间类型
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    series = series.dropna()
    # 根据python原生对象 datetime/date 判断
    if series.apply(lambda x: isinstance(x, (datetime, date, pd.Timestamp))).any():
        return True

    return False


def get_target_column(df: pd.DataFrame, rules: Rule) -> pd.Series | None:
    candidate_columns = []
    for column in df.columns:
        if _is_long_text(df[column], rules):
            continue
        if _is_datetime_column(df[column]):
            continue
        unique_value_rate = df[column].nunique() / len(df)
        absent_value_rate = df[column].isna().sum() / len(df)
        if (
            unique_value_rate < rules.unique_value_rate
            and absent_value_rate < rules.absent_value_rate
        ):
            candidate_columns.append(column)

    if not candidate_columns:
        return None

    return df[random.choice(candidate_columns)]
