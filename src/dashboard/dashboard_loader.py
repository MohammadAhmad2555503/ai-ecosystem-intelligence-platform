"""
Dashboard data loading utilities for the AI Ecosystem Intelligence Platform.

This module keeps the Stage 5 dashboard CSV files as the source-of-truth.
The RAG layer should read from these tables instead of inventing separate
analytics data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


EXPECTED_DASHBOARD_FILES = {
    "kpi_summary": "Dataset3_Dashboard_KPI_Summary.csv",
    "top_organisations": "Dataset3_Dashboard_Top_Organisations.csv",
    "topic_leadership": "Dataset3_Dashboard_Topic_Leadership.csv",
    "platform_dominance": "Dataset3_Dashboard_Platform_Dominance.csv",
    "domain_influence": "Dataset3_Dashboard_Domain_Influence.csv",
    "cluster_bridge": "Dataset3_Dashboard_Cluster_Bridge.csv",
    "kg_overview": "Dataset3_Dashboard_KG_Overview.csv",
    "stage_audit": "Dataset3_Stage5_Dashboard_Audit.csv",
}


@dataclass
class DashboardTable:
    """A named dashboard table loaded from a CSV file."""

    table_name: str
    file_path: Path
    rows: list[dict[str, str]]


@dataclass
class DashboardBundle:
    """All Stage 5 dashboard tables needed by the application layer."""

    tables: dict[str, DashboardTable]

    def get_rows(self, table_name: str) -> list[dict[str, str]]:
        table = self.tables.get(table_name)
        return table.rows if table else []


def clean_text(raw_value: object) -> str:
    if raw_value is None:
        return ""
    return " ".join(str(raw_value).replace("\n", " ").split()).strip()


def clean_row(raw_row: dict[str, object]) -> dict[str, str]:
    return {clean_text(column): clean_text(value) for column, value in raw_row.items()}


def load_csv_rows(file_path: Path) -> list[dict[str, str]]:
    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            return [clean_row(row) for row in csv.DictReader(csv_file)]
    except FileNotFoundError:
        raise FileNotFoundError(f"Dashboard file is missing: {file_path}") from None


def check_dashboard_files(dashboard_folder: Path) -> list[str]:
    missing_files = []
    for file_name in EXPECTED_DASHBOARD_FILES.values():
        if not (dashboard_folder / file_name).exists():
            missing_files.append(file_name)
    return missing_files


def load_dashboard_table(dashboard_folder: Path, table_name: str) -> DashboardTable:
    file_name = EXPECTED_DASHBOARD_FILES[table_name]
    file_path = dashboard_folder / file_name
    return DashboardTable(table_name, file_path, load_csv_rows(file_path))


def load_dashboard_bundle(dashboard_folder: Path) -> DashboardBundle:
    missing_files = check_dashboard_files(dashboard_folder)
    if missing_files:
        joined_names = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing dashboard files: {joined_names}")
    tables = build_dashboard_tables(dashboard_folder)
    return DashboardBundle(tables)


def build_dashboard_tables(dashboard_folder: Path) -> dict[str, DashboardTable]:
    loaded_tables = {}
    for table_name in EXPECTED_DASHBOARD_FILES:
        loaded_tables[table_name] = load_dashboard_table(dashboard_folder, table_name)
    return loaded_tables


def count_total_rows(bundle: DashboardBundle) -> int:
    return sum(len(table.rows) for table in bundle.tables.values())

