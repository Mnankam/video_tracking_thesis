#!/usr/bin/env python3
"""
report.py
=========

Berichtserstellung für die Validierung von Video-Tracking- und IMU-Daten.

Das Modul ist bewusst lose gekoppelt und kann daher mit den Ausgaben aus

    video_loader.py
    imu_loader.py
    preprocessing.py
    synchronization.py
    metrics.py
    plotting.py

verwendet werden, ohne deren interne Implementierung vorauszusetzen.

Unterstützte Eingaben
---------------------
- skalare Kennzahlen als ``dict``
- verschachtelte Kennzahlen
- pandas.DataFrame / pandas.Series (optional)
- NumPy-Arrays (optional)
- Listen und Tupel
- bereits von ``plotting.py`` erzeugte Bilddateien
- Metadaten zu Video, IMU, Synchronisation und Laufzeit

Erzeugte Dateien
----------------
- report.json          Maschinenlesbarer Gesamtbericht
- summary.txt          Kompakte Textzusammenfassung
- report.html          Eigenständiger HTML-Bericht
- metrics.csv          Flache Kennzahlentabelle
- run_metadata.json    Lauf- und Eingabemetadaten

Das Modul benötigt nur die Python-Standardbibliothek. NumPy und pandas werden
automatisch genutzt, wenn sie installiert sind.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import html
import json
import logging
import math
import os
import platform
import shutil
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore


# ---------------------------------------------------------------------------
# Konfiguration und Ergebnisobjekte
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReportConfig:
    """Konfiguration für die Berichtserstellung."""

    output_dir: Path | str
    title: str = "Validierung Video-Tracking vs. IMU"
    run_name: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    copy_figures: bool = True
    overwrite: bool = True
    write_json: bool = True
    write_html: bool = True
    write_text: bool = True
    write_csv: bool = True
    include_environment: bool = True
    float_precision: int = 6
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
        if self.float_precision < 0:
            raise ValueError("float_precision muss >= 0 sein.")


@dataclass(slots=True)
class ReportArtifacts:
    """Pfade der erzeugten Berichtartefakte."""

    output_dir: Path
    json_path: Optional[Path] = None
    html_path: Optional[Path] = None
    text_path: Optional[Path] = None
    metrics_csv_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    figure_paths: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "json_path": str(self.json_path) if self.json_path else None,
            "html_path": str(self.html_path) if self.html_path else None,
            "text_path": str(self.text_path) if self.text_path else None,
            "metrics_csv_path": (
                str(self.metrics_csv_path) if self.metrics_csv_path else None
            ),
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "figure_paths": [str(path) for path in self.figure_paths],
        }


# ---------------------------------------------------------------------------
# Allgemeine Hilfsfunktionen
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Liefert den aktuellen UTC-Zeitpunkt im ISO-8601-Format."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_output_directory(path: Path, overwrite: bool = True) -> Path:
    """Erstellt den Ausgabeordner und prüft dessen Verwendbarkeit."""

    path = path.expanduser().resolve()

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Ausgabepfad ist kein Verzeichnis: {path}")

    path.mkdir(parents=True, exist_ok=True)

    if not overwrite and any(path.iterdir()):
        raise FileExistsError(
            f"Ausgabeordner ist nicht leer und overwrite=False: {path}"
        )

    return path


def _is_numpy_scalar(value: Any) -> bool:
    return np is not None and isinstance(value, np.generic)


def _is_numpy_array(value: Any) -> bool:
    return np is not None and isinstance(value, np.ndarray)


def _is_pandas_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _is_pandas_series(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.Series)


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def to_serializable(value: Any) -> Any:
    """
    Konvertiert typische Pipeline-Objekte rekursiv in JSON-kompatible Werte.

    NaN und unendliche Werte werden als ``None`` gespeichert, damit gültiges
    JSON erzeugt wird.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return _finite_or_none(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if dataclasses.is_dataclass(value):
        return to_serializable(dataclasses.asdict(value))

    if _is_numpy_scalar(value):
        scalar = value.item()
        return to_serializable(scalar)

    if _is_numpy_array(value):
        return to_serializable(value.tolist())

    if _is_pandas_dataframe(value):
        frame = value.copy()
        frame = frame.where(frame.notna(), None)
        return {
            "type": "DataFrame",
            "columns": [str(column) for column in frame.columns],
            "index": [to_serializable(index) for index in frame.index.tolist()],
            "records": to_serializable(frame.to_dict(orient="records")),
            "shape": list(frame.shape),
        }

    if _is_pandas_series(value):
        series = value.copy()
        series = series.where(series.notna(), None)
        return {
            "type": "Series",
            "name": str(series.name) if series.name is not None else None,
            "index": [to_serializable(index) for index in series.index.tolist()],
            "values": to_serializable(series.tolist()),
            "length": int(len(series)),
        }

    if isinstance(value, Mapping):
        return {str(key): to_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_serializable(item) for item in value]

    if hasattr(value, "tolist"):
        try:
            return to_serializable(value.tolist())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return to_serializable(vars(value))
        except Exception:
            pass

    return str(value)


def flatten_mapping(
    mapping: Mapping[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flacht ein verschachteltes Dictionary für CSV- und Textausgaben ab."""

    flattened: dict[str, Any] = {}

    for key, value in mapping.items():
        full_key = f"{parent_key}{separator}{key}" if parent_key else str(key)

        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, full_key, separator))
        elif isinstance(value, (list, tuple)):
            if all(
                item is None or isinstance(item, (str, bool, int, float))
                for item in value
            ):
                flattened[full_key] = value
            else:
                flattened[full_key] = to_serializable(value)
        else:
            flattened[full_key] = value

    return flattened


def format_value(value: Any, precision: int = 6) -> str:
    """Formatiert Werte robust für Text- und HTML-Berichte."""

    if value is None:
        return "n/a"

    if isinstance(value, bool):
        return "ja" if value else "nein"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{precision}g}"

    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item, precision) for item in value)

    if isinstance(value, Mapping):
        return json.dumps(
            to_serializable(value),
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def collect_environment() -> dict[str, Any]:
    """Sammelt reproduzierbare Laufzeitinformationen."""

    environment: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "working_directory": str(Path.cwd()),
        "process_id": os.getpid(),
    }

    if np is not None:
        environment["numpy_version"] = getattr(np, "__version__", "unknown")
    if pd is not None:
        environment["pandas_version"] = getattr(pd, "__version__", "unknown")

    return environment


def file_metadata(path: Path | str) -> dict[str, Any]:
    """Liefert Metadaten einer Eingabedatei, ohne deren Inhalt zu verändern."""

    file_path = Path(path).expanduser()
    result: dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.exists(),
    }

    if file_path.exists():
        stat = file_path.stat()
        result.update(
            {
                "resolved_path": str(file_path.resolve()),
                "size_bytes": int(stat.st_size),
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
            }
        )

    return result


# ---------------------------------------------------------------------------
# Kennzahlenaufbereitung
# ---------------------------------------------------------------------------

def normalize_metrics(metrics: Any) -> dict[str, Any]:
    """
    Normalisiert die Ausgabe von ``metrics.py`` zu einem Dictionary.

    Akzeptiert Dictionaries, DataFrames, Series, Dataclasses und Objekte mit
    ``to_dict()``.
    """

    if metrics is None:
        return {}

    if isinstance(metrics, Mapping):
        return dict(to_serializable(metrics))

    if _is_pandas_dataframe(metrics):
        frame = metrics
        if set(frame.columns) >= {"metric", "value"}:
            return {
                str(row["metric"]): to_serializable(row["value"])
                for _, row in frame.iterrows()
            }
        return {"table": to_serializable(frame)}

    if _is_pandas_series(metrics):
        return dict(to_serializable(metrics.to_dict()))

    if dataclasses.is_dataclass(metrics):
        return dict(to_serializable(dataclasses.asdict(metrics)))

    if hasattr(metrics, "to_dict"):
        converted = metrics.to_dict()
        if isinstance(converted, Mapping):
            return dict(to_serializable(converted))

    if hasattr(metrics, "__dict__"):
        return dict(to_serializable(vars(metrics)))

    raise TypeError(
        "metrics muss ein Mapping, DataFrame, Series, Dataclass oder ein "
        "Objekt mit to_dict()/__dict__ sein."
    )


def infer_quality_assessment(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """
    Erstellt eine vorsichtige qualitative Bewertung bekannter Kennzahlen.

    Die Funktion verändert keine numerischen Ergebnisse und verwendet nur
    generische Hinweise. Projektspezifische Grenzwerte sollten über
    ``assessment`` an ``generate_report`` übergeben werden.
    """

    flat = {
        key.lower(): value
        for key, value in flatten_mapping(metrics).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }

    observations: list[str] = []

    for key, value in flat.items():
        if not math.isfinite(float(value)):
            continue

        if "correlation" in key or key.endswith(".r") or key == "r":
            observations.append(
                f"Korrelationskennzahl '{key}': {format_value(value)}."
            )
        elif "rmse" in key:
            observations.append(f"RMSE '{key}': {format_value(value)}.")
        elif key.endswith("mae") or ".mae" in key:
            observations.append(f"MAE '{key}': {format_value(value)}.")
        elif "lag" in key or "offset" in key:
            observations.append(
                f"Zeitversatz/Offset '{key}': {format_value(value)}."
            )
        elif "coverage" in key or "valid_ratio" in key:
            observations.append(
                f"Datenabdeckung '{key}': {format_value(value)}."
            )

    return {
        "status": "informativ",
        "automatic_interpretation": bool(observations),
        "observations": observations,
        "note": (
            "Die automatische Bewertung ist rein beschreibend. "
            "Fachlich begründete Grenzwerte müssen projektspezifisch "
            "festgelegt werden."
        ),
    }


# ---------------------------------------------------------------------------
# Abbildungen
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"}


def normalize_figure_paths(
    figures: Optional[Iterable[Path | str] | Mapping[str, Path | str]],
) -> list[tuple[str, Path]]:
    """Normalisiert Abbildungspfade aus ``plotting.py``."""

    if figures is None:
        return []

    normalized: list[tuple[str, Path]] = []

    if isinstance(figures, Mapping):
        iterable = figures.items()
    else:
        iterable = ((Path(item).stem, item) for item in figures)

    for label, raw_path in iterable:
        path = Path(raw_path).expanduser()
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            LOGGER.warning("Nicht unterstütztes Bildformat übersprungen: %s", path)
            continue
        if not path.exists() or not path.is_file():
            LOGGER.warning("Abbildung nicht gefunden und übersprungen: %s", path)
            continue
        normalized.append((str(label), path.resolve()))

    return normalized


def prepare_figures(
    figures: Optional[Iterable[Path | str] | Mapping[str, Path | str]],
    output_dir: Path,
    copy_figures: bool,
) -> list[dict[str, Any]]:
    """Kopiert bzw. referenziert Abbildungen für den Bericht."""

    normalized = normalize_figure_paths(figures)
    if not normalized:
        return []

    figure_dir = output_dir / "figures"
    if copy_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for index, (label, source) in enumerate(normalized, start=1):
        if copy_figures:
            candidate = source.name
            if candidate in used_names:
                candidate = f"{source.stem}_{index}{source.suffix}"
            used_names.add(candidate)
            destination = figure_dir / candidate
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            report_path = destination
        else:
            report_path = source

        try:
            relative_path = report_path.relative_to(output_dir)
            html_path = relative_path.as_posix()
        except ValueError:
            html_path = report_path.as_uri()

        records.append(
            {
                "label": label,
                "source_path": str(source),
                "report_path": str(report_path),
                "html_path": html_path,
                "size_bytes": report_path.stat().st_size,
            }
        )

    return records


# ---------------------------------------------------------------------------
# Dateischreiber
# ---------------------------------------------------------------------------

def write_json_file(
    path: Path,
    data: Any,
    encoding: str = "utf-8",
) -> Path:
    """Schreibt gültiges, gut lesbares JSON."""

    with path.open("w", encoding=encoding, newline="\n") as handle:
        json.dump(
            to_serializable(data),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return path


def write_metrics_csv(
    path: Path,
    metrics: Mapping[str, Any],
    precision: int = 6,
    encoding: str = "utf-8",
) -> Path:
    """Schreibt alle Kennzahlen als flache CSV-Tabelle."""

    flat = flatten_mapping(to_serializable(metrics))

    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in sorted(flat):
            writer.writerow([key, format_value(flat[key], precision)])

    return path


def render_text_report(report: Mapping[str, Any], precision: int = 6) -> str:
    """Erzeugt eine kompakte Textdarstellung."""

    lines: list[str] = []
    title = str(report.get("title", "Bericht"))

    lines.extend(
        [
            title,
            "=" * len(title),
            "",
            f"Erstellt: {report.get('created_utc', 'n/a')}",
            f"Laufname: {report.get('run_name') or 'n/a'}",
        ]
    )

    if report.get("author"):
        lines.append(f"Autor: {report['author']}")
    if report.get("description"):
        lines.extend(["", str(report["description"])])

    lines.extend(["", "KENNZAHLEN", "----------"])
    flat_metrics = flatten_mapping(report.get("metrics", {}))
    if flat_metrics:
        for key in sorted(flat_metrics):
            lines.append(f"{key}: {format_value(flat_metrics[key], precision)}")
    else:
        lines.append("Keine Kennzahlen vorhanden.")

    synchronization = report.get("synchronization", {})
    if synchronization:
        lines.extend(["", "SYNCHRONISATION", "---------------"])
        for key, value in sorted(flatten_mapping(synchronization).items()):
            lines.append(f"{key}: {format_value(value, precision)}")

    assessment = report.get("assessment", {})
    if assessment:
        lines.extend(["", "BEWERTUNG", "---------"])
        status = assessment.get("status")
        if status:
            lines.append(f"Status: {status}")
        for observation in assessment.get("observations", []):
            lines.append(f"- {observation}")
        if assessment.get("note"):
            lines.append(str(assessment["note"]))

    figures = report.get("figures", [])
    if figures:
        lines.extend(["", "ABBILDUNGEN", "-----------"])
        for figure in figures:
            lines.append(
                f"- {figure.get('label', 'Abbildung')}: "
                f"{figure.get('report_path', 'n/a')}"
            )

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "WARNUNGEN", "--------"])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.append("")
    return "\n".join(lines)


def _html_table(
    mapping: Mapping[str, Any],
    precision: int,
) -> str:
    flat = flatten_mapping(mapping)
    if not flat:
        return "<p>Keine Daten vorhanden.</p>"

    rows = []
    for key in sorted(flat):
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(format_value(flat[key], precision))}</td>"
            "</tr>"
        )
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def render_html_report(report: Mapping[str, Any], precision: int = 6) -> str:
    """Erzeugt einen eigenständigen, responsiven HTML-Bericht."""

    title = html.escape(str(report.get("title", "Bericht")))
    run_name = html.escape(str(report.get("run_name") or "n/a"))
    created = html.escape(str(report.get("created_utc", "n/a")))
    author = report.get("author")
    description = report.get("description")

    figure_html: list[str] = []
    for figure in report.get("figures", []):
        label = html.escape(str(figure.get("label", "Abbildung")))
        source = html.escape(str(figure.get("html_path", "")), quote=True)
        figure_html.append(
            "<figure>"
            f'<img src="{source}" alt="{label}" loading="lazy">'
            f"<figcaption>{label}</figcaption>"
            "</figure>"
        )

    observations = report.get("assessment", {}).get("observations", [])
    observation_html = (
        "<ul>"
        + "".join(f"<li>{html.escape(str(item))}</li>" for item in observations)
        + "</ul>"
        if observations
        else "<p>Keine automatische Beobachtung vorhanden.</p>"
    )

    warning_items = report.get("warnings", [])
    warnings_html = (
        "<ul>"
        + "".join(f"<li>{html.escape(str(item))}</li>" for item in warning_items)
        + "</ul>"
        if warning_items
        else "<p>Keine Warnungen.</p>"
    )

    author_line = (
        f"<p><strong>Autor:</strong> {html.escape(str(author))}</p>"
        if author
        else ""
    )
    description_html = (
        f"<p class=\"description\">{html.escape(str(description))}</p>"
        if description
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
    color-scheme: light dark;
    --bg: #f5f7fa;
    --card: #ffffff;
    --text: #17202a;
    --muted: #5d6d7e;
    --border: #d5d8dc;
    --accent: #1f618d;
}}
@media (prefers-color-scheme: dark) {{
    :root {{
        --bg: #111827;
        --card: #1f2937;
        --text: #f3f4f6;
        --muted: #cbd5e1;
        --border: #4b5563;
        --accent: #7dd3fc;
    }}
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
}}
main {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1rem 4rem; }}
header, section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}}
h1, h2 {{ color: var(--accent); }}
h1 {{ margin-top: 0; }}
.meta {{ color: var(--muted); }}
.description {{ white-space: pre-wrap; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{
    border-bottom: 1px solid var(--border);
    text-align: left;
    padding: 0.55rem;
    vertical-align: top;
}}
th {{ width: 48%; overflow-wrap: anywhere; }}
figure {{ margin: 1.2rem 0; }}
img {{
    display: block;
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
}}
figcaption {{ margin-top: 0.45rem; color: var(--muted); }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<main>
<header>
    <h1>{title}</h1>
    <p class="meta"><strong>Lauf:</strong> {run_name}<br>
    <strong>Erstellt (UTC):</strong> {created}</p>
    {author_line}
    {description_html}
</header>

<section>
    <h2>Kennzahlen</h2>
    {_html_table(report.get("metrics", {{}}), precision)}
</section>

<section>
    <h2>Synchronisation</h2>
    {_html_table(report.get("synchronization", {{}}), precision)}
</section>

<section>
    <h2>Eingabedaten und Verarbeitung</h2>
    {_html_table(report.get("inputs", {{}}), precision)}
    {_html_table(report.get("preprocessing", {{}}), precision)}
</section>

<section>
    <h2>Bewertung</h2>
    <p><strong>Status:</strong>
    {html.escape(str(report.get("assessment", {{}}).get("status", "n/a")))}</p>
    {observation_html}
    <p>{html.escape(str(report.get("assessment", {{}}).get("note", "")))}</p>
</section>

<section>
    <h2>Abbildungen</h2>
    {''.join(figure_html) if figure_html else '<p>Keine Abbildungen vorhanden.</p>'}
</section>

<section>
    <h2>Warnungen</h2>
    {warnings_html}
</section>

<section>
    <h2>Laufzeitumgebung</h2>
    {_html_table(report.get("environment", {{}}), precision)}
</section>
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Öffentliche Hauptschnittstelle
# ---------------------------------------------------------------------------

def build_report_data(
    *,
    config: ReportConfig,
    metrics: Any,
    video_info: Optional[Mapping[str, Any]] = None,
    imu_info: Optional[Mapping[str, Any]] = None,
    preprocessing_info: Optional[Mapping[str, Any]] = None,
    synchronization_info: Optional[Mapping[str, Any]] = None,
    assessment: Optional[Mapping[str, Any]] = None,
    figures: Optional[Iterable[Path | str] | Mapping[str, Path | str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Baut die vollständige Berichtsstruktur auf."""

    output_dir = Path(config.output_dir).resolve()
    normalized_metrics = normalize_metrics(metrics)
    figure_records = prepare_figures(
        figures=figures,
        output_dir=output_dir,
        copy_figures=config.copy_figures,
    )

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "title": config.title,
        "run_name": config.run_name,
        "author": config.author,
        "description": config.description,
        "created_utc": utc_now_iso(),
        "inputs": {
            "video": to_serializable(video_info or {}),
            "imu": to_serializable(imu_info or {}),
        },
        "preprocessing": to_serializable(preprocessing_info or {}),
        "synchronization": to_serializable(synchronization_info or {}),
        "metrics": normalized_metrics,
        "assessment": to_serializable(
            assessment or infer_quality_assessment(normalized_metrics)
        ),
        "figures": figure_records,
        "warnings": list(warnings or []),
        "extra": to_serializable(extra or {}),
        "environment": collect_environment() if config.include_environment else {},
    }

    return report


def generate_report(
    *,
    output_dir: Path | str,
    metrics: Any,
    figures: Optional[
        Iterable[Path | str] | Mapping[str, Path | str]
    ] = None,
    video_info: Optional[Mapping[str, Any]] = None,
    imu_info: Optional[Mapping[str, Any]] = None,
    preprocessing_info: Optional[Mapping[str, Any]] = None,
    synchronization_info: Optional[Mapping[str, Any]] = None,
    assessment: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
    title: str = "Validierung Video-Tracking vs. IMU",
    run_name: Optional[str] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
    copy_figures: bool = True,
    overwrite: bool = True,
    write_json: bool = True,
    write_html: bool = True,
    write_text: bool = True,
    write_csv: bool = True,
    include_environment: bool = True,
    float_precision: int = 6,
) -> ReportArtifacts:
    """
    Erzeugt den vollständigen Bericht.

    Diese Funktion ist die empfohlene Schnittstelle für
    ``validate_video_vs_imu_v2.py``.

    Beispiel
    --------
    artifacts = generate_report(
        output_dir="results/run_001",
        metrics=metric_results,
        figures=plot_paths,
        video_info=video_metadata,
        imu_info=imu_metadata,
        preprocessing_info=preprocessing_metadata,
        synchronization_info=sync_results,
    )
    """

    config = ReportConfig(
        output_dir=output_dir,
        title=title,
        run_name=run_name,
        author=author,
        description=description,
        copy_figures=copy_figures,
        overwrite=overwrite,
        write_json=write_json,
        write_html=write_html,
        write_text=write_text,
        write_csv=write_csv,
        include_environment=include_environment,
        float_precision=float_precision,
    )

    config.output_dir = ensure_output_directory(
        Path(config.output_dir),
        overwrite=config.overwrite,
    )

    report = build_report_data(
        config=config,
        metrics=metrics,
        video_info=video_info,
        imu_info=imu_info,
        preprocessing_info=preprocessing_info,
        synchronization_info=synchronization_info,
        assessment=assessment,
        figures=figures,
        extra=extra,
        warnings=warnings,
    )

    artifacts = ReportArtifacts(output_dir=config.output_dir)
    artifacts.figure_paths = [
        Path(item["report_path"]) for item in report.get("figures", [])
    ]

    metadata = {
        "schema_version": report["schema_version"],
        "title": report["title"],
        "run_name": report["run_name"],
        "created_utc": report["created_utc"],
        "inputs": report["inputs"],
        "preprocessing": report["preprocessing"],
        "synchronization": report["synchronization"],
        "environment": report["environment"],
        "extra": report["extra"],
    }
    artifacts.metadata_path = write_json_file(
        config.output_dir / "run_metadata.json",
        metadata,
        encoding=config.encoding,
    )

    if config.write_json:
        artifacts.json_path = write_json_file(
            config.output_dir / "report.json",
            report,
            encoding=config.encoding,
        )

    if config.write_csv:
        artifacts.metrics_csv_path = write_metrics_csv(
            config.output_dir / "metrics.csv",
            report["metrics"],
            precision=config.float_precision,
            encoding=config.encoding,
        )

    if config.write_text:
        artifacts.text_path = config.output_dir / "summary.txt"
        artifacts.text_path.write_text(
            render_text_report(report, config.float_precision),
            encoding=config.encoding,
            newline="\n",
        )

    if config.write_html:
        artifacts.html_path = config.output_dir / "report.html"
        artifacts.html_path.write_text(
            render_html_report(report, config.float_precision),
            encoding=config.encoding,
            newline="\n",
        )

    LOGGER.info("Bericht wurde in %s erzeugt.", config.output_dir)
    return artifacts


# Rückwärtskompatible Aliasnamen für eine einfache Integration.
create_report = generate_report
save_report = generate_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_json(path: Path | str) -> Any:
    """Lädt eine JSON-Datei."""

    json_path = Path(path).expanduser()
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt einen HTML-, JSON-, CSV- und Textbericht aus bereits "
            "berechneten Video/IMU-Kennzahlen."
        )
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        required=True,
        help="JSON-Datei mit den Kennzahlen aus metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Zielordner des Berichts.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help=(
            "Optionale JSON-Datei mit video_info, imu_info, "
            "preprocessing_info, synchronization_info, assessment und extra."
        ),
    )
    parser.add_argument(
        "--figure",
        action="append",
        default=[],
        type=Path,
        help="Abbildung aus plotting.py; Option kann mehrfach angegeben werden.",
    )
    parser.add_argument("--title", default="Validierung Video-Tracking vs. IMU")
    parser.add_argument("--run-name")
    parser.add_argument("--author")
    parser.add_argument("--description")
    parser.add_argument(
        "--no-copy-figures",
        action="store_true",
        help="Abbildungen nicht in den Berichtsordner kopieren.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Keinen HTML-Bericht erzeugen.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Keinen JSON-Gesamtbericht erzeugen.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Keine Textzusammenfassung erzeugen.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Keine metrics.csv erzeugen.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Kommandozeileneinstieg."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        metrics = load_json(args.metrics_json)
        metadata: MutableMapping[str, Any] = {}

        if args.metadata_json:
            loaded_metadata = load_json(args.metadata_json)
            if not isinstance(loaded_metadata, Mapping):
                raise TypeError("--metadata-json muss ein JSON-Objekt enthalten.")
            metadata.update(loaded_metadata)

        artifacts = generate_report(
            output_dir=args.output_dir,
            metrics=metrics,
            figures=args.figure,
            video_info=metadata.get("video_info", metadata.get("video", {})),
            imu_info=metadata.get("imu_info", metadata.get("imu", {})),
            preprocessing_info=metadata.get(
                "preprocessing_info",
                metadata.get("preprocessing", {}),
            ),
            synchronization_info=metadata.get(
                "synchronization_info",
                metadata.get("synchronization", {}),
            ),
            assessment=metadata.get("assessment"),
            extra=metadata.get("extra"),
            warnings=metadata.get("warnings"),
            title=args.title,
            run_name=args.run_name,
            author=args.author,
            description=args.description,
            copy_figures=not args.no_copy_figures,
            write_html=not args.no_html,
            write_json=not args.no_json,
            write_text=not args.no_text,
            write_csv=not args.no_csv,
        )
    except Exception as exc:
        LOGGER.exception("Berichtserstellung fehlgeschlagen: %s", exc)
        return 1

    print(json.dumps(artifacts.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())