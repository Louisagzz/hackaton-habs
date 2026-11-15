"""CPU-only, local-file EDA. No downloads; gaze/pupillometry are never read."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starting_notebooks"))
from bdf_loader import uV_to_uS


def save_csv(frame, out, name):
    frame.to_csv(out / f"{name}.csv", index=False, float_format="%.9g")


def table_audit(data, out):
    """Keep sentinel rows in coverage; use only available labels for label counts."""
    result = {}
    tables = {}
    for name in ["participants_train", "labels_train"]:
        path = data / f"{name}.csv"
        if not path.exists():
            result[name] = {"status": "MISSING", "required_path": str(path)}
            continue
        raw = pd.read_csv(path, dtype="string")
        df = raw.replace("TESTSUBJECT", pd.NA)
        tables[name] = df
        result[name] = {"status": "available", "rows": len(df),
                        "sentinel_rows": int(raw.eq("TESTSUBJECT").any(axis=1).sum()),
                        "columns": list(df.columns),
                        "participant_ids": int(df.participant_id.nunique()) if "participant_id" in df else None}
        save_csv(pd.DataFrame({"field": df.columns, "missing_n": df.isna().sum().values,
                              "missing_fraction": df.isna().mean().values,
                              "sentinel_n": raw.eq("TESTSUBJECT").sum().values}), out, name + "_missing")
        for field in ["cohort", "gender"]:
            if field in df:
                save_csv(df.groupby(field, dropna=False).size().rename("rows").reset_index(), out, name + "_" + field)
        if "cohort" in df and "participant_id" in df:
            save_csv(df.groupby("cohort", dropna=False).participant_id.nunique().rename("participant_ids").reset_index(), out, name + "_cohort_participants")
        if "age" in df:
            save_csv(pd.to_numeric(df.age, errors="coerce").describe().rename_axis("statistic").reset_index(name="age"), out, "age")
        for group in [["participant_id"], ["participant_id", "day"], ["cohort"]]:
            if name == "labels_train" and set(group) <= set(df):
                save_csv(df.groupby(group, dropna=False).size().rename("trial_rows").reset_index(), out, "trials_by_" + "_".join(group))
        if "cohort" in df:
            miss = df.drop(columns="cohort").isna().groupby(df.cohort, dropna=False).mean().reset_index()
            save_csv(miss, out, name + "_missing_by_cohort")
    df = tables.get("labels_train")
    if df is None:
        return result
    if "participants_train" in tables and "participant_id" in df:
        p = tables["participants_train"]
        result["join_checks"] = {"duplicate_participant_ids": int(p.participant_id.dropna().duplicated().sum()),
                                 "trial_ids_absent_from_demographics": sorted(set(df.participant_id.dropna()) - set(p.participant_id.dropna()))}
    key = ["participant_id", "day", "start_marker"]
    if set(key) <= set(df):
        complete = df[key].notna().all(axis=1)
        result["trial_key"] = {"incomplete_rows": int((~complete).sum()),
                               "duplicate_complete_rows": int(df.loc[complete].duplicated(key, keep=False).sum())}
    emotions = [c for c in ["default_emotion_label", "expected_emotion_label", "self_reported_emotion_1", "self_reported_emotion_2", "self_reported_emotion_3", "final_emotion_label"] if c in df]
    counts = []
    for field in emotions:
        for label, n in df[field].value_counts(dropna=False).items():
            counts.append({"source": field, "emotion": label, "n": n})
    save_csv(pd.DataFrame(counts), out, "emotion_distributions")
    agreements = []
    for a, b in combinations(emotions, 2):
        if a.endswith(("_2", "_3")) or b.endswith(("_2", "_3")):
            continue
        pair = df[[a, b]].dropna()
        if len(pair):
            po = float(pair[a].eq(pair[b]).mean())
            pa, pb = pair[a].value_counts(normalize=True), pair[b].value_counts(normalize=True)
            pe = float(pa.mul(pb, fill_value=0).sum())
            agreements.append({"source_a": a, "source_b": b, "paired_n": len(pair), "agreement": po,
                               "cohen_kappa": (po-pe)/(1-pe) if pe < 1 else np.nan})
            save_csv(pd.crosstab(pair[a], pair[b]).reset_index(), out, f"confusion_{a}_{b}")
    save_csv(pd.DataFrame(agreements), out, "annotation_agreement")
    rating_cols = [c for c in ["rating_valence", "rating_arousal", "rating_dominance"] if c in df]
    numeric = df[rating_cols].apply(pd.to_numeric, errors="coerce")
    if rating_cols:
        save_csv(numeric.describe().T.reset_index(names="field"), out, "rating_summary")
    save_csv(pd.DataFrame({"field": rating_cols, "invalid_numeric_n": (df[rating_cols].notna() & numeric.isna()).sum().values,
                          "outside_1_9_n": ((numeric < 1) | (numeric > 9)).sum().values}), out, "rating_validation")
    if {"rating_valence", "rating_arousal"} <= set(numeric):
        fig, ax = plt.subplots(figsize=(6, 5))
        pairs = numeric[["rating_valence", "rating_arousal"]].dropna()
        ax.hist2d(pairs.rating_valence, pairs.rating_arousal, bins=np.linspace(1, 9, 33))
        ax.set(xlabel="Valence", ylabel="Arousal", title="Available training ratings (1-9)")
        fig.tight_layout(); fig.savefig(out / "ratings.png", dpi=150); plt.close(fig)
    if "flagged_outliers" in df:
        for groups in [["flagged_outliers"], ["cohort", "flagged_outliers"], ["final_emotion_label", "flagged_outliers"]]:
            if set(groups) <= set(df):
                save_csv(df.groupby(groups, dropna=False).size().rename("trials").reset_index(), out, "outliers_by_" + "_".join(groups))
        flag = pd.to_numeric(df.flagged_outliers, errors="coerce")
        result["invalid_outlier_values"] = int((df.flagged_outliers.notna() & ~flag.isin([0, 1])).sum())
    # Source comparisons are descriptive dependencies, not independent raters.
    result["agreement_interpretation"] = "Final labels derive from other sources; agreement/kappa is descriptive, not inter-rater reliability."
    return result


def signal_audit(sample, out):
    rows, segments, events, inventory = [], [], [], []
    for meta_path in sorted(sample.rglob("*_metadata.json")):
        meta = json.loads(meta_path.read_text())
        export = meta["dataset_export"]
        source, pid, day = export["source"], export["participant_id"], export["day"]
        for e in meta["events"]:
            events.append({"source": source, "participant_id": pid, "day": day, **e})
        for task, segment in export["segments"].items():
            path = meta_path.parent / segment["file"]
            origin = segment["metadata_time_origin_s"]
            overlaps = [e for e in meta["events"] if e["onset_s"] < origin + segment["duration_s"] + 1e-8 and e["onset_s"] + (e.get("duration_s") or 0) > origin - 1e-8]
            for e in overlaps:
                inventory.append({"source": source, "task": task, "marker_id": e["marker_id"],
                                  "relative_onset_s": e["onset_s"]-origin,
                                  "duration_s": e["duration_s"], "signal_present": path.exists(),
                                  "partial": e["onset_s"] < origin-1e-8 or e["onset_s"]+(e.get("duration_s") or 0) > origin+segment["duration_s"]+1e-8})
            segment_row = {"source": source, "participant_id": pid, "day": day, "task": task,
                           "signal_present": path.exists(), "metadata_event_count": len(overlaps), **segment}
            segments.append(segment_row)
            if not path.exists():
                continue
            record_map = json.loads(path.with_suffix(".record_map").read_text())
            selected = []
            for specs in record_map["stream_map"].values():
                for spec in specs.values():
                    correction = spec.get("correction", {})
                    modality = correction.get("RecordType", "").upper()
                    # Strict allowlist: no gaze or pupil samples read, reconstructed or fused.
                    if modality not in {"EEG", "ECG", "GSR", "RESPIRATION"}:
                        continue
                    indices = spec.get("channel_indices")
                    if indices is None:
                        a, b = spec["channel_ranges"]; indices = list(range(a, b+1))
                    if len(indices) != len(correction["Channels"]):
                        raise ValueError(f"Map mismatch: {path}")
                    for i, detail in zip(indices, correction["Channels"], strict=True):
                        selected.append((i, detail, modality, correction))
            traces = {}
            spectra = []
            with pyedflib.EdfReader(str(path)) as reader:
                segment_row["bdf_storage_duration_s"] = reader.file_duration
                for i, detail, modality, correction in selected:
                    label = reader.getLabel(i).strip()
                    if not detail["name"].startswith(label):
                        raise ValueError(f"Channel name mismatch: {path}, {i}")
                    fs = reader.getSampleFrequency(i)
                    n = min(int(record_map["TotalSamples"]), int(segment["n_samples"]), int(reader.getNSamples()[i]))
                    stored = reader.readSignal(i, 0, n)
                    digital = reader.readSignal(i, 0, n, digital=True)
                    x = uV_to_uS(stored) if modality == "GSR" else stored
                    unit = "uS" if modality == "GSR" else reader.getPhysicalDimension(i)
                    finite = x[np.isfinite(x)]
                    f, power = welch(finite, fs=fs, nperseg=min(len(finite), int(fs*4)))
                    total = power[(f >= .5) & (f <= 100)].sum()
                    row = {"source": source, "task": task, "channel": label, "modality": modality,
                           "header_unit": reader.getPhysicalDimension(i), "map_unit": detail.get("units"), "analysis_unit": unit,
                           "rate_hz": fs, "map_effective_rate_hz": correction.get("EffectiveSamplingFrequency"),
                           "valid_samples": n, "last_sample_s": (n-1)/fs, "storage_samples": int(reader.getNSamples()[i]),
                           "nonfinite_fraction": 1-len(finite)/n,
                           "flat_step_fraction": float(np.mean(np.diff(digital) == 0)),
                           "digital_rail_fraction": float(np.mean((digital == reader.getDigitalMinimum(i)) | (digital == reader.getDigitalMaximum(i)))),
                           "median": float(np.median(finite)), "sd": float(np.std(finite)),
                           "p01": float(np.percentile(finite, 1)), "p99": float(np.percentile(finite, 99)),
                           "line_49_51_fraction_0p5_100": float(power[(f >= 49) & (f <= 51)].sum()/total) if total else None,
                           "gsr_outside_calibration_fraction": float(np.mean((stored < -2.14e6) | (stored > 2.00199e6))) if modality == "GSR" else None,
                           "ecg_abs_over_100mV_fraction": float(np.mean(np.abs(x) > 100)) if modality == "ECG" and unit == "mV" else None,
                           "eeg_abs_over_100uV_fraction": float(np.mean(np.abs(x) > 100)) if modality == "EEG" and unit == "uV" else None}
                    rows.append(row)
                    if modality not in traces:
                        traces[modality] = (label, unit, fs, x)
                    if modality == "EEG": spectra.append(power)
            stem = f"{source.lower()}_{task}"
            fig, axes = plt.subplots(4, 1, figsize=(10, 8))
            for ax, modality in zip(axes, ["EEG", "ECG", "GSR", "RESPIRATION"], strict=True):
                label, unit, fs, x = traces[modality]
                step = max(1, int(fs/100))
                ax.plot(np.arange(0, len(x), step)/fs, x[::step], linewidth=.55)
                ax.set(ylabel=f"{label} ({unit})", xlim=(0, (len(x)-1)/fs))
            axes[0].set_title(f"P001 day 1 {source} {task}: native units, valid samples only")
            axes[-1].set_xlabel("Segment-relative seconds (display subsampled; metrics use every sample)")
            fig.tight_layout(); fig.savefig(out / f"{stem}_signals.png", dpi=150); plt.close(fig)
            # f is common to all selected 1 kHz channels; keep 0.5-100 Hz display.
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.semilogy(f, np.median(spectra, axis=0))
            ax.set(xlim=(.5, 100), xlabel="Frequency (Hz)", ylabel="Median channel PSD (uV²/Hz)", title=f"P001 {source} {task}: EEG Welch PSD")
            fig.tight_layout(); fig.savefig(out / f"{stem}_eeg_psd.png", dpi=150); plt.close(fig)
    channels = pd.DataFrame(rows)
    save_csv(channels, out, "signal_quality")
    save_csv(pd.DataFrame(segments), out, "segments")
    save_csv(pd.DataFrame(inventory), out, "segment_events")
    ev = pd.DataFrame(events)
    save_csv(ev, out, "metadata_events")
    if not ev.empty:
        # CURATED/PREPROCESSED are versions of the same events, not independent trials.
        unique = ev[ev.source == "CURATED"]
        for field in ["trial_type", "emotion_label", "rating_tool"]:
            save_csv(unique.groupby(field, dropna=False).size().rename("events").reset_index(), out, "metadata_" + field)
        save_csv(unique[["valence_score", "arousal_score"]].describe().reset_index(names="statistic"), out, "metadata_ratings")
    return channels, segments, ev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Dataset root containing training CSVs; never scanned for signals")
    parser.add_argument("--sample", type=Path, default=ROOT / "sample", help="Small signal subset only")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/eda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = table_audit(args.data, args.out)
    channels, segments, events = signal_audit(args.sample, args.out)
    inputs = sorted(set(args.sample.rglob("*.bdf")) | set(args.sample.rglob("*.record_map")) | set(args.sample.rglob("*_metadata.json")) | set(args.data.glob("*_train.csv")))
    manifest = {"arguments": {k: str(v) for k, v in vars(args).items()}, "python": sys.version,
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "pyedflib", "mne"]},
                "excluded_modalities": ["gaze", "pupillometry"], "tables": tables, "inputs": []}
    inputs += [Path(__file__), Path(__file__).with_name("interpretation.md"), Path(__file__).with_name("requirements.txt"), ROOT / "starting_notebooks/bdf_loader.py"]
    for path in inputs:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["inputs"].append({"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Phase 1: dataset audit and exploratory analysis", "", "Local audit date: 2026-11-15. CPU only; no dataset downloads or model training.", "", "## Observed input availability", ""]
    for name, result in tables.items():
        lines.append(f"- `{name}`: {json.dumps(result, ensure_ascii=True)}")
    lines += ["", f"Observed signal files: {sum(s['signal_present'] for s in segments)}. Metadata-described missing segments: {sum(not s['signal_present'] for s in segments)}.", "", "## Signal findings", "", "| Version | Task | Channels EEG/ECG/GSR/Resp | Valid samples | Last sample (s) | Storage duration (s) |", "|---|---|---|---:|---:|---:|"]
    for (source, task), group in channels.groupby(["source", "task"], sort=True):
        counts = "/".join(str(int(group.modality.eq(m).sum())) for m in ["EEG", "ECG", "GSR", "RESPIRATION"])
        s = next(s for s in segments if s["source"] == source and s["task"] == task)
        lines.append(f"| {source} | {task} | {counts} | {group.valid_samples.iloc[0]} | {group.last_sample_s.iloc[0]:.3f} | {s['bdf_storage_duration_s']:.3f} |")
    lines += ["", "All analyzed channels have a 1,000 Hz stored rate. EEG is declared uV; ECG mV; respiration AU. GSR is stored in uV and converted to uS using the supplied piecewise resistance calibration (values outside the calibration interval are clamped).", ""]
    for (source, task, modality), group in channels.groupby(["source", "task", "modality"], sort=True):
        lines.append(f"- {source} {task} {modality}: median across channels of channel SD = {group.sd.median():.4g} {group.analysis_unit.iloc[0]}; channel p01-p99 envelope {group.p01.min():.4g} to {group.p99.max():.4g}; nonfinite {group.nonfinite_fraction.max():.3%}; digital rails {group.digital_rail_fraction.max():.3%}; repeated adjacent digital values {group.flat_step_fraction.max():.3%} (maximum channel fractions).")
    lines += ["", "Median EEG 49-51 Hz power fraction of 0.5-100 Hz power (descriptive; channel sets/time spans differ):"]
    for (source, task), group in channels[channels.modality == "EEG"].groupby(["source", "task"]):
        lines.append(f"- {source} {task}: {group.line_49_51_fraction_0p5_100.median():.6%}.")
    gsr = channels[channels.modality == "GSR"]
    lines += [f"GSR outside-calibration fraction: maximum {gsr.gsr_outside_calibration_fraction.max():.3%} across supplied recordings."]
    if not events.empty:
        ev = events[events.source == "CURATED"]
        vids = ev[ev.media_file.notna()]
        lines += ["", "## Sample metadata (not training annotations)", "", f"CURATED contains {len(ev)} events; {len(vids)} media-bearing events, {vids.media_file.nunique()} unique media filenames. Event types: {ev.trial_type.value_counts().to_dict()}. PREPROCESSED repeats these session annotations and is not counted as additional observations.", f"Metadata ratings: {ev.valence_score.notna().sum()} valence and {ev.arousal_score.notna().sum()} arousal values. Their provenance is the event rating_tool, not verified labels_train ratings. See metadata_ratings.csv and metadata_rating_tool.csv."]
        lines.append(f"Metadata rating means: valence {ev.valence_score.mean():.3f}, arousal {ev.arousal_score.mean():.3f}; observed ranges {ev.valence_score.min():g}-{ev.valence_score.max():g} and {ev.arousal_score.min():g}-{ev.arousal_score.max():g}. Emotion-label counts: {vids.emotion_label.value_counts().to_dict()}. Rating-tool counts on media events: {vids.rating_tool.value_counts().to_dict()}.")
    lines += ["", (ROOT / "scripts/eda/interpretation.md").read_text(encoding="utf-8")]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}; audited {len(channels)} channel-recording pairs. Training status: {tables}")


if __name__ == "__main__":
    main()
