"""
============================================================
 generate_subject_split.py
 Generates the ONE shared subject-level Train/Val/Test split
 used by ALL FIVE models (DB-HLSTM, CNN, ANN, SVM, RF).

 WHY THIS FILE EXISTS
 ---------------------
 Previously, DB-HLSTM/ANN/CNN split subjects with sklearn's
 train_test_split() at 70/10/20, while SVM/RF split subjects
 with a different custom function at 70/15/15. Same SEED, but
 different method + different fractions => different subjects
 landed in each model's test set. That makes cross-model
 comparison (the whole point of the baseline scripts) invalid,
 since each model would be evaluated on a different exam.

 This script removes that risk entirely: it is run ONCE, before
 any of the 5 training scripts, and writes subject_split.json.
 Every training script then LOADS this file instead of computing
 its own split, guaranteeing all 5 models train/validate/test on
 the exact same subjects.

 Split protocol (matches the supervisor-approved spec):
   - Subject-level split (a subject never appears in >1 split).
   - Stratified by each subject's Class label.
   - Fractions: Train ~70% / Val ~10% / Test ~20%.
   - Two-step sklearn train_test_split (test carved off first,
     then train/val split from the remainder) — same method
     DB-HLSTM/ANN/CNN already used; SVM/RF are updated to match.

 Run this BEFORE any of the 5 baseline/DB-HLSTM scripts:
     python generate_subject_split.py

 Output : /kaggle/working/subject_split.json
============================================================
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH   = '/kaggle/working/EEG_BOLD_Data.csv'
OUT_PATH   = '/kaggle/working/subject_split.json'

SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

print("=" * 60)
print("  GENERATING SHARED SUBJECT-LEVEL SPLIT")
print(f"  Train:{TRAIN_FRAC:.0%}  Val:{VAL_FRAC:.0%}  Test:{TEST_FRAC:.0%}  seed={SEED}")
print("=" * 60)

# Only need Subject + Class columns for this — no need to load
# the full (huge) CSV of raw signal columns.
df = pd.read_csv(CSV_PATH, usecols=['Subject', 'Class'])

# One row per subject, with that subject's Class label (assumed
# constant per subject — AD/Healthy is a subject-level diagnosis).
subj_labels = (df.groupby('Subject')['Class'].first().reset_index()
                 .rename(columns={'Class': 'y'}))

print(f"  Total unique subjects: {len(subj_labels)}")
print(f"  Subjects by class: {subj_labels['y'].value_counts().to_dict()}")

# Split 1: carve off the TEST subjects first.
train_val_subj, test_subj = train_test_split(
    subj_labels, test_size=TEST_FRAC,
    stratify=subj_labels['y'], random_state=SEED)

# Split 2: split the remaining subjects into TRAIN and VAL.
relative_val_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
train_subj, val_subj = train_test_split(
    train_val_subj, test_size=relative_val_frac,
    stratify=train_val_subj['y'], random_state=SEED)

train_ids = sorted(train_subj['Subject'].astype(str).tolist())
val_ids   = sorted(val_subj['Subject'].astype(str).tolist())
test_ids  = sorted(test_subj['Subject'].astype(str).tolist())

# Sanity check — no subject leakage across splits.
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)

n_total = len(train_ids) + len(val_ids) + len(test_ids)
print(f"\n  Subjects — Train: {len(train_ids)} ({len(train_ids)/n_total:.1%})  "
      f"Val: {len(val_ids)} ({len(val_ids)/n_total:.1%})  "
      f"Test: {len(test_ids)} ({len(test_ids)/n_total:.1%})")

def _class_counts(subj_df, ids):
    sub = subj_df[subj_df['Subject'].astype(str).isin(ids)]
    counts = sub['y'].value_counts().to_dict()
    return {'Healthy': int(counts.get(0, 0)), 'AD': int(counts.get(1, 0))}

split_record = {
    'seed': SEED,
    'train_frac': TRAIN_FRAC,
    'val_frac': VAL_FRAC,
    'test_frac': TEST_FRAC,
    'n_total_subjects': n_total,
    'train_subjects': train_ids,
    'val_subjects': val_ids,
    'test_subjects': test_ids,
    'train_class_counts': _class_counts(subj_labels, train_ids),
    'val_class_counts': _class_counts(subj_labels, val_ids),
    'test_class_counts': _class_counts(subj_labels, test_ids),
}

with open(OUT_PATH, 'w') as f:
    json.dump(split_record, f, indent=2)

print(f"  Train class counts: {split_record['train_class_counts']}")
print(f"  Val   class counts: {split_record['val_class_counts']}")
print(f"  Test  class counts: {split_record['test_class_counts']}")
print(f"\n  Saved: {OUT_PATH}")
print("  All 5 model scripts (DB-HLSTM, CNN, ANN, SVM, RF) will now "
      "load this SAME split — run this script first, exactly once, "
      "before training any of them.")
