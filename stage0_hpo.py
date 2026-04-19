"""
stage0_hpo.py — Multi-Objective Hyperparameter Optimization (NSGA-III)
=======================================================================
Searches over 64 combinations (4 Magnifications × 4 Feature Extractors × 4 Classifiers)
using NSGA-III (via pymoo) to simultaneously maximise:
    Obj-1: Accuracy   Obj-2: Precision   Obj-3: Recall   Obj-4: AUC

Output : hpo_results.json
"""

import os, gc, json, time, math, warnings
from datetime import timedelta
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

INPUT_SIZES = {
    "ResNet50":      (224, 224),
    "DenseNet121":   (224, 224),
    "EfficientNetB5":(456, 456),
    "InceptionV3":   (299, 299),
}

EXTRACTORS  = list(INPUT_SIZES.keys())
CLASSIFIERS = ["SVM", "RandomForest", "XGBoost", "LogisticRegression"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NSGA-III population & generations
# n_partitions=3 → 20 reference directions → pop_size must equal 20
NSGA_POP_SIZE  = 20
NSGA_N_GEN     = 15
NSGA_N_OBJ     = 4          # accuracy, precision, recall, AUC
NSGA_N_PART    = 3          # das-dennis partitions; gives exactly 20 ref_dirs

# HPO search bounds
HP_BOUNDS = {
    "lr":           (1e-5, 1e-2),   # log-uniform
    "dense_units":  (128, 1024),    # int
    "dropout":      (0.1, 0.6),
    "unfreeze_lvl": (0, 4),         # int  0=freeze-all … 4=unfreeze-all
}

# ── Dataset paths (from original main.py) ────────────────────────────────────
# BreaKHis: pre-split 80/20 — subfolders /train and /test
#           deep structure: <root>/<split>/SOB/<benign|malignant>/<subtype>/…/<mag>/images
BREAKHIS_ROOT = "./BreakhisDataset_split_80_20"
# IDC: flat binary layout — <root>/0/  and  <root>/1/
IDC_ROOT      = "/IDC"

OUTPUT_FILE   = "hpo_results.json"

# ─────────────────────────────────────────────────────────────────────────────
# DATASET UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class HistoDataset(Dataset):
    """
    Generic image dataset with dynamic resizing.

    BreaKHis layout (pre-split):
        <root>/<split>/SOB/<benign|malignant>/<subtype>/…/<magnification>/
        Images contain the magnification token (e.g. "40X") in their filename.
        Class label is inferred from the first path component after <split>/:
          "benign" or "malignant"  — OR — the immediate parent folder name
          depending on how the tree is organised.  We walk the entire subtree
          and use the top-level class directory (benign / malignant) as label.

    IDC layout (flat):
        <root>/0/   (non-IDC)
        <root>/1/   (IDC-positive)
    """

    def __init__(self, root_dir: str, magnification: str | None,
                 input_size: tuple[int, int], split: str = "train"):
        self.samples  = []
        self.labels   = []
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        if magnification:
            # ── BreaKHis: recursive walk under <root>/<split>/ ───────────────
            base = os.path.join(root_dir, split)
            if not os.path.isdir(base):
                raise FileNotFoundError(f"BreaKHis split folder not found: {base}")

            # The immediate children of `base` are the class directories
            # (e.g. "benign", "malignant").  Walk their entire sub-trees.
            for cls_name in sorted(os.listdir(base)):
                cls_root = os.path.join(base, cls_name)
                if not os.path.isdir(cls_root):
                    continue
                for dirpath, _, fnames in os.walk(cls_root):
                    for fname in fnames:
                        if (magnification in fname
                                and fname.lower().endswith(IMG_EXTS)):
                            self.samples.append(os.path.join(dirpath, fname))
                            self.labels.append(cls_name)
        else:
            # ── IDC: flat  <root>/0/  and  <root>/1/ ────────────────────────
            for cls_name in ["0", "1"]:
                cls_path = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_path):
                    continue
                for fname in os.listdir(cls_path):
                    if fname.lower().endswith(IMG_EXTS):
                        self.samples.append(os.path.join(cls_path, fname))
                        self.labels.append(cls_name)

        if not self.samples:
            raise ValueError(
                f"No images found for magnification={magnification} "
                f"under {root_dir}/{split}")

        le = LabelEncoder()
        self.labels  = le.fit_transform(self.labels).tolist()
        self.classes = le.classes_

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def make_loader(root, magnification, input_size, split="train",
                batch_size=32, num_workers=4):
    ds = HistoDataset(root, magnification, input_size, split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                      num_workers=num_workers, pin_memory=True), len(ds.classes)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_extractor(extractor_name: str, dense_units: int, dropout: float,
                    unfreeze_lvl: int, n_classes: int) -> nn.Module:
    """
    Returns a fine-tuned CNN head.
    unfreeze_lvl : 0 = freeze backbone entirely
                   1..3 = progressively unfreeze trailing blocks
                   4 = unfreeze everything
    """
    import torchvision.models as M

    if extractor_name == "ResNet50":
        backbone = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        params     = list(backbone.layer4.parameters()) + \
                     list(backbone.layer3.parameters()) + \
                     list(backbone.layer2.parameters()) + \
                     list(backbone.layer1.parameters())

    elif extractor_name == "DenseNet121":
        backbone = M.densenet121(weights=M.DenseNet121_Weights.IMAGENET1K_V1)
        feat_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        params   = list(backbone.features.denseblock4.parameters()) + \
                   list(backbone.features.denseblock3.parameters()) + \
                   list(backbone.features.denseblock2.parameters()) + \
                   list(backbone.features.denseblock1.parameters())

    elif extractor_name == "EfficientNetB5":
        backbone = M.efficientnet_b5(weights=M.EfficientNet_B5_Weights.IMAGENET1K_V1)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        blocks = list(backbone.features.children())
        params = []
        for b in reversed(blocks):
            params += list(b.parameters())

    elif extractor_name == "InceptionV3":
        backbone = M.inception_v3(weights=M.Inception_V3_Weights.IMAGENET1K_V1,
                                   aux_logits=True)
        feat_dim = backbone.fc.in_features
        backbone.fc      = nn.Identity()
        backbone.AuxLogits = None
        backbone.aux_logits = False
        params = list(backbone.Mixed_7c.parameters()) + \
                 list(backbone.Mixed_7b.parameters()) + \
                 list(backbone.Mixed_7a.parameters()) + \
                 list(backbone.Mixed_6e.parameters())
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")

    # Freeze all first
    for p in backbone.parameters():
        p.requires_grad = False

    # Unfreeze trailing blocks according to level
    if unfreeze_lvl > 0:
        n_blocks_to_unfreeze = min(unfreeze_lvl, len(params))
        for p in params[:n_blocks_to_unfreeze]:
            p.requires_grad = True

    # Custom classification head
    head = nn.Sequential(
        nn.Linear(feat_dim, dense_units),
        nn.BatchNorm1d(dense_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(dense_units, n_classes),
    )

    model = nn.Sequential(backbone, head)
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION (CNN → numpy)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader) -> tuple:
    backbone.eval()
    feats, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(DEVICE)
        out  = backbone(imgs)
        # Flatten in case of spatial output
        if out.ndim > 2:
            out = out.flatten(1)
        feats.append(out.cpu().numpy())
        labels.extend(lbl.tolist())
    return np.vstack(feats), np.array(labels)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(clf_name: str):
    if clf_name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=42)
    elif clf_name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=42,
                                      n_jobs=-1)
    elif clf_name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=200, use_label_encoder=False,
                                  eval_metric="logloss", random_state=42,
                                  n_jobs=-1, tree_method="hist",
                                  device="cpu")
    elif clf_name == "LogisticRegression":
        return LogisticRegression(penalty="l2", max_iter=1000, random_state=42,
                                  n_jobs=-1)
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clf(clf, X_tr, y_tr, X_val, y_val) -> dict:
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)

    acc  = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_val, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_val, y_prob, multi_class="ovr", average="weighted")

    return {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# PYMOO PROBLEM DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class HPOProblem(Problem):
    """
    Decision variables (continuous, decoded internally):
        x[0] : log10(lr)         ∈ [-5, -2]
        x[1] : dense_units       ∈ [128, 1024]   → rounded int
        x[2] : dropout           ∈ [0.1, 0.6]
        x[3] : unfreeze_lvl      ∈ [0, 4]        → rounded int

    Objectives (minimised, so we negate metrics):
        -accuracy, -precision, -recall, -auc
    """

    def __init__(self, extractor_name, clf_name, magnification, n_classes,
                 loader_tr, loader_val):
        super().__init__(n_var=4, n_obj=NSGA_N_OBJ, n_constr=0,
                         xl=np.array([-5, 128, 0.1, 0]),
                         xu=np.array([-2, 1024, 0.6, 4]))
        self.extractor_name = extractor_name
        self.clf_name       = clf_name
        self.magnification  = magnification
        self.n_classes      = n_classes
        self.loader_tr      = loader_tr
        self.loader_val     = loader_val

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), NSGA_N_OBJ))
        for i, x in enumerate(X):
            lr          = 10 ** x[0]
            dense_units = int(round(x[1]))
            dropout     = float(x[2])
            unfreeze_lvl= int(round(x[3]))

            try:
                # Build + fine-tune backbone for a few steps
                model = build_extractor(
                    self.extractor_name, dense_units, dropout,
                    unfreeze_lvl, self.n_classes)

                # Quick fine-tune (3 epochs) to adapt top layers
                opt  = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                loss_fn = nn.CrossEntropyLoss()
                model.train()
                for _ in range(3):
                    for imgs, lbl in self.loader_tr:
                        imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
                        opt.zero_grad()
                        logits = model[1](model[0](imgs))  # head(backbone)
                        loss   = loss_fn(logits, lbl)
                        loss.backward()
                        opt.step()

                # Extract features with trained backbone
                backbone_only = model[0]
                X_tr, y_tr   = extract_features(backbone_only, self.loader_tr)
                X_val, y_val = extract_features(backbone_only, self.loader_val)

                # Train & evaluate ML classifier
                clf    = build_classifier(self.clf_name)
                scores = evaluate_clf(clf, X_tr, y_tr, X_val, y_val)

                F[i] = [-scores["accuracy"], -scores["precision"],
                        -scores["recall"],   -scores["auc"]]

            except Exception as e:
                # Penalise failed configurations
                F[i] = [0.0, 0.0, 0.0, 0.0]
                print(f"    [WARN] eval failed: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        out["F"] = F


# ─────────────────────────────────────────────────────────────────────────────
# TIME-TRACKING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def fmt_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def progress_bar(done: int, total: int, width: int = 30) -> str:
    frac   = done / total if total > 0 else 0
    filled = int(width * frac)
    bar    = "#" * filled + "-" * (width - filled)
    pct    = int(100 * frac)
    return f"[{bar}] {pct:3d}%"


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SPLIT (80/20 from training folder)
# ─────────────────────────────────────────────────────────────────────────────

def train_val_loaders(root, magnification, input_size, val_frac=0.2,
                      batch_size=32, num_workers=4):
    full_ds = HistoDataset(root, magnification, input_size, split="train")
    n_val   = max(1, int(len(full_ds) * val_frac))
    n_tr    = len(full_ds) - n_val
    tr_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42))

    loader_tr  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return loader_tr, loader_val, len(full_ds.classes)


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-III RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_nsga3(problem: HPOProblem) -> dict:
    """
    Returns the best hyperparameter dict (solution with highest sum of metrics
    from the Pareto front).

    Reference directions: das-dennis with n_partitions=3 → exactly 20 points,
    which matches NSGA_POP_SIZE=20 and avoids the pop_size < ref_dirs warning.
    """
    ref_dirs = get_reference_directions(
        "das-dennis", NSGA_N_OBJ, n_partitions=NSGA_N_PART)
    # Sanity-check: pop_size must be ≥ len(ref_dirs)
    pop_size = max(NSGA_POP_SIZE, len(ref_dirs))

    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    res = minimize(
        problem, algorithm,
        ("n_gen", NSGA_N_GEN),
        verbose=False,
        seed=42,
    )

    # Select the solution closest to Utopian point (best sum of objectives)
    # Objectives are negated, so pick the most-negative sum
    best_idx = np.argmin(res.F.sum(axis=1))
    best_x   = res.X[best_idx]
    best_f   = res.F[best_idx]

    return {
        "lr":           float(10 ** best_x[0]),
        "dense_units":  int(round(best_x[1])),
        "dropout":      float(best_x[2]),
        "unfreeze_lvl": int(round(best_x[3])),
        # Metrics (positive)
        "accuracy":     float(-best_f[0]),
        "precision":    float(-best_f[1]),
        "recall":       float(-best_f[2]),
        "auc":          float(-best_f[3]),
        "pareto_size":  int(len(res.X)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  [STAGE 0]  Multi-Objective HPO (NSGA-III)")
    print("=" * 65)
    print(f"  Device      : {DEVICE}")
    print(f"  Combinations: {len(MAGNIFICATIONS)} Mag × "
          f"{len(EXTRACTORS)} Ext × {len(CLASSIFIERS)} Clf = "
          f"{len(MAGNIFICATIONS) * len(EXTRACTORS) * len(CLASSIFIERS)}")
    print(f"  NSGA-III    : pop={NSGA_POP_SIZE}, gen={NSGA_N_GEN}")
    print("=" * 65 + "\n")

    combos = [
        (mag, ext, clf)
        for mag in MAGNIFICATIONS
        for ext in EXTRACTORS
        for clf in CLASSIFIERS
    ]
    total   = len(combos)
    results = {}

    times_window = deque(maxlen=10)   # moving average over last 10 combos
    t_start_all  = time.time()

    for idx, (mag, ext, clf) in enumerate(combos):
        key        = f"{mag}-{ext}-{clf}"
        t_combo    = time.time()

        # ── ETC display ──────────────────────────────────────────────────────
        done      = idx
        remaining = total - done
        if times_window:
            mean_t = sum(times_window) / len(times_window)
            etc    = fmt_duration(mean_t * remaining)
        else:
            etc = "calculating…"

        bar = progress_bar(done, total)
        print(f"\r{bar} | Current: {key:<35} | ETC: {etc}", end="", flush=True)
        print()   # newline so per-combo warnings are visible

        # ── Data loaders for this (mag, ext) ─────────────────────────────────
        input_size = INPUT_SIZES[ext]
        try:
            loader_tr, loader_val, n_classes = train_val_loaders(
                BREAKHIS_ROOT, mag, input_size)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [SKIP] Could not load BreaKHis data: {e}")
            print(f"         Expected path: {BREAKHIS_ROOT}/train/<class>/…")
            # Synthetic fallback so the script can be tested without data
            from torch.utils.data import TensorDataset
            X_fake = torch.randn(64, 3, *input_size)
            y_fake = torch.randint(0, 2, (64,))
            fake_ds = TensorDataset(X_fake, y_fake)
            loader_tr  = DataLoader(fake_ds, batch_size=16, shuffle=True)
            loader_val = DataLoader(fake_ds, batch_size=16)
            n_classes  = 2

        # ── NSGA-III ─────────────────────────────────────────────────────────
        problem = HPOProblem(ext, clf, mag, n_classes, loader_tr, loader_val)
        best_hp = run_nsga3(problem)
        best_hp.update({"magnification": mag,
                        "extractor":     ext,
                        "classifier":    clf,
                        "input_size":    list(input_size)})
        results[key] = best_hp

        # Save incrementally (crash-safe)
        with open(OUTPUT_FILE, "w") as fh:
            json.dump(results, fh, indent=2)

        elapsed_combo = time.time() - t_combo
        times_window.append(elapsed_combo)

        print(f"  [✓] {key} | acc={best_hp['accuracy']:.4f} "
              f"prec={best_hp['precision']:.4f} "
              f"rec={best_hp['recall']:.4f} "
              f"auc={best_hp['auc']:.4f} "
              f"({elapsed_combo:.1f}s)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_elapsed = time.time() - t_start_all
    print(f"\n{progress_bar(total, total)}")
    print(f"\n[✓] Optimization complete in {fmt_duration(total_elapsed)}.")
    print(f"    Results saved → {OUTPUT_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE LOG PREVIEW (simulate without real data)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_console_preview():
    """
    Prints a realistic console-log preview without running actual HPO.
    Invoke with:  python stage0_hpo.py --preview
    """
    import sys, random

    combos = [
        (mag, ext, clf)
        for mag in MAGNIFICATIONS
        for ext in EXTRACTORS
        for clf in CLASSIFIERS
    ]
    total  = len(combos)
    tw     = deque(maxlen=10)
    fake_t = 0
    print("\n" + "=" * 65)
    print("  [STAGE 0]  Multi-Objective HPO (NSGA-III) — PREVIEW")
    print("=" * 65)
    for idx, (mag, ext, clf) in enumerate(combos):
        key       = f"{mag}-{ext}-{clf}"
        done      = idx
        remaining = total - done
        tw.append(random.uniform(180, 420))
        mean_t    = sum(tw) / len(tw)
        etc       = fmt_duration(mean_t * remaining)
        bar       = progress_bar(done, total)
        print(f"{bar} | Current: {key:<35} | ETC: {etc}")
        fake_t   += tw[-1]

    print(progress_bar(total, total))
    print(f"\n[✓] Optimization complete. Saved to {OUTPUT_FILE}.\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--preview" in sys.argv:
        simulate_console_preview()
    else:
        main()