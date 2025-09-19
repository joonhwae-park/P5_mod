import sys, os, re, json, time, random
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------- Adjust Python Path (same style as test_ml1m_small.py) --------------------
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# -------------------- Imports (matching P5 src layout) --------------------
from transformers import T5Config
from src.tokenization import P5Tokenizer
from src.pretrain_model import P5Pretraining
from src.utils import load_state_dict
from peft import get_peft_model, PromptTuningConfig, PromptEncoderConfig, TaskType

# -------------------- Environment & Defaults --------------------
P5_BACKBONE = os.getenv("P5_BACKBONE", "t5-small")
P5_CKPT = os.getenv("P5_CKPT", os.path.join(project_root, "snap", "Epoch10_64.pth"))

# datamaps: env, then project_root/data/datamaps.json
DATAMAPS_PATH = os.getenv(
    "DATAMAPS_PATH",
    os.path.join(project_root, "data", "datamaps_64.json") if os.path.exists(os.path.join(project_root, "data", "datamaps_64.json"))
    else "/mnt/data/datamaps.json"
)

SVD_DIR = os.getenv("SVD_DIR", os.path.join(project_root, "svd_5core"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


P5_MAX_LEN = int(os.getenv("P5_MAX_LEN", "256"))
P5_GEN_NEW = int(os.getenv("P5_GEN_NEW", "6"))      # max_new_tokens
P5_DROPOUT = float(os.getenv("P5_DROPOUT", "0.1"))
P5_BATCH = int(os.getenv("P5_BATCH", "32"))

# Soft prompt
SOFTPROMPT_STEPS = int(os.getenv("SOFTPROMPT_STEPS", "10"))
SOFTPROMPT_BSZ = int(os.getenv("SOFTPROMPT_BSZ", "8"))
SOFTPROMPT_LR = float(os.getenv("SOFTPROMPT_LR", "5e-4"))
SOFTPROMPT_VTOKENS = int(os.getenv("SOFTPROMPT_VTOKENS", "40"))
SOFTPROMPT_METHOD = os.getenv("SOFTPROMPT_METHOD", "prompt_tuning")  # or 'p_tuning'
HISTORY_MAX_TRAIN = int(os.getenv("HISTORY_MAX_TRAIN", "30"))
TOPK_PER_MODEL = int(os.getenv("TOPK_PER_MODEL", "10"))

SEED = int(os.getenv("SEED", "42"))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -------------------- Utils --------------------
def _first_float(text: str, default: float = -1.0) -> float:
    m = re.search(r"-?\d+(\.\d+)?", text.strip())
    return float(m.group(0)) if m else default

# -------------------- datamaps --------------------
ITEM2ID: Dict[str, str] = {}
ID2ITEM: Dict[str, str] = {}

def load_datamaps(path: str):
    global ITEM2ID, ID2ITEM
    with open(path, "r", encoding="utf-8") as f:
        dm = json.load(f)
    ITEM2ID = dm.get("item2id", {})
    ID2ITEM = {v: k for k, v in ITEM2ID.items()}
    print(f"[datamaps] item2id loaded: {len(ITEM2ID)} from {path}")

def check_history_mapping_coverage(history_ext, ITEM2ID, max_n=50, show=10):
    tot = min(len(history_ext), max_n)
    miss = [(mid, r) for mid, r in history_ext[:tot] if str(mid) not in ITEM2ID]
    cov = 0.0 if tot == 0 else 1.0 - (len(miss) / tot)
    print(f"[datamaps] history coverage: {cov*100:.1f}% ({tot-len(miss)}/{tot})")
    if miss:
        print(f"[datamaps] history unmapped sample (top {show}): {[m for m,_ in miss[:show]]}")
    return cov

def check_top100_mapping_coverage(svd_top100, ITEM2ID, show=10):
    ext_ids = [mid for mid, _ in svd_top100]
    miss = [mid for mid in ext_ids if str(mid) not in ITEM2ID]
    tot = len(ext_ids)
    cov = 0.0 if tot == 0 else 1.0 - (len(miss) / tot)
    print(f"[datamaps] top100 coverage: {cov*100:.1f}% ({tot-len(miss)}/{tot})")
    if miss:
        print(f"[datamaps] top100 unmapped sample (top {show}): {miss[:show]}")
    return cov

def roundtrip_consistency_check(ITEM2ID, ID2ITEM, sample_keys=None, show=10):
    keys = sample_keys or list(ITEM2ID.keys())[:show]
    bad = []
    for k in keys:
        internal = ITEM2ID.get(k)
        back = ID2ITEM.get(internal)
        if back != k:
            bad.append((k, internal, back))
    if bad:
        print("[datamaps] roundtrip mismatches:")
        for k, inter, back in bad:
            print("  ext ->", k, " / int ->", inter, " / back ->", back)
    else:
        print("[datamaps] roundtrip OK on sample")
    return not bad
    
def is_external_id(mid: str) -> bool:
    """??ID(?? MovieID)? datamaps? ?(item2id)? ???? ??ID?? ??."""
    return str(mid) in ITEM2ID

def is_internal_id(mid: str) -> bool:
    """??ID(??? movie_#### ?)? ???(ID2ITEM)? ?? ??."""
    return str(mid) in ID2ITEM

def to_internal_id(mid: str) -> Optional[str]:
    """??ID -> ??ID (?? ??? ???), ??? None"""
    s = str(mid)
    if is_internal_id(s):
        return s
    return ITEM2ID.get(s)

def to_external_id(mid: str) -> Optional[str]:
    """??ID -> ??ID (?? ??? ???), ??? None"""
    s = str(mid)
    if is_external_id(s):
        return s
    return ID2ITEM.get(s)

def detect_id_space(sample_ids: List[str], k: int = 50) -> str:
    """
    CSV? ??? ??ID??(??) ??ID??(??) ??.
    'external' / 'internal' / 'mixed' ??
    """
    sample = [str(x) for x in sample_ids[:k]]
    ext = sum(is_external_id(x) for x in sample)
    inter = sum(is_internal_id(x) for x in sample)
    if ext > 0 and inter == 0:
        return "external"
    if inter > 0 and ext == 0:
        return "internal"
    if ext == 0 and inter == 0:
        # ??? datamaps? ?? ??: ?? external? ??(?? ???? ??? ????)
        return "external"
    return "mixed"
    
def is_svd_item(mid: str) -> bool:
    return str(mid) in IDX

def normalize_history_space(
    history: List[Tuple[str, float]],
    target_space: str
) -> List[Tuple[str, float]]:
    """
    history? target_space('external' or 'internal')? ?? ??.
    ? ??? ID? ?? ??? ???, ?? ??(SVD/P5)?? ???? ???.
    """
    out: List[Tuple[str, float]] = []
    for mid, r in history:
        if target_space == "external":
            new_mid = to_external_id(mid)
            if new_mid is None:
            # ??ID? ?? ????(SVD/ITEM_IDS?? ??), ?? ??ID ??? ??
                if is_external_id(mid) or is_svd_item(mid):
                    new_mid = str(mid)
            else:
                new_mid = None
        else:
            new_mid = to_internal_id(mid)
            new_mid = new_mid if new_mid is not None else (mid if is_internal_id(mid) else None)
        if new_mid is not None:
            out.append((str(new_mid), float(r)))
    return out

def map_history_for_p5(history: List[Tuple[str, float]], max_n: int = HISTORY_MAX_TRAIN) -> List[Tuple[str, float]]:
    """[(ext_mid, rating)] -> [(internal_mid, rating)]"""
    out = []
    for mid, r in history[:max_n]:
        internal = ITEM2ID.get(str(mid))
        if internal is not None:
            out.append((internal, float(r)))
    return out

# -------------------- SVD --------------------
V: Optional[np.ndarray] = None
ITEM_IDS: Optional[np.ndarray] = None 
ITEM_BIAS: Optional[np.ndarray] = None
MU: float = 0.0
REG: float = 0.05
IDX: Dict[str, int] = {}

def load_svd(dir_path: str):
    global V, ITEM_IDS, ITEM_BIAS, MU, REG, IDX
    v_path = os.path.join(dir_path, "V.npy")
    ids_path = os.path.join(dir_path, "item_ids.json")
    bias_path = os.path.join(dir_path, "item_bias.npy")
    mu_path = os.path.join(dir_path, "global_mean.json")
    reg_path = os.path.join(dir_path, "lambda.json")

    V = np.load(v_path).astype("float32")
    with open(ids_path, "r", encoding="utf-8") as f:
        ITEM_IDS = np.array(json.load(f))
    ITEM_BIAS = np.load(bias_path).astype("float32") if os.path.exists(bias_path) else np.zeros(len(ITEM_IDS), dtype="float32")
    MU = json.load(open(mu_path)).get("mu", 0.0) if os.path.exists(mu_path) else 0.0
    REG = json.load(open(reg_path)).get("reg", 0.05) if os.path.exists(reg_path) else 0.05
    IDX = {str(mid): i for i, mid in enumerate(ITEM_IDS)}
    print(f"[svd] V={V.shape}, items={len(ITEM_IDS)}, mu={MU:.3f}, reg={REG}")

def infer_user_vec_svd(history_ext: List[Tuple[str, float]]) -> Optional[np.ndarray]:
    idxs, y = [], []
    for mid, r in history_ext:
        ix = IDX.get(str(mid))
        if ix is None:
            continue
        idxs.append(ix)
        y.append(float(r) - MU - ITEM_BIAS[ix])
    if not idxs:
        return None
    V_R = V[idxs, :]
    y = np.array(y, dtype="float32")
    A = V_R.T @ V_R
    A[np.diag_indices_from(A)] += REG
    p_u = np.linalg.solve(A, V_R.T @ y)
    return p_u.astype("float32")

def score_candidates_svd(p_u: Optional[np.ndarray], candidate_ext_ids: List[str]) -> List[Tuple[str, float]]:
    if p_u is None:
        return [(mid, float(MU)) for mid in candidate_ext_ids]
    pairs = [(mid, IDX.get(str(mid))) for mid in candidate_ext_ids]
    pairs = [(mid, ix) for mid, ix in pairs if ix is not None]
    if not pairs:
        return []
    ix = np.array([ix for _, ix in pairs], dtype=int)
    scores = (V[ix, :] @ p_u).astype("float32")
    return [(mid, float(s)) for (mid, _), s in zip(pairs, scores)]
    
def svd_predict_rating(p_u: Optional[np.ndarray], mid: str) -> Optional[float]:
    """
    SVD? ?? ??? ?? ??? ??.
    ?? ? r - MU - ITEM_BIAS ? ???? ?????,
    ?? ??? MU + ITEM_BIAS[ix] + V[ix] @ p_u ? ??.
    """
    ix = IDX.get(str(mid))
    if (p_u is None) or (ix is None):
        return None
    return float(MU + ITEM_BIAS[ix] + (V[ix, :] @ p_u))

def evaluate_svd_loo(history_ext: List[Tuple[str, float]]) -> Dict[str, object]:
    """
    ? ??? ?? ???? ?? LOO:
      - ? (mid, r)? ??? ?? ???? p_u ??
      - ??? mid? ?? ?? ?? (???? ?? ??)
      - ?? MAE/RMSE ??
    """
    preds: List[float] = []
    trues: List[float] = []
    n = len(history_ext)
    for i in range(n):
        left_mid, left_r = history_ext[i]
        train = history_ext[:i] + history_ext[i+1:]
        if not train:
            continue
        p_u = infer_user_vec_svd(train)
        yhat = svd_predict_rating(p_u, left_mid)
        if yhat is None:
            continue
        preds.append(yhat)
        trues.append(float(left_r))

    if not preds:
        return {"n": 0, "mae": None, "rmse": None, "trues": [], "preds": []}

    mae = mean_absolute_error(trues, preds)
    rmse = mean_squared_error(trues, preds, squared=False)
    return {"n": len(preds), "mae": float(mae), "rmse": float(rmse), "trues": trues, "preds": preds}

# -------------------- P5 --------------------
TOKENIZER, BASE_STATE = None, None

def create_config_eval():
    cfg = T5Config.from_pretrained(P5_BACKBONE)
    cfg.dropout_rate = P5_DROPOUT
    cfg.dropout = P5_DROPOUT
    cfg.attention_dropout = P5_DROPOUT
    cfg.activation_dropout = P5_DROPOUT
    cfg.losses = "rating"
    return cfg

def load_tokenizer_once():
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = P5Tokenizer.from_pretrained(P5_BACKBONE, max_length=P5_MAX_LEN, do_lower_case=False)

def load_base_state_once():
    global BASE_STATE
    if BASE_STATE is None and os.path.exists(P5_CKPT):
        BASE_STATE = load_state_dict(P5_CKPT, DEVICE)
        print(f"[p5] checkpoint cached from {P5_CKPT}")
    elif BASE_STATE is None:
        print(f"[WARN] P5_CKPT not found at {P5_CKPT}. Using randomly initialized weights.")

def create_per_request_base():
    cfg = create_config_eval()
    model = P5Pretraining.from_pretrained(P5_BACKBONE, config=cfg).to(DEVICE)
    model.resize_token_embeddings(TOKENIZER.vocab_size)
    model.tokenizer = TOKENIZER
    model.eval()
    if BASE_STATE is not None:
        _ = model.load_state_dict(BASE_STATE, strict=False)
    return model

def attach_soft_prompt(model, method=SOFTPROMPT_METHOD, vtokens=SOFTPROMPT_VTOKENS):

    model_cpu = model.to("cpu")

    if method == "prompt_tuning":
        cfg = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=vtokens,
            tokenizer_name_or_path=getattr(TOKENIZER, "name_or_path", P5_BACKBONE),
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="rating prediction",
        )
    else:
        cfg = PromptEncoderConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=vtokens,
            encoder_hidden_size=128,
        )
        
    peft_model = get_peft_model(model_cpu, cfg)
    peft_model = peft_model.to(DEVICE)
    
    return peft_model

def make_prompt_inprompt(session_id: str, internal_target: str, history_mapped: List[Tuple[str, float]], hist_k:int=10) -> str:
    htxt = ", ".join([f"movie_{mid}:{r:.1f}" for mid, r in history_mapped[:hist_k]]) or "none"
    return (f"Task: rating prediction.\n"
            f"user_session_{session_id} history: {htxt}\n"
            f"Question: what rating would user_session_{session_id} give to movie_{internal_target} ?\n"
            f"Answer:")

def make_prompt_nohistory(session_id: str, internal_target: str) -> str:
    return (f"Task: rating prediction.\n"
            f"Question: what rating would user_session_{session_id} give to movie_{internal_target} ?\n"
            f"Answer:")

@torch.no_grad()
def p5_predict_batch(model, texts: List[str]) -> List[float]:
    res = []
    for i in range(0, len(texts), P5_BATCH):
        batch = texts[i:i+P5_BATCH]
        enc = TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True, max_length=P5_MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=P5_GEN_NEW, num_beams=1)
        dec = TOKENIZER.batch_decode(out, skip_special_tokens=True)
        res.extend([_first_float(t, default=-1.0) for t in dec])
    return res

def build_training_examples(session_id: str, history_mapped: List[Tuple[str, float]]) -> List[Tuple[str,str]]:
    exs = []
    for internal_mid, rating in history_mapped:
        src = make_prompt_nohistory(session_id, internal_mid)
        tgt = f"{float(rating):.1f}"
        exs.append((src, tgt))
    return exs

def finetune_soft_prompt(model, session_id: str, history_mapped: List[Tuple[str,float]],
                         steps=SOFTPROMPT_STEPS, bsz=SOFTPROMPT_BSZ, lr=SOFTPROMPT_LR):
    exs = build_training_examples(session_id, history_mapped[:HISTORY_MAX_TRAIN])
    if not exs:
        return
    device = next(model.parameters()).device
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    random.shuffle(exs)
    idx, step = 0, 0
    while step < steps:
        batch = exs[idx:idx+bsz]
        if not batch:
            idx = 0
            continue
        idx += bsz
        srcs = [s for s,_ in batch]
        tgts = [t for _,t in batch]
        enc = TOKENIZER(srcs, return_tensors="pt", padding=True, truncation=True, max_length=P5_MAX_LEN).to(device)
        enc = {k: v.to(device) for k, v in enc.items()}
        with TOKENIZER.as_target_tokenizer():
            dec = TOKENIZER(tgts, return_tensors="pt", padding=True, truncation=True, max_length=8).to(device)
        labels = dec["input_ids"].to(device)
        labels = labels.long()
        pad_id = TOKENIZER.pad_token_id
        if pad_id is None:
            pad_id = -100
        labels[labels == pad_id] = -100
        
        optim.zero_grad(set_to_none=True)
        out = model(**enc, labels=labels)
        
        loss = None
        if hasattr(out, "loss"):
            loss = out.loss
        if loss is None and isinstance(out, dict) and "loss" in out:
            loss = out["loss"]
        if loss is None and isinstance(out, dict) and "losses" in out:
            losses_obj = out["losses"]
            if isinstance(losses_obj, dict):
                parts = []
                for v in losses_obj.values():
                    if isinstance(v, torch.Tensor):
                        parts.append(v.mean())
                if parts:
                    loss = torch.stack(parts).mean()
            elif isinstance(losses_obj, (list, tuple)):
                parts = []
                for v in losses_obj:
                    if isinstance(v, torch.Tensor):
                        parts.append(v.mean())
                if parts:
                    loss = torch.stack(parts).mean()

        if isinstance(loss, torch.Tensor) and loss.ndim > 0:
            loss = loss.mean()

        if loss is None:
            raise RuntimeError("Model did not return a usable loss (checked .loss / ['loss'] / ['losses']).")

        loss.backward()
        torch.nn.utils.clip_grad_norm_( [p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        step += 1
    model.eval()

# -------------------- Data Load (64 users CSV) --------------------
def load_csv_groupby_users(csv_path: str) -> Dict[str, List[Tuple[str, float]]]:
    df = pd.read_csv(csv_path)
    need = {"UserID","MovieID","Rating"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must have columns {need}")
    users: Dict[str, List[Tuple[str,float]]] = {}
    for uid, g in df.groupby("UserID"):
        seq = [(str(r["MovieID"]), float(r["Rating"])) for _, r in g.iterrows()]
        users[str(uid)] = seq
    return users

# -------------------- Main: 1-user at a time; aggregate for 64 users --------------------
def main(csv_path: str, out_prefix: Optional[str] = None):
    load_datamaps(DATAMAPS_PATH)
    load_svd(SVD_DIR)
    
    roundtrip_consistency_check(ITEM2ID, ID2ITEM)
    
    load_tokenizer_once()
    load_base_state_once()

    users = load_csv_groupby_users(csv_path)
    candidate_pool = list(map(str, ITEM_IDS))

    rows_time, rows_top = [], []
    
    y_true_all, y_pred_in_all, y_pred_soft_all, y_pred_svd_all = [], [], [], []
    
    svdloo_trues_all, svdloo_preds_all = [], []

    #for uid, history_ext in users.items():
    for uid, history_any in users.items():
    
        csv_space = detect_id_space([mid for mid, _ in history_any])
        train_any = history_any[:-1]
        test_mid_any, test_rating = history_any[-1]
        
        train_ext = normalize_history_space(train_any, target_space="external")
        test_mid_ext = to_external_id(test_mid_any) or (test_mid_any if is_external_id(test_mid_any) else None)
        
        train_int = normalize_history_space(train_any, target_space="internal")
        test_internal = to_internal_id(test_mid_any) or (test_mid_any if is_internal_id(test_mid_any) else None)
        
        check_history_mapping_coverage(train_ext, ITEM2ID)
    
        #train_hist = history_ext[:-1]
        #test_mid, test_rating = history_ext[-1]
        #test_internal = ITEM2ID.get(str(test_mid))
        
        #check_history_mapping_coverage(train_hist, ITEM2ID)
        
        print(f"\n=== user {uid} (csv_space={csv_space}) ===")
        
        base_in = None
        base_soft = None
        per_user = None
    
        print(f"\n=== user {uid} ===")
        

        # ----- SVD Top-100 -----
        t0 = time.perf_counter()
        #p_u = infer_user_vec_svd(train_hist)
        #p_u = infer_user_vec_svd(train_ext)
        train_svd = [(ID2ITEM.get(str(m), str(m)), float(r)) for m, r in train_any if (ID2ITEM.get(str(m), str(m)) in IDX)]
        p_u = infer_user_vec_svd(train_svd)
        svd_scored_all = score_candidates_svd(p_u, candidate_pool)
        svd_top100 = sorted(svd_scored_all, key=lambda x: x[1], reverse=True)[:100]
        #svd_test_pred = score_candidates_svd(p_u, [str(test_mid)])[0][1]
        
        check_top100_mapping_coverage(svd_top100, ITEM2ID)
        
        t1 = time.perf_counter()
        svd_ms = (t1 - t0) * 1000.0
        
        history_ext_full = normalize_history_space(history_any, target_space="external")
        svd_loo = evaluate_svd_loo(history_ext_full)
        
        if svd_loo["n"] > 0:
            svdloo_trues_all.extend(svd_loo["trues"])
            svdloo_preds_all.extend(svd_loo["preds"])        

        #train_map = map_history_for_p5(train_hist, max_n=HISTORY_MAX_TRAIN)
        
        pairs = []
        for ext_mid, _ in svd_top100:
            internal = ITEM2ID.get(str(ext_mid))
            if internal is not None:
                pairs.append((ext_mid, internal))
        mapped_ids = [internal for _, internal in pairs]

        # ----- P5 In-Prompt -----
        t2 = time.perf_counter()
        if test_internal is not None:
            base_in = create_per_request_base()
            #text_in = make_prompt_inprompt(uid, test_internal, train_map, hist_k=10)
            text_in = make_prompt_inprompt(uid, test_internal, train_int, hist_k=10)
            in_score = p5_predict_batch(base_in, [text_in])[0]
        else:
            in_score = -1.0
        t3 = time.perf_counter()
        p5_in_ms = (t3 - t2) * 1000.0

        # ----- P5 Soft-Prompt (10 step) -----
        t4 = time.perf_counter()
        #if train_map and (test_internal is not None):
        if train_int and (test_internal is not None):
            base_soft = create_per_request_base()
            per_user = attach_soft_prompt(base_soft)
            #finetune_soft_prompt(per_user, uid, train_map)
            finetune_soft_prompt(per_user, uid, train_int)
            text_soft = make_prompt_nohistory(uid, test_internal)
            soft_score = p5_predict_batch(per_user, [text_soft])[0]
        else:
            soft_score = -1.0
        t5 = time.perf_counter()
        p5_soft_ms = (t5 - t4) * 1000.0

        rows_time.append({
            "UserID": uid,
            "svd_ms": svd_ms,
            "p5_inprompt_ms": p5_in_ms,
            "p5_softprompt_ms": p5_soft_ms,
            "svd_top100_size": len(svd_top100),
            "p5_candidates_used": len(mapped_ids),
            "svd_loo_n": svd_loo["n"],
            "svd_loo_mae": svd_loo["mae"],
            "svd_loo_rmse": svd_loo["rmse"],
        })
        for rank, (mid, sc) in enumerate(sorted(svd_top100, key=lambda x:x[1], reverse=True)[:TOPK_PER_MODEL], 1):
            rows_top.append({"UserID": uid, "model": "svd", "rank": rank, "MovieID": mid, "score": float(sc)})

        print(f"  SVD: {svd_ms:.1f} ms | P5 In-Prompt: {p5_in_ms:.1f} ms | P5 Soft-Prompt: {p5_soft_ms:.1f} ms")
        
        if test_internal is not None and p_u is not None:
            y_true_all.append(float(test_rating))
            #y_pred_svd_all.append(float(svd_test_pred))
            if in_score >= 0:
                y_pred_in_all.append(float(in_score))
            else:
                pass
            if soft_score >= 0:
                y_pred_soft_all.append(float(soft_score))

        if DEVICE == "cuda":
            if base_in is not None:
                del base_in
            if per_user is not None:
                del per_user
            if base_soft is not None:
                del base_soft
            torch.cuda.empty_cache()
            


    df_time = pd.DataFrame(rows_time)
    print("\n=== RUNTIME SUMMARY (per user) ===")
    if not df_time.empty:
        print(df_time.describe()[["svd_ms", "p5_inprompt_ms", "p5_softprompt_ms"]])
        
    def _report(name, y_true, y_pred):
        atrue, apred = np.array(y_true), np.array(y_pred)
        mask = ~np.isnan(apred)
        atrue, apred = atrue[mask], apred[mask]
        if len(atrue) == 0:
            print(f"{name}: insufficient pairs")
            return
        rmse = np.sqrt(mean_squared_error(atrue, apred))
        mae = mean_absolute_error(atrue, apred)
        print(f"{name}: RMSE={rmse:.4f} | MAE={mae:.4f} | N={len(atrue)}")
        
    print("\n=== FINAL METRICS (LOO over users) ===")
    #_report("SVD       ", y_true_all, y_pred_svd_all)
    _report("In-Prompt ", y_true_all[:len(y_pred_in_all)], y_pred_in_all)
    _report("Soft-Prompt", y_true_all[:len(y_pred_soft_all)], y_pred_soft_all)
    
    if out_prefix and len(svdloo_trues_all) > 0:
        overall_mae = mean_absolute_error(svdloo_trues_all, svdloo_preds_all)
        overall_rmse = mean_squared_error(svdloo_trues_all, svdloo_preds_all, squared=False)
        print(f"[SVD LOO] overall n={len(svdloo_trues_all)} | MAE={overall_mae:.4f} | RMSE={overall_rmse:.4f}")


    if out_prefix:
        df_time.to_csv(out_prefix.replace(".csv","_times.csv"), index=False)
        pd.DataFrame(rows_top).to_csv(out_prefix.replace(".csv","_tops.csv"), index=False)
        print(f"[saved] timings -> {out_prefix.replace('.csv','_times.csv')}")
        print(f"[saved] tops    -> {out_prefix.replace('.csv','_tops.csv')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="ratings csv with columns: user_id, item_id, rating")
    ap.add_argument("--out", default=None, help="optional prefix to save timings and top lists")
    args = ap.parse_args()
    main(args.csv, out_prefix=args.out)
