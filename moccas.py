from __future__ import annotations
import os, glob, math, time, argparse, json, warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable, Dict, Optional
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

OBJ_COLS = ["dock","solubility","sa","qed","similarity_to_topK"]

# ----------------------- Feature loader -----------------------
class FeatureLoader:
    def __init__(self, parts_dir: str, pattern: str = "feats_part_*_final.npz"):
        self.parts_dir = parts_dir
        self.files = sorted(glob.glob(os.path.join(parts_dir, pattern)))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No *_final.npz shards found in {parts_dir}.")
        meta_path = os.path.join(parts_dir, "meta_final.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing {meta_path}")
        meta = np.load(meta_path, allow_pickle=True)
        self.names = meta["names"]
        first = np.load(self.files[0])["X"]
        self.d = first.shape[1]
        self.starts, self.sizes = [], []
        n = 0
        for fp in self.files:
            Xi = np.load(fp)["X"]
            self.starts.append(n)
            self.sizes.append(Xi.shape[0])
            n += Xi.shape[0]
        self.n = n

    def num_rows(self) -> int: return self.n
    def dim(self) -> int: return self.d

    def iter_chunks(self):
        for start, fp in zip(self.starts, self.files):
            Xi = np.load(fp)["X"]
            yield start, Xi

    def get_rows_by_indices(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        out = np.empty((idx.shape[0], self.d), dtype=np.float32)
        order = np.argsort(idx); idx_sorted = idx[order]
        filled = 0; ptr = 0
        for start, size, fp in zip(self.starts, self.sizes, self.files):
            end = start + size
            while ptr < idx_sorted.size and idx_sorted[ptr] < start: ptr += 1
            j = ptr
            while j < idx_sorted.size and idx_sorted[j] < end: j += 1
            if j > ptr:
                Xi = np.load(fp)["X"]
                take = idx_sorted[ptr:j] - start
                block = Xi[take]
                out[filled:filled+block.shape[0]] = block
                idx_sorted[ptr:j] = -1
                filled += block.shape[0]
                ptr = j
            if filled == idx.shape[0]: break
        inv = np.argsort(order)
        return out[inv]

# ----------------------- Objectives & S -----------------------
def load_objectives(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=["smiles"] + OBJ_COLS)
    return df

def build_S_outputs(df: pd.DataFrame, tau: Dict[str, float]):
    mask = (df["dock"]>=tau["dock"]) & (df["solubility"]>=tau["solubility"]) & \
           (df["sa"]>=tau["sa"]) & (df["qed"]>=tau["qed"]) & \
           (df["similarity_to_topK"]>=tau["similarity_to_topK"])
    Z = df[OBJ_COLS].values.astype(np.float32)
    S_outputs = Z[mask.values]
    return S_outputs, mask.values

# ----------------------- GP wrapper -----------------------
class GPWrapper:
    def __init__(self, d: int, noise_init: float = 1e-3, random_state: int = 0, suppress_warnings: bool = True):
        self.d = d
        self.random_state = random_state
        self.kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=noise_init, noise_level_bounds=(1e-6, 1e-1))
        self._kernel_fixed = None
        self._gpr = None
        self.suppress_warnings = suppress_warnings

    def prefit_hyperparams(self, X_sub: np.ndarray, y_sub: np.ndarray) -> None:
        if self.suppress_warnings:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, normalize_y=True,
                                       random_state=self.random_state, n_restarts_optimizer=1)
        gpr.fit(X_sub, y_sub)
        self._kernel_fixed = gpr.kernel_
        self._gpr = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        gpr = GaussianProcessRegressor(kernel=self._kernel_fixed, alpha=0.0, normalize_y=True, optimizer=None)
        gpr.fit(X, y)
        self._gpr = gpr

    def predict(self, X: np.ndarray, batch: int = 100000):
        mus, stds = [], []
        n = X.shape[0]
        for i in range(0, n, batch):
            Xi = X[i:i+batch]
            mu, std = self._gpr.predict(Xi, return_std=True)
            mus.append(mu.astype(np.float32)); stds.append(std.astype(np.float32))
        return np.concatenate(mus, axis=0), np.concatenate(stds, axis=0)

# ----------------------- Helpers -----------------------
def ucb_scores_for_objective(gp: GPWrapper, beta_t: float, loader: FeatureLoader,
                             used_mask: np.ndarray, top_per_shard: int,
                             consider_idx: Optional[np.ndarray] = None,
                             scan_cap_per_shard: Optional[int] = None) -> np.ndarray:
    tops = []
    rng = np.random.default_rng()
    for start, X in loader.iter_chunks():
        n = X.shape[0]
        shard_unused = np.where(~used_mask[start:start+n])[0]
        if consider_idx is not None:
            g = consider_idx[(consider_idx >= start) & (consider_idx < start+n)] - start
            if g.size == 0: 
                continue
            mask = np.zeros(n, dtype=bool); mask[shard_unused] = True; mask[g] &= True
            local = np.where(mask)[0]
        else:
            local = shard_unused
        if local.size == 0: continue
        if scan_cap_per_shard is not None and local.size > scan_cap_per_shard:
            take = rng.choice(local, size=scan_cap_per_shard, replace=False)
            local = np.sort(take)
        Xu = X[local]
        mu, std = gp.predict(Xu, batch=max(50000, 10000))
        ucb = mu + math.sqrt(beta_t) * std
        if Xu.shape[0] > top_per_shard:
            idx_local = np.argpartition(ucb, -top_per_shard)[-top_per_shard:]
        else:
            idx_local = np.arange(Xu.shape[0], dtype=int)
        shard_indices = local[idx_local] + start
        tops.append(shard_indices)
    if len(tops)==0: return np.array([], dtype=int)
    return np.unique(np.concatenate(tops).astype(int))

def build_shortlist(gps: List[GPWrapper], beta_t: float, loader: FeatureLoader,
                    used_mask: np.ndarray, topk_per_obj: int, random_k: int,
                    consider_idx: Optional[np.ndarray], scan_cap_per_shard: Optional[int]) -> np.ndarray:
    per_shard = max(1, topk_per_obj // max(1, sum(1 for _ in loader.iter_chunks())))
    cand_sets = []
    for gp in gps:
        idxs = ucb_scores_for_objective(gp, beta_t, loader, used_mask, per_shard, consider_idx, scan_cap_per_shard)
        cand_sets.append(idxs)
    pool = np.unique(np.concatenate(cand_sets)) if len(cand_sets)>0 else np.array([], dtype=int)
    
    unused = np.where(~used_mask)[0]
    if unused.size > 0 and random_k > 0:
        rng = np.random.default_rng()
        if consider_idx is not None:
            unused = np.intersect1d(unused, consider_idx)
        if unused.size > 0:
            rand = rng.choice(unused, size=min(random_k, unused.size), replace=False)
            pool = np.unique(np.concatenate([pool, rand]))
    return pool

def optimistic_outputs(gps: List[GPWrapper], X: np.ndarray, beta_t: float) -> np.ndarray:
    Us = []
    for gp in gps:
        mu, std = gp.predict(X, batch=max(50000, 10000))
        Us.append(mu + math.sqrt(beta_t) * std)
    return np.vstack(Us).T

def farthest_output_tiebreak(U: np.ndarray, obs_outputs: List[np.ndarray]) -> np.ndarray:
    if len(obs_outputs) == 0:
        return np.full(U.shape[0], np.inf, dtype=np.float32)
    Y = np.vstack(obs_outputs)
    tree = KDTree(Y, metric='euclidean')
    dists, _ = tree.query(U, k=1, return_distance=True)
    return dists[:,0].astype(np.float32)

def update_covered_with_observation(y_true: np.ndarray, S_tree: BallTree, covered_mask: np.ndarray, r: float) -> None:
    idxs = S_tree.query_radius(y_true[None, :], r=r)[0]
    covered_mask[idxs] = True

def coverage_recall(covered_mask: np.ndarray) -> float:
    return float(np.count_nonzero(covered_mask)) / float(covered_mask.size)

def fill_distance(obs_outputs_in_S: np.ndarray, S_outputs: np.ndarray) -> float:
    if obs_outputs_in_S.shape[0] == 0:
        return float("inf")
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean').fit(obs_outputs_in_S)
    dists, _ = nbrs.kneighbors(S_outputs, n_neighbors=1, return_distance=True)
    return float(dists.max())

# --------- Coverage gain: soft kernel + feasible cap / k-means reps ---------
def soft_coverage_gain(U: np.ndarray, S_tree: BallTree, covered_mask: np.ndarray, r: float) -> np.ndarray:
    ind_list, dist_list = S_tree.query_radius(U, r=r, return_distance=True, sort_results=True)
    gains = np.zeros(U.shape[0], dtype=np.float32)
    for i, (idxs, dists) in enumerate(zip(ind_list, dist_list)):
        if idxs.size == 0: 
            continue
        mask = ~covered_mask[idxs]
        if not np.any(mask):
            continue
        dd = dists[mask]
        t = dd / max(r, 1e-8)
        w = 1.0 - (t*t)
        w[w < 0] = 0.0
        gains[i] = float(w.sum())
    return gains

# ----------------------- Main MOC-CAS loop -----------------------
def prefit_all_gps(loader: FeatureLoader, df: pd.DataFrame, seed: int, prefit_n: int = 200) -> List[GPWrapper]:
    rng = np.random.default_rng(seed)
    N = loader.num_rows()
    idx = rng.choice(N, size=min(prefit_n, N), replace=False)
    X_sub = loader.get_rows_by_indices(idx)
    gps = []
    for k, col in enumerate(OBJ_COLS):
        y_sub = df[col].values[idx].astype(np.float32)
        gp = GPWrapper(d=loader.dim(), random_state=seed+31*k, suppress_warnings=True)
        gp.prefit_hyperparams(X_sub, y_sub)
        gps.append(gp)
    return gps

def run_moccas(df, loader, tau, r, rounds, init_idx, seed=123,
                       beta0=2.0, beta_anneal_t=150, beta_anneal_factor=0.7,
                       shortlist_k=3000, random_k=500, prefit_n=200,
                       candidate_cap=80000, scan_cap_per_shard=5000,
                       cover_cap=800, k_div=40, lambda_repulse=0.2,
                       print_every=5):
    rng = np.random.default_rng(seed)
    S_outputs, S_mask = build_S_outputs(df, tau)
    used = np.zeros(loader.num_rows(), dtype=bool)
    obs_idx = []; obs_outputs = []; obs_outputs_in_S = []
    covered = np.zeros(S_outputs.shape[0], dtype=bool)
    S_tree = BallTree(S_outputs, leaf_size=40, metric='euclidean')

    # initialize
    for ix in init_idx:
        used[ix] = True
        y = df.loc[ix, OBJ_COLS].values.astype(np.float32)
        obs_idx.append(int(ix)); obs_outputs.append(y)
        if S_mask[ix]:
            obs_outputs_in_S.append(y)
            update_covered_with_observation(y, S_tree, covered, r)

    gps = prefit_all_gps(loader, df, seed, prefit_n=prefit_n)
    t0 = time.time(); metrics=[]

    for step in range(1, rounds+1):
        # fit GPs
        X_obs = loader.get_rows_by_indices(np.array(obs_idx, dtype=int))
        for k, col in enumerate(OBJ_COLS):
            y_obs = df[col].values[obs_idx].astype(np.float32)
            gps[k].fit(X_obs, y_obs)

        t_obs = len(obs_idx)
        beta_t = beta0 * math.log1p(t_obs)
        if t_obs > beta_anneal_t:
            beta_t *= beta_anneal_factor  # late-stage less optimism

        # candidate subset
        avail = np.where(~used)[0]
        consider_idx = rng.choice(avail, size=min(candidate_cap, avail.size), replace=False) if avail.size>candidate_cap else avail

        # shortlist
        shortlist = build_shortlist(gps, beta_t, loader, used, shortlist_k, random_k, consider_idx, scan_cap_per_shard)
        if shortlist.size == 0:
            cand_idx = np.array([np.where(~used)[0][0]], dtype=int)
        else:
            X_cand = loader.get_rows_by_indices(shortlist)
            U = optimistic_outputs(gps, X_cand, beta_t)
            tau_vec = np.array([tau[c] for c in OBJ_COLS], dtype=np.float32)
            feas = np.all(U >= tau_vec[None,:], axis=1)

            # ----- coverage gain (soft) on a capped subset of feasible candidates -----
            gains = np.zeros(U.shape[0], dtype=np.float32)
            feas_idx = np.where(feas)[0]
            if feas_idx.size > 0:
                # cap & cluster reps if many feasible
                if feas_idx.size > cover_cap:
                    k = min(k_div, feas_idx.size, cover_cap)
                    km = KMeans(n_clusters=k, n_init=3, random_state=seed)
                    labels = km.fit_predict(U[feas_idx])
                    reps = []
                    for c in range(k):
                        bucket = feas_idx[labels==c]
                        if bucket.size == 0: 
                            continue
                        center = km.cluster_centers_[c]
                        sub = U[bucket]
                        j = np.argmin(np.sum((sub - center[None,:])**2, axis=1))
                        reps.append(bucket[j])
                    reps = np.array(reps, dtype=int)
                    gains_reps = soft_coverage_gain(U[reps], S_tree, covered, r)
                    gains[reps] = gains_reps
                else:
                    gains[feas_idx] = soft_coverage_gain(U[feas_idx], S_tree, covered, r)

            # repulsion from already-covered S (use observed-in-S points)
            repulse = farthest_output_tiebreak(U, obs_outputs_in_S) if len(obs_outputs_in_S)>0 else np.ones(U.shape[0], dtype=np.float32)

            # combined score
            score = gains + lambda_repulse * repulse

            if not np.any(score > 0):
                score = repulse

            best = int(np.argmax(score))
            cand_idx = np.array([shortlist[best]], dtype=int)

        i_sel = int(cand_idx[0])
        if used[i_sel]: 
            i_sel = int(np.where(~used)[0][0])
        used[i_sel] = True
        y_true = df.loc[i_sel, OBJ_COLS].values.astype(np.float32)
        obs_idx.append(i_sel); obs_outputs.append(y_true)
        if S_mask[i_sel]:
            obs_outputs_in_S.append(y_true); update_covered_with_observation(y_true, S_tree, covered, r)

        # metrics
        pos = len(obs_outputs_in_S)
        rec = coverage_recall(covered)
        fd = fill_distance(np.vstack(obs_outputs_in_S) if pos>0 else np.empty((0, len(OBJ_COLS))), S_outputs)
        elapsed = time.time() - t0
        metrics.append({"t": len(obs_idx), "positives": pos, "recall": rec, "fill": fd, "runtime_sec": elapsed})

        if (step % print_every)==0 or step==1:
            print(f"    [MOC-CAS] Round {step:3d}/{rounds} | shortlist={shortlist.size} | feas={int(np.sum(feas)) if shortlist.size>0 else 0} | pos={pos} | rec={rec:.4f} | fill={fd:.3f} | time={elapsed:.1f}s", flush=True)

    return pd.DataFrame(metrics), obs_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="processed.parquet")
    ap.add_argument("--features_dir", default="rdkit_desc_parts")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n_init", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--r", type=float, default=0.05)
    ap.add_argument("--tau_json", type=str, required=True)
    
    ap.add_argument("--shortlist_k", type=int, default=3000)
    ap.add_argument("--random_k", type=int, default=500)
    ap.add_argument("--prefit_n", type=int, default=200)
    ap.add_argument("--candidate_cap", type=int, default=80000)
    ap.add_argument("--scan_cap_per_shard", type=int, default=5000)
    
    ap.add_argument("--beta0", type=float, default=2.0)
    ap.add_argument("--beta_anneal_t", type=int, default=150)
    ap.add_argument("--beta_anneal_factor", type=float, default=0.7)
    ap.add_argument("--cover_cap", type=int, default=800)
    ap.add_argument("--k_div", type=int, default=40)
    ap.add_argument("--lambda_repulse", type=float, default=0.2)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--out_csv", type=str, default="moccas.csv")
    ap.add_argument("--save_selections_dir", type=str, default=None, help="If set, dump per-trial selection indices as JSON.")

    args = ap.parse_args()

    print("Loading objectives...")
    df = load_objectives(args.parquet)
    print("Preparing features...")
    loader = FeatureLoader(args.features_dir)

    # fixed init set for reproducibility
    rng = np.random.default_rng(args.seed)
    init_idx = rng.choice(loader.num_rows(), size=args.n_init, replace=False)

    tau = json.loads(args.tau_json)

    print(f"[MOC-CAS] seed={args.seed} rounds={args.rounds} n_init={args.n_init}")
    t0 = time.time()
    df_m, selections = run_moccas(df, loader, tau, args.r, args.rounds, init_idx, seed=args.seed,
                              beta0=args.beta0, beta_anneal_t=args.beta_anneal_t, beta_anneal_factor=args.beta_anneal_factor,
                              shortlist_k=args.shortlist_k, random_k=args.random_k, prefit_n=args.prefit_n,
                              candidate_cap=args.candidate_cap, scan_cap_per_shard=args.scan_cap_per_shard,
                              cover_cap=args.cover_cap, k_div=args.k_div, lambda_repulse=args.lambda_repulse,
                              print_every=args.print_every)
    runtime = time.time() - t0
    df_m.to_csv(args.out_csv, index=False)
    print(f"Done. Wrote {args.out_csv} (runtime {runtime:.2f}s)")
    
    if args.save_selections_dir:
        try:
            os.makedirs(args.save_selections_dir, exist_ok=True)
        except Exception:
            pass
        out_js = os.path.join(args.save_selections_dir, f"moccas_seed{args.seed}.json")
        payload = {
            "seed": args.seed,
            "n_init": args.n_init,
            "rounds": args.rounds,
            "r": args.r,
            "tau": json.loads(args.tau_json),
            "obj_cols": OBJ_COLS,
            # Full ordered list including the n_init picks first, then round-by-round choices
            "selected_indices": [int(i) for i in selections],
        }
        with open(out_js, "w") as f:
            json.dump(payload, f)
        print(f"Wrote selections to {out_js}")

if __name__ == "__main__":
    main()