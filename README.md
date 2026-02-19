# MOC-CAS (AAMAS 2026) — Reference code for SARS-CoV-2 3CLPro dataset

This repository contains a reference implementation of **MOC-CAS** (Multi-Objective Coverage via Constraint Active Search)
used in the **AAMAS 2026 camera-ready** experiments on the **SARS-CoV-2 3CLPro target dataset**.
The same code/logic was applied to other datasets in the paper.

## Repository contents

- `moccas.py` — implementation of the MOC-CAS algorithm
- `processed.parquet` — processed SARS-CoV-2 3CLPro target dataset containing the five objectives used in the paper
- `rdkit_desc_parts/` — precomputed RDKit descriptor features (sharded)
  - `feats_part_*_final.npz` (20 shards)
  - `meta_final.npz` (and any other metadata files included)
- `README.md` — repository documentation (setup, quick start, multi-seed loop, outputs, and how to choose `tau_json`)
- `requirements.txt` — Python dependencies needed to run `moccas.py`
- `.gitignore` — ignore rules for common generated files (e.g., Python bytecode caches like `__pycache__/` and macOS metadata like `.DS_Store`)


## Environment setup (Conda)

```bash
# create a fresh environment
conda create -n moccas python=3.11 -y
conda activate moccas

# install required packages
pip install -r requirements.txt
```

---

## Quick start (single run)

```bash
mkdir -p results_all selections_jsons

python moccas.py   --parquet processed.parquet   --features_dir rdkit_desc_parts   --rounds 200 --n_init 20   --seed 222   --r 0.035   --tau_json '{"dock":0.2728923883427605,"solubility":0.28617925137032474,"sa":0.7008041910986325,"qed":0.5565723694469105,"similarity_to_topK":0.22950819672131148}'   --shortlist_k 3000 --random_k 500   --prefit_n 200   --candidate_cap 80000   --scan_cap_per_shard 5000   --beta0 4.0 --beta_anneal_t 100 --beta_anneal_factor 0.5   --cover_cap 800 --k_div 40 --lambda_repulse 0.2   --save_selections_dir selections_jsons   --print_every 5   --out_csv results_all/moccas_seed222.csv
```

---

## Running multiple seeds

```bash
outdir="results_all"
mkdir -p "$outdir" selections_jsons

i=0
for s in 222 323 523 555; do
  i=$((i+1))
  echo "===> MOC-CAS trial $i (seed=$s)"
  python moccas.py     --parquet processed.parquet     --features_dir rdkit_desc_parts     --rounds 200 --n_init 20     --seed "$s"     --r 0.035     --tau_json '{"dock":0.2728923883427605,"solubility":0.28617925137032474,"sa":0.7008041910986325,"qed":0.5565723694469105,"similarity_to_topK":0.22950819672131148}'     --shortlist_k 3000 --random_k 500     --prefit_n 200     --candidate_cap 80000     --scan_cap_per_shard 5000     --beta0 4.0 --beta_anneal_t 100 --beta_anneal_factor 0.5     --cover_cap 800 --k_div 40 --lambda_repulse 0.2     --save_selections_dir selections_jsons     --print_every 5     --out_csv "$outdir/moccas_trial${i}.csv"
done
```

---

## Output files

Each run produces:

- **CSV log** (one per seed), e.g. `results_all/moccas_trial1.csv`  
  This file records the per-round progress and the selected candidates (and their objective values) as written by the script.
- **Selections JSONs** in `selections_jsons/` (optional, if enabled in the script)  
  These store the selected indices / candidate metadata per round for later analysis or plotting.

---

## Choosing `tau_json`

`tau_json` defines the **per-objective thresholds** that determine which points are considered *satisfactory/feasible* in the objective space for constructing the constraint set used by MOC-CAS.

In the experiment referenced here, the provided `tau_json` thresholds were chosen so that the feasible set size in the objective space is
approximately **|S| ≈ 300,000** (about **30%** of the dataset) out of **1,000,000** total tuples.

---

## Citation

If you use this code, please cite our paper (accepted at AAMAS 2026):

Zakaria Shams Siam, Xuefeng Liu, Chong Liu. *Multi-Objective Coverage via Constraint Active Search*. arXiv:2602.15595, 2026.

### BibTeX
```bibtex
@misc{siam2026moccas,
  title        = {Multi-Objective Coverage via Constraint Active Search},
  author       = {Siam, Zakaria Shams and Liu, Xuefeng and Liu, Chong},
  year         = {2026},
  eprint       = {2602.15595},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  doi          = {10.48550/arXiv.2602.15595},
  url          = {https://arxiv.org/abs/2602.15595}
}
