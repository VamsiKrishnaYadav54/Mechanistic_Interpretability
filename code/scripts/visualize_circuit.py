"""
Circuit visualiser. Reads scores_structural.json; needs no GPU and no model.
Runs fine on the login node.

Single method:
    python code/scripts/visualize_circuit.py \
        --scores results/ioi_eap_gpt2/scores_structural.json \
        --top 40 --out results/figures/eap_top40

Compare two methods at the same size:
    python code/scripts/visualize_circuit.py \
        --compare results/ioi_eap_gpt2/scores_structural.json \
                  results/ioi_acdc_gpt2/scores_structural.json \
        --top 40 --out results/figures/eap_vs_acdc

How to read the picture:
  y-axis = layer (embeddings at the bottom, logits at the top)
  x-axis = head index 0-11, MLPs in the column on the right
  node colour = the head's role in Wang et al. (2022), grey = no known role
  edge width  = |score| (or survival level, for ACDC)
  edge colour = single mode: blue supports the correct name, red opposes it
                compare mode: purple = found by both, orange = only this one
  Q/K/V inputs of the same head are merged into one arrow; the CSV keeps them.

TIES: if the top-N cut falls inside a group of equal scores the whole group
is taken (ACDC has big ties), so the picture may hold more than N edges.
The title says how many.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402

# Wang et al. (2022), as encoded in auto_circuit ioi_official.py IOI_CIRCUIT
IOI_ROLES = {
    "name mover":        [(9, 9), (10, 0), (9, 6)],
    "negative name mover": [(10, 7), (11, 10)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2),
                          (9, 7), (9, 0), (11, 9)],
    "s2 inhibition":     [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction":         [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token":   [(0, 1), (0, 10), (3, 0)],
    "previous token":    [(2, 2), (4, 11)],
}
ROLE_COLOURS = {
    "name mover": "#1b9e77",
    "negative name mover": "#d95f02",
    "backup name mover": "#66c2a5",
    "s2 inhibition": "#7570b3",
    "induction": "#e7298a",
    "duplicate token": "#e6ab02",
    "previous token": "#a6761d",
}
HEAD_ROLE = {f"A{l}.{h}": role for role, hs in IOI_ROLES.items() for l, h in hs}

POS, NEG, ACDC_C = "#2166ac", "#b2182b", "#404040"
SHARED, ONLY = "#6a3d9a", "#e66101"
MLP_X = 12.8


# ------------------------------------------------------------- data
def load(path):
    with open(path) as f:
        d = json.load(f)
    return d["meta"], d["edges"]


def _key(r):
    return (bool(r["is_inf"]), r["abs_score"] or 0.0)


def select(edges, n):
    """Top n by rank, extended to include the whole tie group at the cut."""
    if n >= len(edges):
        return list(edges)
    cut = _key(edges[n - 1])
    return [r for r in edges if _key(r) >= cut]


def dest_node(dest):
    """'A11.10.Q' -> ('A11.10', 'Q');  'MLP 4' -> ('MLP 4', '')."""
    if dest.startswith("A") and dest.count(".") == 2:
        node, ch = dest.rsplit(".", 1)
        return node, ch
    return dest, ""


def merge_qkv(sel):
    merged = {}
    for r in sel:
        dn, ch = dest_node(r["dest"])
        k = (r["src"], dn)
        m = merged.setdefault(k, {"src": r["src"], "dest": dn, "channels": set(),
                                  "a": -1.0, "sign": 0, "is_inf": False})
        if ch:
            m["channels"].add(ch)
        a = math.inf if r["is_inf"] else (r["abs_score"] or 0.0)
        if a > m["a"]:
            m["a"], m["sign"] = a, r["sign"]
        m["is_inf"] = m["is_inf"] or r["is_inf"]
    return merged


# ------------------------------------------------------------- layout
def xy(node):
    if node == "Resid Start":
        return (5.5, -1.3)
    if node == "Resid End":
        return (5.5, 12.5)
    if node.startswith("MLP "):
        return (MLP_X, int(node.split()[1]) + 0.45)
    if node.startswith("A"):
        layer, head = node[1:].split(".")
        return (int(head), int(layer))
    raise ValueError(node)


def short(node):
    if node == "Resid Start":
        return "emb"
    if node == "Resid End":
        return "logits"
    if node.startswith("MLP "):
        return "M" + node.split()[1]
    return node[1:]


def draw_background(ax):
    for layer in range(12):
        ax.scatter(range(12), [layer] * 12, s=10, c="#e0e0e0", zorder=1)
        ax.scatter([MLP_X], [layer + 0.45], s=10, c="#e0e0e0", marker="s", zorder=1)
    ax.set_xlim(-1, MLP_X + 1)
    ax.set_ylim(-2.1, 13.3)
    ax.set_xticks(list(range(12)) + [MLP_X])
    ax.set_xticklabels([str(h) for h in range(12)] + ["MLP"])
    ax.set_yticks([-1.3] + list(range(12)) + [12.5])
    ax.set_yticklabels(["emb"] + [str(l) for l in range(12)] + ["logits"])
    ax.set_xlabel("head index")
    ax.set_ylabel("layer")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_edges(ax, merged, colour_fn):
    finite = [m["a"] for m in merged.values() if math.isfinite(m["a"])]
    amax = max(finite) if finite else 1.0
    amax = amax if amax > 0 else 1.0
    items = sorted(merged.items(),
                   key=lambda kv: kv[1]["a"] if math.isfinite(kv[1]["a"]) else amax)
    # inf edges: full width only if finite edges exist to contrast with;
    # if everything is inf (pure ACDC survivors) draw them all medium.
    inf_frac = 1.0 if finite else 0.3
    for i, (k, m) in enumerate(items):
        frac = inf_frac if not math.isfinite(m["a"]) else m["a"] / amax
        rad = (0.10, 0.22, -0.16)[i % 3]
        ax.add_patch(FancyArrowPatch(
            xy(m["src"]), xy(m["dest"]),
            connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=7, lw=0.4 + 3.6 * frac,
            color=colour_fn(k, m), alpha=0.35 + 0.5 * frac,
            shrinkA=5, shrinkB=5, zorder=2))


def draw_nodes(ax, merged):
    active = {m["src"] for m in merged.values()} | {m["dest"] for m in merged.values()}
    for node in active:
        x, y = xy(node)
        role = HEAD_ROLE.get(node)
        ax.scatter([x], [y], s=150, zorder=3, edgecolors="black", linewidths=0.6,
                   c=ROLE_COLOURS[role] if role else "#bdbdbd",
                   marker="s" if node.startswith("MLP") else "o")
        ax.text(x, y + 0.33, short(node), ha="center", va="bottom",
                fontsize=7, zorder=4)
    return active


def role_legend(ax, extra):
    handles = [Patch(color=c, label=r) for r, c in ROLE_COLOURS.items()]
    handles.append(Patch(color="#bdbdbd", label="no known IOI role"))
    ax.legend(handles=handles + extra, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, frameon=False)


# ------------------------------------------------------------- reports
def coverage(active):
    out = {}
    for role, heads in IOI_ROLES.items():
        names = [f"A{l}.{h}" for l, h in heads]
        out[role] = [n for n in names if n in active]
    return out


def print_coverage(label, active):
    print(f"\n  IOI roles covered by {label}:")
    for role, hit in coverage(active).items():
        total = len(IOI_ROLES[role])
        print(f"    {role:22s} {len(hit)}/{total}   {', '.join(hit)}")


def write_csv(path, sel):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "src", "dest", "score", "abs_score", "is_inf",
                    "best_pos", "best_token", "src_role", "dest_role"])
        for r in sel:
            dn, _ = dest_node(r["dest"])
            w.writerow([r["rank"], r["src"], r["dest"], r["score"], r["abs_score"],
                        r["is_inf"], r["best_pos"], r["best_token"],
                        HEAD_ROLE.get(r["src"], ""), HEAD_ROLE.get(dn, "")])


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out.with_suffix(ext), dpi=160, bbox_inches="tight")
    print(f"  saved {out.with_suffix('.png')} and .pdf")


# ------------------------------------------------------------- modes
def single(path, top, out):
    meta, edges = load(path)
    sel = select(edges, top)
    merged = merge_qkv(sel)
    is_acdc = meta["method"] == "acdc"

    fig, ax = plt.subplots(figsize=(13, 9))
    draw_background(ax)
    if is_acdc:
        colour = lambda k, m: ACDC_C
        extra = [Line2D([], [], color=ACDC_C, lw=2,
                        label="ACDC edge (width = survival level)")]
    else:
        colour = lambda k, m: POS if m["sign"] >= 0 else NEG
        extra = [Line2D([], [], color=POS, lw=2, label="supports correct name"),
                 Line2D([], [], color=NEG, lw=2, label="opposes correct name")]
    draw_edges(ax, merged, colour)
    active = draw_nodes(ax, merged)
    role_legend(ax, extra)
    tie = "" if len(sel) == top else f"  (top {top} extended to {len(sel)}: tie at cut)"
    ax.set_title(f"{meta['method'].upper()} on {meta['model']}: {len(sel)} structural "
                 f"edges, {len(merged)} arrows after merging Q/K/V{tie}", fontsize=11)

    save(fig, out)
    write_csv(Path(out).with_suffix(".csv"), sel)
    print(f"  {meta['method']}: selected {len(sel)} structural edges "
          f"(asked {top}), {len(merged)} arrows")
    print_coverage(meta["method"], active)


def compare(path_a, path_b, top, out):
    meta_a, edges_a = load(path_a)
    meta_b, edges_b = load(path_b)

    sel_a, sel_b = select(edges_a, top), select(edges_b, top)
    n = max(len(sel_a), len(sel_b))
    if n != top:
        print(f"  tie at the cut: evaluating both methods at {n} edges")
        sel_a, sel_b = select(edges_a, n), select(edges_b, n)

    set_a = {(r["src"], r["dest"]) for r in sel_a}
    set_b = {(r["src"], r["dest"]) for r in sel_b}
    jac = len(set_a & set_b) / max(1, len(set_a | set_b))
    ma, mb = merge_qkv(sel_a), merge_qkv(sel_b)
    shared = set(ma) & set(mb)

    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    actives = []
    for ax, meta, merged, sel in ((axes[0], meta_a, ma, sel_a), (axes[1], meta_b, mb, sel_b)):
        draw_background(ax)
        draw_edges(ax, merged, lambda k, m: SHARED if k in shared else ONLY)
        actives.append(draw_nodes(ax, merged))
        ax.set_title(f"{meta['method'].upper()}: {len(sel)} structural edges, "
                     f"{len(merged)} arrows", fontsize=11)
    role_legend(axes[1], [Line2D([], [], color=SHARED, lw=2, label="found by both"),
                          Line2D([], [], color=ONLY, lw=2, label="only this method")])
    fig.suptitle(f"{meta_a['method'].upper()} vs {meta_b['method'].upper()} at "
                 f"{n} edges: structural overlap (Jaccard) = {jac:.3f}", fontsize=13)

    save(fig, out)
    write_csv(Path(out).with_name(Path(out).name + f"_{meta_a['method']}.csv"), sel_a)
    write_csv(Path(out).with_name(Path(out).name + f"_{meta_b['method']}.csv"), sel_b)
    print(f"\n  size compared       : {n} structural edges each")
    print(f"  shared edges        : {len(set_a & set_b)}")
    print(f"  Jaccard (structural): {jac:.3f}")
    print(f"  Jaccard (Q/K/V merged arrows): "
          f"{len(shared) / max(1, len(set(ma) | set(mb))):.3f}")
    print_coverage(meta_a["method"], actives[0])
    print_coverage(meta_b["method"], actives[1])


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scores", help="one scores_structural.json")
    g.add_argument("--compare", nargs=2, help="two scores_structural.json files")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()
    if args.scores:
        single(args.scores, args.top, args.out)
    else:
        compare(args.compare[0], args.compare[1], args.top, args.out)


if __name__ == "__main__":
    sys.exit(main())
