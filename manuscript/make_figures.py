"""Plot the manuscript figures from the archived numerical results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

MODELS = {
    "cvsm_art": ("CVSM-ART", "#141414", "-", "o"),
    "cvsm_a": ("CVSM-A", "#565656", "--", "s"),
    "ssl": ("Torres SSL", "#292929", "-.", "^"),
    "ecapa": ("ECAPA-TDNN", "#707070", ":", "D"),
    "mert": ("MERT", "#4b4b4b", (0, (3, 1, 1, 1)), "x"),
}
PANELS = [
    ("reverb", "Synthetic reverb", "Wet fraction", [0.1, 0.2, 0.3, 0.5]),
    ("codec", "EnCodec", "Rate (kb/s)", [1.5, 3, 6, 12, 24]),
    ("mp3", "MP3", "Rate (kb/s)", [32, 64, 128]),
    ("eq", "Equalization", "Shelf gain (dB)", [3, 6, 12]),
    ("mixsep", "Mix and separate", "Vocal/accomp. (dB)", [-6, 0, 6, 12]),
]


def read(path: Path):
    return json.loads(path.read_text())


def save(fig, path: Path):
    fig.savefig(path, dpi=1200, facecolor="white")
    plt.close(fig)


def sweep(pilots: Path, output: Path):
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.85), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.983, top=0.87, bottom=0.20, hspace=0.53, wspace=0.3)
    for i, (dataset, source) in enumerate((("VocalSet", "f2"), ("GTSinger", "g1"))):
        rows = {r["model"]:r for r in read(pilots / f"{source}_results.json")}
        for j, (name, heading, label, values) in enumerate(PANELS):
            ax = axes[i,j]
            for key, (title, color, style, marker) in MODELS.items():
                results = [rows[key]["levels"][f"{name}@{v:g}"]["error_pct"] for v in values]
                ax.plot(values, results, label=title, color=color, ls=style, marker=marker,
                        lw=1.12, ms=3.4, markerfacecolor="white", markeredgewidth=0.85)
            if i == 0:
                ax.set_title(heading, fontsize=8)
            if i == 1:
                ax.set_xlabel(label, fontsize=6.6)
            if name in ("codec", "mp3"):
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_xticks(values, [f"{v:g}" for v in values])
            ax.tick_params(axis="both",labelsize=6.5,length=2.5)
            ax.set_ylim(0, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y",color="#d0d0d0", linewidth=0.4)
            ax.spines[["top", "right"]].set_visible(False)
        axes[i,0].set_ylabel(f"{dataset}\nRanking error (%)", fontsize=8)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=5,bbox_to_anchor=(0.54,0.01),
               fontsize=8,frameon=False)
    save(fig,output)


def jvs_control(path: Path, output: Path):
    cells = read(path)["jvs"]["models"]["cvsm_art"]["roles"]["common"]["conditions"]["reverb_20"]["cells"]
    fig,ax=plt.subplots(figsize=(7.2, 2.75))
    keys = ("000", "010", "011")
    settings = ("Both candidates clean", "Genuine reverberant", "Both candidates reverberant")
    x=np.arange(len(keys))
    ax.plot(x, [cells[k]["genuine_cosine"]["value"] for k in keys], marker="o",
            color="#151515",label="Genuine candidate", lw=1.4)
    ax.plot(x, [cells[k]["impostor_cosine"]["value"] for k in keys], marker="s",
            color="#5b5b5b",ls="--",label="Impostor", lw=1.4)
    ax.set_ylim(0.818,0.977)
    ax.set_yticks([0.82,0.85,0.88,0.91,0.94,0.97])
    ax.set_xlim(-0.44,2.44)
    ax.set_xticks(x,settings)
    ax.set_ylabel("Mean cosine similarity")
    for i,k in enumerate(keys):
        ax.text(i, 0.956, f"{cells[k]['ranking_error_pct']['value']:.1f}% error",
                ha="center",fontsize=9)
    ax.grid(axis="y",color="#d0d0d0",linewidth=0.45)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=2,fontsize=8.6)
    fig.subplots_adjust(left=0.11,right=0.98,top=0.98,bottom=0.26)
    save(fig,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilots",type=Path,required=True)
    parser.add_argument("--factorial",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"figures")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sweep(args.pilots,args.output/"figure-1-dose-response.png")
    jvs_control(args.factorial,args.output/"figure-2-jvs-control.png")


if __name__=="__main__":
    main()
