import glob
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

files = glob.glob("vis_*_source.txt")

for source_file in files:
    base = source_file.replace("_source.txt", "")
    target_file = base + "_target.txt"
    target_exact_file = base + "_target_exact.txt"
    
    if not os.path.exists(target_file) or not os.path.exists(target_exact_file):
        continue
        
    def read_mesh(filename):
        polys = []
        vals = []
        with open(filename) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cx, cy, val, nverts = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                verts = []
                idx = 4
                for i in range(nverts):
                    verts.append([float(parts[idx]), float(parts[idx+1])])
                    idx += 2
                polys.append(verts)
                vals.append(val)
        return polys, np.array(vals)
        
    src_polys, src_vals = read_mesh(source_file)
    tgt_polys, tgt_vals = read_mesh(target_file)
    _, tgt_exact_vals = read_mesh(target_exact_file)
    
    tgt_error = tgt_vals - tgt_exact_vals
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(src_vals.min(), tgt_vals.min(), tgt_exact_vals.min())
    vmax = max(src_vals.max(), tgt_vals.max(), tgt_exact_vals.max())
    
    coll1 = PolyCollection(src_polys, array=src_vals, cmap='viridis', edgecolors='k', linewidths=0.5)
    coll1.set_clim(vmin, vmax)
    ax1.add_collection(coll1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("Source Grid & Solution")
    ax1.set_aspect('equal')
    
    coll2 = PolyCollection(tgt_polys, array=tgt_vals, cmap='viridis', edgecolors='k', linewidths=0.5)
    coll2.set_clim(vmin, vmax)
    ax2.add_collection(coll2)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("Target Grid & Reconstructed Solution")
    ax2.set_aspect('equal')
    
    # Error plot needs its own symmetric colorbar, e.g. RdBu or coolwarm, centered at 0
    max_err = np.abs(tgt_error).max()
    if max_err == 0: max_err = 1e-16
    
    coll3 = PolyCollection(tgt_polys, array=tgt_error, cmap='RdBu', edgecolors='k', linewidths=0.5)
    coll3.set_clim(-max_err, max_err)
    ax3.add_collection(coll3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title(f"Target Grid & Error Profile\\n(Max Error: {max_err:.2e})")
    ax3.set_aspect('equal')
    
    # Add colorbars
    fig.colorbar(coll2, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.1)
    fig.colorbar(coll3, ax=ax3, orientation='horizontal', fraction=0.05, pad=0.1)
    
    fig.suptitle(base.replace("vis_", "").replace("_", " "), fontsize=14)
    
    plt.savefig(base + ".png", dpi=150, bbox_inches='tight')
    plt.close()
    
print("Visualization PNGs generated with error profiles.")
