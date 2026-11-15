"""Paper/README figures from a calendar build: peak month map, annual activity map, global monthly series.

    python scripts/make_figures.py v2 [CACHE_DIR]      # needs out/.../calendar_<tag>, harmonized_<tag>, and pred_aqua_<tag> in CACHE_DIR
Writes docs/figures/*.png.
"""
import os, sys
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
ROOT=Path(__file__).resolve().parent.parent
tag=sys.argv[1]; R=ROOT/"out/features/v2/ll0p25"
CACHE=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/"cache"
OUT=ROOT/"docs/figures"; OUT.mkdir(parents=True,exist_ok=True); os.chdir(OUT)
plt.rcParams.update({"font.size":10,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb"})
c=pd.read_parquet(f"{R}/calendar_{tag}/climatology.parquet"); iy,ix=np.divmod(c.cell.to_numpy(),1440)
def raster(v):
    a=np.full((720,1440),np.nan); a[iy,ix]=v; return a
land=raster(np.ones(len(c))); act=c.annual_mean.to_numpy()>=1; gray=LinearSegmentedColormap.from_list("l",["#a9a79f","#a9a79f"])
def mapfig(arr,cmap,norm,title,fn,cbfmt):
    fig,ax=plt.subplots(figsize=(12,5.6))
    ax.imshow(land,origin="lower",extent=[-180,180,-90,90],cmap=gray,interpolation="nearest")
    im=ax.imshow(arr,origin="lower",extent=[-180,180,-90,90],cmap=cmap,interpolation="nearest",**norm)
    ax.set_ylim(-60,80); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=0.04,pad=0.02); cb.outline.set_visible(False); cbfmt(cb)
    ax.set_title(title,loc="left",color="#0b0b0b"); fig.savefig(fn,dpi=150,bbox_inches="tight"); plt.close()
def months(cb):
    cb.set_ticks(range(1,13)); cb.ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
mapfig(raster(np.where(act&~c.persistent.to_numpy(),c.peak_month.to_numpy(),np.nan)),"twilight_shifted",dict(vmin=0.5,vmax=12.5),
  "Peak burning month, MODIS-Aqua-equivalent record 2003–2025\n(gray = land with <1 fire pixel-day/yr; gas flares and other steady sources left out)","peak_month_map.png",months)
blues=LinearSegmentedColormap.from_list("b",["#cde2fb","#86b6ef","#3987e5","#256abf","#184f95","#0d366b"])
mapfig(raster(np.where(act,c.annual_mean.to_numpy(),np.nan)),blues,dict(norm=LogNorm(1,3000)),"Mean annual burning activity, MODIS-Aqua-equivalent, 2003–2025\n(gray = land with <1 fire pixel-day/yr)","annual_mean_map.png",
  lambda cb: cb.set_label("fire pixel-days per year per 0.25° cell (log scale)",color="#52514e"))
obs=[];mod=[];drift=[]
for y in range(2002,2027):
    h=pd.read_parquet(f"{R}/harmonized_{tag}/{y}.parquet"); obs.append(h[h.source=="aqua_obs"].groupby("ym").fire_pd.sum())
for y in range(2012,2027):
    p=pd.read_parquet(CACHE/f"pred_aqua_{tag}"/f"{y}.parquet",columns=["ym","pred_aqua_equiv","pred_from_j","a_fire","a_ok"])
    v=np.where((p.ym>=202611)&p.pred_from_j.notna(),p.pred_from_j,p.pred_aqua_equiv); mod.append(pd.Series(v).groupby(p.ym.values).sum())
    q=p[(p.ym>202301)&(p.ym<202611)&p.a_ok]; drift.append(q.groupby("ym").a_fire.sum())
def ts(parts):
    s=pd.concat(parts); s=s[~s.index.duplicated()]; s.index=pd.to_datetime((s.index*100+1).astype(str)); return s/1e3
O,M,D=ts(obs),ts(mod),ts(drift); M=M[M.index<"2026-11-01"]
fig,ax=plt.subplots(figsize=(12,4.4))
ax.axvspan(pd.Timestamp("2023-02-01"),pd.Timestamp("2026-11-01"),color="#f0efec",zorder=0)
ax.plot(O.index,O.values,color="#2a78d6",lw=1.6,label="Aqua observed (in-spec orbit)")
ax.plot(D.index,D.values,color="#2a78d6",lw=1.6,ls=(0,(3,2)),label="Aqua observed (drifting orbit)")
ax.plot(M.index,M.values,color="#eb6834",lw=1.6,label="Model from VIIRS")
ax.set_ylim(0,1350); ax.text(pd.Timestamp("2023-03-01"),1320,"Aqua off its\n13:30 orbit",color="#52514e",va="top",fontsize=9)
ax.set_ylabel("thousand fire pixel-days / month"); ax.grid(axis="y",color="#e6e5e1",lw=0.8); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=3)
ax.set_title("Global land burning in MODIS-Aqua terms, 2002 – Oct 2026",loc="left",color="#0b0b0b")
fig.savefig("global_monthly.png",dpi=150,bbox_inches="tight"); plt.close()
ov=pd.concat([O.rename("o"),M.rename("m")],axis=1).dropna(); ov=ov[ov.index<="2023-01-01"]
print("overlap",len(ov),"corr",round(np.corrcoef(ov.o,ov.m)[0,1],4),"ratio",round(ov.m.sum()/ov.o.sum(),4),"median abs %",round(float((100*abs(ov.m-ov.o)/ov.o).median()),2))
