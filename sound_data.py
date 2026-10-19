import numpy as np, librosa, json
from scipy.signal import find_peaks
def sm(x,k): return np.convolve(x,np.ones(k)/k,'same')
out={}
# --- 연주2: envelope at 20fps + bow segments ---
F=np.load('a/feat.npz'); r=F['rms']; m=F['midi']; o=F['ons']
p95=np.percentile(r,95)
rs=sm(r,21); pk,_=find_peaks(-rs,distance=45,prominence=0.002)
B=np.r_[60,pk[pk>60],len(r)-60]
segs=[]
for i in range(len(B)-1):
    s0,s1=B[i],B[i+1]
    segs.append([round(s0/30,2),round(s1/30,2),round(float(m[s0:s1].mean()),1),round(float(r[s0:s1].mean()/p95),2)])
idx=np.arange(0,len(r),1.5).astype(int)
env=np.clip(sm(r,5)/p95,0,1.5)[idx]; mm=sm(m,5)[idx]
onz=o/np.percentile(o,99); onsets=[round(i/30,2) for i in np.where((onz>0.8)&(onz>=np.roll(onz,1))&(onz>=np.roll(onz,-1)))[0]]
out['c2']=dict(fps=20,env=[round(float(v),2) for v in env],midi=[round(float(v),1) for v in mm],segs=segs,onsets=onsets)
# --- 연주1: 56 cells ---
y,sr=librosa.load('a/p1.wav',sr=11025); hop=sr//20
r1=librosa.feature.rms(y=y,hop_length=hop)[0]; r1s=sm(r1,15)
pk1,_=find_peaks(-r1s,distance=40,prominence=0.001)
b=np.r_[10,pk1,len(r1)-10]
cand=[(b[i],b[i+1]) for i in range(len(b)-1) if 50<=b[i+1]-b[i]<=140]
pick=[cand[int(j)] for j in np.linspace(0,len(cand)-1,56)]
f0,_,_=librosa.pyin(y,fmin=55,fmax=800,sr=sr,hop_length=hop,frame_length=1024)
mid=librosa.hz_to_midi(f0)
cells=[]
for s0,s1 in pick:
    seg=mid[s0:s1]; seg=seg[~np.isnan(seg)]
    cells.append([round(s0/20,2),round(s1/20,2),round(float(np.median(seg)) if len(seg) else 48.0,1),round(float(r1[s0:s1].mean()/np.percentile(r1,95)),2)])
out['c1']=dict(cells=cells)
json.dump(out,open('mvp/assets/sound.json','w'),separators=(',',':'))
print(len(segs),len(cand),len(cells),len(onsets))
