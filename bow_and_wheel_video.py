import sys, math, subprocess, numpy as np, cairo
from scipy.signal import find_peaks
rng = np.random.default_rng(23)
W, H, FPS, SS = 1920, 1080, 30, 2          # paint is kept at 2x for close-ups
F = np.load('feat.npz')
rms, midi, ons = F['rms'], F['midi'], F['ons']
N = len(rms)
PRE, POST = int(1.5*FPS), int(8*FPS)
TOT = PRE+N+POST
def sm(x, k): return np.convolve(x, np.ones(k)/k, 'same')
a = np.clip(sm(rms, 5)/np.percentile(rms, 95), 0, 1.4)
m = sm(midi, 5); m2 = sm(midi, 11)
onz = ons/np.percentile(ons, 99)
ease = lambda u: u*u*(3-2*u)

# ---------- bow strokes ----------
rs = sm(rms, 21)
pk, _ = find_peaks(-rs, distance=45, prominence=0.002)
B = np.r_[60, pk[pk > 60], N-60]; S = len(B)-1
P = dict(black=(0.05,0.045,0.06), plum=(0.20,0.07,0.18), dred=(0.58,0.06,0.07), red=(0.87,0.11,0.09),
         orange=(0.96,0.46,0.09), yellow=(0.97,0.82,0.16), lime=(0.66,0.85,0.25), green=(0.09,0.47,0.26),
         cobalt=(0.10,0.20,0.68), sky=(0.22,0.68,0.93), pink=(0.95,0.42,0.65), violet=(0.38,0.20,0.60),
         white=(0.95,0.93,0.86), ochre=(0.82,0.58,0.20))
LADDER = [(50,'dred'),(54,'orange'),(57,'yellow'),(58.7,'lime'),(59.7,'green'),(60.6,'red'),
          (61.6,'cobalt'),(62.6,'ochre'),(64,'pink'),(66,'violet'),(69,'sky'),(99,'white')]
NEIGH = {'dred':'orange','orange':'cobalt','yellow':'violet','lime':'plum','green':'pink','red':'sky',
         'cobalt':'yellow','ochre':'cobalt','pink':'green','violet':'lime','sky':'red','white':'black','black':'plum','plum':'black'}
def pick(mm, loud):
    if loud < 0.45: return str(rng.choice(['black','black','plum']))
    for lim, c in LADDER:
        if mm < lim: return c
strokes = []; prev = None
top, bot = 22, H-22; gap = (bot-top)/S
for k in range(S):
    s0, s1 = B[k], B[k+1]
    loud = float(rms[s0:s1].mean()/np.percentile(rms, 95)); mm = float(m[s0:s1].mean())
    c = 'dred' if k == 0 else pick(mm, loud)
    if c == prev: c = NEIGH[c]                 # neighbouring bands contrast, as in 선의 기억
    prev = c
    base = np.array(P[c])
    T = gap*(1.25+1.3*loud)
    nb = int(T/1.3)+8
    prog = np.cumsum(a[s0:s1]**1.2+0.05); prog /= prog[-1]
    sec = rng.random(nb)
    # shade bristles within one colour (pastel builds up unevenly)
    shades = [tuple(np.clip(base*rng.uniform(0.78, 1.12)+rng.normal(0, 0.025, 3), 0, 1)) for _ in range(nb)]
    for b in range(nb):
        if sec[b] < 0.07: shades[b] = P['white'] if c not in ('white', 'yellow') else P['black']
    slot = 0 if k == 0 else ((k+1)//2)*(1 if k % 2 else -1)   # centre first, then out above/below
    strokes.append(dict(s0=s0, s1=s1, yc=H/2+slot*gap, T=T, cols=shades, prog=prog, mseg=mm,
                        dir=1 if k % 2 == 0 else -1, nb=nb, boff=np.sort(rng.uniform(-0.5, 0.5, nb)),
                        bal=rng.uniform(0.55, 1.0, nb), bw=rng.uniform(1.6, 3.6, nb),
                        ph=rng.uniform(0, 1000, nb), done=0.0, head=None))
def noise1(xs, ph, sc):
    u = xs/sc+ph; i = np.floor(u); f = u-i; f = f*f*(3-2*f)
    h = lambda n: (np.sin(n*127.1+ph*3.7)*43758.5453) % 1.0
    return h(i)*(1-f)+h(i+1)*f

paint = cairo.ImageSurface(cairo.FORMAT_ARGB32, W*SS, H*SS)
pc = cairo.Context(paint); pc.scale(SS, SS); pc.set_line_cap(cairo.LINE_CAP_BUTT)
CH = 5.0
def draw_stroke(st, i):
    j = i-st['s0']
    p = 1.0 if j >= len(st['prog']) else float(st['prog'][j])
    p0, p1 = st['done'], p
    if p1 <= p0: return
    st['done'] = p1
    xa, xb = (-20+p0*(W+40), -20+p1*(W+40)) if st['dir'] > 0 else (W+20-p0*(W+40), W+20-p1*(W+40))
    st['head'] = xb
    lo, hi = min(xa, xb), max(xa, xb)
    xs = np.arange(math.floor(lo/CH)*CH, hi+CH, CH)
    if len(xs) < 2: xs = np.array([lo, hi])
    ii = min(i, N-1)
    dev1 = float(np.clip(m2[ii]-st['mseg'], -6, 6)*2.8)    # the melody bends the band
    dev0 = st.get('ld', dev1); st['ld'] = dev1
    press = 0.75+0.35*a[ii]                              # bow pressure → band body
    for b in range(st['nb']):
        y0b = st['yc']+st['boff'][b]*st['T']*press
        nz = noise1(xs, st['ph'][b], 30+st['T']*0.8)
        edge = 1-abs(st['boff'][b])*0.9
        pc.set_line_width(st['bw'][b])
        pc.set_source_rgba(*st['cols'][b], 1)
        for q in range(len(xs)-1):
            al = st['bal'][b]*np.clip((nz[q]-0.12)*1.9, 0, 1)*(0.6+0.4*edge)
            if al < 0.05: continue
            pc.set_source_rgba(*st['cols'][b], al)
            jit = (nz[q]-0.5)*1.2
            qa, qb = max(xs[q], lo), min(xs[q+1], hi)
            fa = abs(qa-xa)/max(1e-6, hi-lo); fb = abs(qb-xa)/max(1e-6, hi-lo)
            pc.move_to(qa, y0b+jit+dev0+(dev1-dev0)*fa); pc.line_to(qb, y0b+jit+dev0+(dev1-dev0)*fb); pc.stroke()

CUR = np.zeros(N)                                         # y of the band being drawn
for st in strokes: CUR[st['s0']:st['s1']+1] = st['yc']
CUR[:B[0]] = strokes[0]['yc']; CUR[B[-1]:] = strokes[-1]['yc']

# ---------- the wheel ----------
WH0, WH1 = int(58*FPS), int(120*FPS)
x, y, th, curv, load = W*0.25, H/2, -0.2, 0.035, 1.0
K = 12; wo = rng.uniform(-0.5, 0.5, K); wa = rng.uniform(0.2, 1.0, K); ww = rng.uniform(0.9, 2.8, K)
pw = cairo.Context(paint); pw.scale(SS, SS); pw.set_line_cap(cairo.LINE_CAP_ROUND)
def wheel(i):
    global x, y, th, curv, load
    ai = a[i]
    if onz[i] > 0.62 and rng.random() < 0.5:
        curv = rng.uniform(0.03, 0.085)*rng.choice([-1, 1]); load = min(1, load+0.65)
    curv *= 0.993
    steer = curv+0.04*np.clip(m[i]-m[i-1], -2, 2)
    ext = max(150, abs(CUR[i]-H/2)-20)                   # drive only on what is already painted
    cx, cy, rx, ry = W/2, H/2, W*0.46, ext
    dx, dy = cx-x, cy-y; d = math.hypot(dx/rx, dy/ry)
    if d > 0.62:
        want = math.atan2(dy, dx); diff = (want-th+math.pi) % (2*math.pi)-math.pi
        steer += diff*0.09*min(1.5, (d-0.62)/0.38)
    th += steer
    v = 3+11*ai
    ramp = min(1, (i-WH0)/(3*FPS), (WH1-i)/(4*FPS))
    col = P['white']
    for _ in range(3):
        nx, ny = x+math.cos(th)*v/3, y+math.sin(th)*v/3
        w = (5+26*ai)*(0.35+0.65*load)
        px, py = -math.sin(th), math.cos(th)
        for k in range(K):
            o = wo[k]*w
            pw.set_source_rgba(*col, wa[k]*(0.15+0.85*load)*ramp)
            pw.set_line_width(ww[k]); pw.move_to(x+px*o, y+py*o); pw.line_to(nx+px*o, ny+py*o); pw.stroke()
        x, y = nx, ny
    load = max(0.0, load-0.008-0.008*ai)
    if onz[i] > 0.85:
        for _ in range(int(5+16*onz[i])):
            r = rng.gamma(2, 1.5); dist = rng.gamma(2, 20); ang = th+rng.normal(0, 1.0)
            pw.set_source_rgba(*P[rng.choice(['white', 'white', 'yellow', 'red', 'black'])], rng.uniform(0.6, 1)*ramp)
            pw.arc(x+math.cos(ang)*dist, y+math.sin(ang)*dist, r, 0, 2*math.pi); pw.fill()

# ---------- lattice and the red band ----------
LAT0 = int(124*FPS)
lattice = [((-120, -60), (1500, 1140)), ((420, 1140), (2040, 40)), ((-120, 760), (1200, -60)),
           ((900, -60), (2040, 860)), ((-120, 300), (700, 1140))]
RED0, RED1 = int(149*FPS), int(157*FPS)
def lat(i):
    t = (i-LAT0)/FPS; e = lambda u: 1-(1-u)**2.2
    for j, ((x0, y0), (x1, y1)) in enumerate(lattice):
        s, s0 = [np.clip((tt-j*1.6)/1.4, 0, 1) for tt in (t, t-1/FPS)]
        if s <= 0 or s0 >= 1: continue
        pc.set_source_rgba(0.04, 0.035, 0.04, 0.96); pc.set_line_width(24)
        pc.move_to(x0+(x1-x0)*e(s0), y0+(y1-y0)*e(s0)); pc.line_to(x0+(x1-x0)*e(s), y0+(y1-y0)*e(s)); pc.stroke()
def red(i):
    s, s0 = [np.clip((ii-RED0)/(RED1-RED0), 0, 1) for ii in (i, i-1)]
    xa, xb = -10+ease(s0)*(W+20), -10+ease(s)*(W+20)
    if xb <= xa: return
    xs = np.arange(xa, xb+5, 5)
    for b in range(46):
        yy = H/2+(b/45-0.5)*60
        nz = noise1(xs, b*13.7, 70)
        for q in range(len(xs)-1):
            al = (0.7+0.3*nz[q]) if 3 < b < 42 else 0.6*np.clip((nz[q]-0.25)*2, 0, 1)
            pc.set_source_rgba(*P['red'], al); pc.set_line_width(2.2)
            pc.move_to(xs[q], yy); pc.line_to(min(xs[q+1], xb), yy); pc.stroke()

# ---------- camera ----------
HEAD = np.full(N, W/2)
for st in strokes:
    pass
def zoom_at(i):
    t = i/FPS
    if t < 14: return 2.7
    if t < 116: return 2.7-1.7*ease((t-14)/102)
    return 1.0
FOCY = np.convolve(np.pad(CUR, 90, mode='edge'), np.ones(181)/181, 'valid')

yy, xx = np.mgrid[0:H, 0:W]
vig = (1-0.40*(((xx-W/2)/(W*0.66))**2+((yy-H/2)/(H*0.66))**2)).clip(0.3, 1).astype(np.float32)[..., None]
weave = (1+0.045*np.sin(xx*1.9)*np.sin(yy*1.7)+rng.normal(0, 0.03, (H, W))).astype(np.float32)[..., None]
tex = vig*weave
grains = [rng.normal(0, 5, (H, W, 1)).astype(np.float32) for _ in range(5)]
BG = (0.028, 0.028, 0.036)
from scipy.ndimage import zoom as nzoom, gaussian_filter
def layer(sc, sig):
    n = rng.random((H*SS//sc+2, W*SS//sc+2)).astype(np.float32)
    n = nzoom(n, sc, order=1)[:H*SS, :W*SS]; return gaussian_filter(n, sig)
t1 = layer(3, 0.8); t2 = layer(1, 0.0); t3 = layer(14, 3)
hs = rng.random((H*SS, W*SS//24+2)).astype(np.float32); hs = gaussian_filter(nzoom(hs, (1, 24), order=1)[:, :W*SS], (0.6, 4))
Y2, X2 = np.mgrid[0:H*SS, 0:W*SS]
weave2 = 0.5+0.5*np.sin(X2*0.55)*np.sin(Y2*0.6)
tooth = (0.28*t1/t1.max()+0.22*t2+0.08*weave2+0.27*hs/hs.max()+0.15*t3/t3.max())
tooth = np.clip((tooth-0.16)*2.3, 0.0, 1.0)**0.5
del hs
del Y2, X2, weave2, t1, t2, t3
stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, W*SS)
tbuf = np.zeros((H*SS, stride), np.uint8); tbuf[:, :W*SS] = (tooth*255).astype(np.uint8)
toothS = cairo.ImageSurface.create_for_data(tbuf, cairo.FORMAT_A8, W*SS, H*SS, stride)
frame = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); fc = cairo.Context(frame)
fx_s = W/2
def compose(f):
    global fx_s
    i = f-PRE; ii = min(max(i, 0), N-1)
    fc.set_source_rgb(*BG); fc.paint()
    z = zoom_at(ii)
    vh, vw = H/z, W/z
    # keep the band being drawn in the lower third of the view; history above it
    fy = H/2
    hx = next((st['head'] for st in strokes if st['s0'] <= ii <= st['s1'] and st['head'] is not None), W/2)
    fx_s += (hx-fx_s)*0.012
    fx = float(np.clip(fx_s, vw/2, W-vw/2))
    dim = 1.0 if i < N+1.5*FPS else max(0, 1-(i-N-1.5*FPS)/(3.0*FPS))
    fc.save(); fc.translate(W/2, H/2); fc.scale(z/SS, z/SS); fc.translate(-fx*SS, -fy*SS)
    fc.push_group(); fc.set_source_surface(paint, 0, 0); fc.get_source().set_filter(cairo.FILTER_GOOD)
    fc.mask_surface(toothS, 0, 0); fc.pop_group_to_source(); fc.identity_matrix(); fc.paint_with_alpha(dim); fc.restore()
    if f >= PRE+N+int(4.8*FPS):
        t = (f-(PRE+N+4.8*FPS))/FPS; al = np.clip(t/0.9, 0, 1)
        fc.select_font_face('Noto Serif CJK KR'); fc.set_font_size(34); fc.set_source_rgba(0.92, 0.9, 0.86, al*0.9)
        s = '활 과 바 퀴'; e = fc.text_extents(s); fc.move_to(W/2-e.width/2-e.x_bearing, H/2+6); fc.show_text(s)
        fc.set_font_size(16); fc.set_source_rgba(0.92, 0.9, 0.86, al*0.45)
        s = f'김종훈 첼로 연주 · {S}번의 활'; e = fc.text_extents(s); fc.move_to(W/2-e.width/2-e.x_bearing, H/2+50); fc.show_text(s)
    frame.flush()
    arr = np.ndarray((H, W, 4), np.uint8, frame.get_data()).astype(np.float32)
    arr[..., :3] = arr[..., :3]*tex+grains[f % 5]
    return np.clip(arr, 0, 255).astype(np.uint8)

def advance(i):
    if 0 <= i < N:
        for st in strokes:
            if st['s0'] <= i <= st['s1']+2: draw_stroke(st, i)
        if WH0 <= i < WH1: wheel(i)
        if LAT0 <= i < LAT0+12*FPS: lat(i)
        if RED0 <= i <= RED1+1: red(i)

if sys.argv[1] == 'stills':
    want = sorted(int(float(s)*FPS)+PRE for s in sys.argv[2:])
    from PIL import Image
    for f in range(TOT):
        advance(f-PRE)
        if f in want:
            Image.fromarray(compose(f)[..., [2, 1, 0]]).save(f'v5_{(f-PRE)/FPS:05.1f}.png')
            if f == want[-1]: break
else:
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-i', sys.argv[2], '-af', f'adelay={int(PRE/FPS*1000)}:all=1,apad', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k', '-t', f'{TOT/FPS:.3f}', '-movflags', '+faststart', sys.argv[3]], stdin=subprocess.PIPE)
    for f in range(TOT):
        advance(f-PRE); p.stdin.write(compose(f).tobytes())
        if f % 900 == 0: print(f, '/', TOT, flush=True)
    p.stdin.close(); p.wait()
