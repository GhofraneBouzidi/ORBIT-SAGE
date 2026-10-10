# ORBIT-SAGE: export one real packet example per mode
import argparse, io, json, shutil, statistics
from pathlib import Path
from PIL import Image, ImageDraw
FLOOR = 0.25

def jpg(img, q):
    b = io.BytesIO()
    img.convert('RGB').save(b, 'JPEG', quality=q)
    return b.getvalue()

def build(img, dets, mode, outdir, tag):
    sizes = {}
    meta = json.dumps({'ships': [[round(v, 1) for v in d] for d in dets]}).encode()
    (outdir / f'{tag}_meta.json').write_bytes(meta)
    sizes['metadata (JSON)'] = len(meta) + 120
    if mode in ('M1', 'M2'):
        for k, d in enumerate(dets):
            c = jpg(img.crop(tuple((int(v) for v in d[:4]))), 85)
            (outdir / f'{tag}_crop{k + 1}.jpg').write_bytes(c)
            sizes[f'crop {k + 1}'] = len(c)
    if mode == 'M2':
        t = img.copy()
        t.thumbnail((256, 256))
        c = jpg(t, 60)
        (outdir / f'{tag}_thumb.jpg').write_bytes(c)
        sizes['thumbnail 256px'] = len(c)
    if mode == 'M3':
        c = jpg(img, 50)
        (outdir / f'{tag}_full_q50.jpg').write_bytes(c)
        sizes['full image JPEG-50'] = len(c)
    return sizes

def overlay(img, dets):
    o = img.convert('RGB').copy()
    d = ImageDraw.Draw(o)
    for x0, y0, x1, y1, c in dets:
        d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=max(3, img.width // 300))
        d.text((x0 + 4, max(0, y0 - 14)), f'ship {c:.2f}', fill=(255, 255, 0))
    return o

def pick(rows, mode):
    cand = [r for r in rows if r['mode'] == mode and r['ships'] in (1, 2, 3)]
    if not cand:
        cand = [r for r in rows if r['mode'] == mode]
    if not cand:
        return None
    med = statistics.median((r['packet_bytes'] for r in cand))
    return min(cand, key=lambda r: abs(r['packet_bytes'] - med))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/content/hrsc_yolo')
    ap.add_argument('--weights', default='/content/drive/MyDrive/orbit_runs/ship_final/weights/best.pt')
    ap.add_argument('--results', default='/content/drive/MyDrive/orbit_runs/ship_final/pipeline_results.json')
    ap.add_argument('--out', default='/content/orbit_examples')
    a, _ = ap.parse_known_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows = json.loads(Path(a.results).read_text())['per_image']
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    report = {}
    for mode in ('M1', 'M2', 'M3'):
        r = pick(rows, mode)
        if r is None:
            continue
        ip = Path(a.data) / 'images' / 'test' / r['image']
        img = Image.open(ip)
        res = model.predict(str(ip), conf=FLOOR, imgsz=800, verbose=False)[0]
        dets = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in res.boxes]
        tag = f'{mode}_{ip.stem}'
        shutil.copy(ip, out / f'{tag}_original.jpg')
        overlay(img, dets).save(out / f'{tag}_detections.jpg', quality=92)
        sizes = build(img, dets, mode, out, tag)
        raw = ip.stat().st_size
        total = sum(sizes.values())
        report[mode] = dict(image=ip.name, raw_bytes=raw, packet_bytes=total, pieces=sizes, reduction=1 - total / raw, top_conf=max((d[4] for d in dets), default=None), ships_detected=len(dets), ships_in_ground_truth=r['ships'])
        print(mode, ip.name, 'raw', raw, 'packet', total, f'-{100 * (1 - total / raw):.0f}%', sizes)
    (out / 'examples.json').write_text(json.dumps(report, indent=2))
    shutil.make_archive('/content/orbit_examples', 'zip', out)
    print('zip: /content/orbit_examples.zip')
if __name__ == '__main__':
    main()
