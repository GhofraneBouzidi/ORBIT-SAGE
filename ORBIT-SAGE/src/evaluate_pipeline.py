# ORBIT-SAGE: packet builder and Problem 7 evaluation on the test set
import argparse, io, json, time
from pathlib import Path
from PIL import Image
LINK_BPS = 50000
HIGH, MID, FLOOR = (0.9, 0.6, 0.25)

def jpeg_bytes(img, q):
    b = io.BytesIO()
    img.convert('RGB').save(b, 'JPEG', quality=q)
    return len(b.getvalue())

def build_packet_bytes(img, dets):
    top = max((d[4] for d in dets), default=0.0)
    meta = len(json.dumps({'ships': [[round(v, 1) for v in d] for d in dets]}).encode()) + 120
    if not dets:
        return ('M0', meta)
    crops = sum((jpeg_bytes(img.crop(tuple((int(v) for v in d[:4]))), 85) for d in dets))
    if top >= HIGH:
        return ('M1', meta + crops)
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    if top >= MID:
        return ('M2', meta + crops + jpeg_bytes(thumb, 60))
    return ('M3', meta + jpeg_bytes(img, 50))

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/content/hrsc_yolo')
    ap.add_argument('--weights', default='/content/drive/MyDrive/orbit_runs/ship_final/weights/best.pt')
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--out', default='/content/drive/MyDrive/orbit_runs/ship_final/pipeline_results.json')
    a, _ = ap.parse_known_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    rows = []
    tot = dict(raw=0, jpeg50=0, packet=0, gt=0, found=0, inf_ms=0.0)
    for ip in sorted((Path(a.data) / 'images' / 'test').glob('*.jpg')):
        img = Image.open(ip)
        W, H = img.size
        gts = []
        for l in (Path(a.data) / 'labels' / 'test' / f'{ip.stem}.txt').read_text().splitlines():
            _, cx, cy, w, h = map(float, l.split())
            gts.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        t = time.perf_counter()
        r = model.predict(str(ip), conf=FLOOR, imgsz=a.imgsz, verbose=False)[0]
        inf = (time.perf_counter() - t) * 1000
        dets = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]
        mode, pbytes = build_packet_bytes(img, dets)
        raw = ip.stat().st_size
        j50 = jpeg_bytes(img, 50)
        found = len(gts) if mode == 'M3' else sum((any((iou(d[:4], g) >= 0.5 for d in dets)) for g in gts))
        rows.append(dict(image=ip.name, mode=mode, raw_bytes=raw, packet_bytes=pbytes, reduction=1 - pbytes / raw, ships=len(gts), ships_delivered=found, inference_ms=inf))
        tot['raw'] += raw
        tot['jpeg50'] += j50
        tot['packet'] += pbytes
        tot['gt'] += len(gts)
        tot['found'] += found
        tot['inf_ms'] += inf
    n = len(rows)
    neg = [r for r in rows if r['ships'] == 0]
    modes = {m: sum((r['mode'] == m for r in rows)) for m in ('M0', 'M1', 'M2', 'M3')}
    res = dict(images=n, mode_counts=modes, data_reduction_vs_raw=1 - tot['packet'] / tot['raw'], data_reduction_vs_jpeg50=1 - tot['packet'] / tot['jpeg50'], mean_packet_kb=tot['packet'] / n / 1024, mean_raw_kb=tot['raw'] / n / 1024, information_preserved=tot['found'] / max(tot['gt'], 1), empty_scene_images=len(neg), empty_scene_false_alarm_rate=sum((r['mode'] != 'M0' for r in neg)) / len(neg) if neg else None, empty_scene_data_reduction=1 - sum((r['packet_bytes'] for r in neg)) / sum((r['raw_bytes'] for r in neg)) if neg else None, mean_inference_ms=tot['inf_ms'] / n, mean_link_seconds_raw=tot['raw'] / n / LINK_BPS, mean_link_seconds_packet=tot['packet'] / n / LINK_BPS, note='raw = original JPEG size; link at 50 KB/s; inference measured on this machine')
    Path(a.out).write_text(json.dumps(dict(summary=res, per_image=rows), indent=2))
    print(json.dumps(res, indent=2))
if __name__ == '__main__':
    main()
