# ORBIT-SAGE: convert HRSC2016 to YOLO, train, tune threshold, export ONNX
import argparse, json, math, random, shutil, sys
import xml.etree.ElementTree as ET
from pathlib import Path
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def obb_to_hbb(cx, cy, w, h, ang):
    c, s = (math.cos(ang), math.sin(ang))
    xs, ys = ([], [])
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        xs.append(cx + dx * c - dy * s)
        ys.append(cy + dx * s + dy * c)
    return (min(xs), min(ys), max(xs), max(ys))

def parse_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    W = int(root.find('Img_SizeWidth').text)
    H = int(root.find('Img_SizeHeight').text)
    boxes = []
    for o in root.findall('HRSC_Objects/HRSC_Object'):
        g = lambda k: float(o.find(k).text)
        x0, y0, x1, y1 = obb_to_hbb(g('mbox_cx'), g('mbox_cy'), g('mbox_w'), g('mbox_h'), g('mbox_ang'))
        x0, y0, x1, y1 = (max(x0, 0), max(y0, 0), min(x1, W), min(y1, H))
        if x1 - x0 > 2 and y1 - y0 > 2:
            boxes.append(((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return boxes

def convert(root, out, split_mode='random', seed=42, neg_ratio=0.5):
    from PIL import Image
    root, out = (Path(root), Path(out))
    BAD = ('segment', 'mask', 'visual', 'vis', 'label')

    def ok(p):
        return not any((b in '/'.join(p.parts[len(root.parts):]).lower() for b in BAD))
    xmls = [p for p in root.rglob('*.xml') if 'Annotations' in p.parts and ok(p)]
    full = [p for p in xmls if 'FullDataSet' in p.parts]
    xmls = sorted(full or xmls)
    if not xmls:
        sys.exit(f'No Annotations/*.xml under {root}')
    cands = {}
    for p in root.rglob('*'):
        if p.suffix.lower() in IMG_EXTS and out not in p.parents and ok(p):
            cands.setdefault(p.stem, []).append(p)

    def pick(stem):
        c = cands.get(stem, [])
        pref = [p for p in c if any((k in p.parts for k in ('AllImages', 'JPEGImages')))]
        return ((pref or c or [None])[0], bool(pref))
    samples, imgs, nonpref = ({}, {}, 0)
    for x in xmls:
        ip, pref = pick(x.stem)
        if ip is None:
            continue
        try:
            b = parse_annotation(x)
        except Exception as e:
            print('skip', x.name, e)
            continue
        if b:
            samples[x.stem] = b
            imgs[x.stem] = ip
            nonpref += not pref
    print(f'{len(samples)} usable images (of {len(xmls)} annotations); {nonpref} not from AllImages/JPEGImages')
    if nonpref:
        print('WARNING: some images come from other folders - check them:', [str(imgs[k]) for k in list(samples) if not any((q in imgs[k].parts for q in ('AllImages', 'JPEGImages')))][:5])
    stems = sorted(samples)
    official = {}
    for name in ('train', 'val', 'test'):
        f = next(iter(root.rglob(f'ImageSets/{name}.txt')), None)
        if f:
            official[name] = [l.strip() for l in f.read_text().split() if l.strip() in samples]
    if split_mode == 'official' and len(official) == 3:
        splits = official
    else:
        if split_mode == 'official':
            print('official ImageSets not found -> random split')
        random.Random(seed).shuffle(stems)
        n = len(stems)
        a, b = (int(n * 0.8), int(n * 0.9))
        splits = {'train': stems[:a], 'val': stems[a:b], 'test': stems[b:]}
    if out.exists():
        shutil.rmtree(out)
    for sp, items in splits.items():
        (out / 'images' / sp).mkdir(parents=True)
        (out / 'labels' / sp).mkdir(parents=True)
        for st in items:
            Image.open(imgs[st]).convert('RGB').save(out / 'images' / sp / f'{st}.jpg', quality=95)
            (out / 'labels' / sp / f'{st}.txt').write_text(''.join((f'0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}\n' for a, b, c, d in samples[st])))
        nneg = 0
        rng = random.Random(seed + len(sp))
        for st in items:
            if rng.random() >= neg_ratio:
                continue
            im = Image.open(out / 'images' / sp / f'{st}.jpg')
            Wd, Hd = im.size
            boxes = [((cx - w / 2) * Wd, (cy - h / 2) * Hd, (cx + w / 2) * Wd, (cy + h / 2) * Hd) for cx, cy, w, h in samples[st]]
            for _ in range(40):
                ts = int(rng.uniform(0.3, 0.5) * min(Wd, Hd))
                x0 = rng.randint(0, Wd - ts)
                y0 = rng.randint(0, Hd - ts)
                if all((x0 + ts < b[0] - 10 or x0 > b[2] + 10 or y0 + ts < b[1] - 10 or (y0 > b[3] + 10) for b in boxes)):
                    im.crop((x0, y0, x0 + ts, y0 + ts)).save(out / 'images' / sp / f'{st}_neg.jpg', quality=95)
                    (out / 'labels' / sp / f'{st}_neg.txt').write_text('')
                    nneg += 1
                    break
        print(f'{sp}: {len(items)} images + {nneg} empty-scene tiles')
    yaml = out / 'dataset.yaml'
    yaml.write_text(f'path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n')
    return yaml

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0

def prf_at(per_image, thr):
    tp = fp = fn = 0
    for preds, gts in per_image:
        used = set()
        for p in sorted((p for p in preds if p[4] >= thr), key=lambda p: -p[4]):
            best, bj = (0.5, None)
            for j, g in enumerate(gts):
                if j not in used and iou(p[:4], g) >= best:
                    best, bj = (iou(p[:4], g), j)
            if bj is None:
                fp += 1
            else:
                tp += 1
                used.add(bj)
        fn += len(gts) - len(used)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return (tp, fp, fn, P, R, 2 * P * R / (P + R) if P + R else 0.0)

def collect(model, data_root, split, imgsz):
    from PIL import Image
    out = []
    for ip in sorted((Path(data_root) / 'images' / split).glob('*.jpg')):
        W, H = Image.open(ip).size
        gts = []
        for l in (Path(data_root) / 'labels' / split / f'{ip.stem}.txt').read_text().splitlines():
            _, cx, cy, w, h = map(float, l.split())
            gts.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        r = model.predict(str(ip), conf=0.001, imgsz=imgsz, verbose=False)[0]
        preds = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]
        out.append((preds, gts))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/content/HRSC2016_extracted')
    ap.add_argument('--data', default='/content/hrsc_yolo')
    ap.add_argument('--project', default='/content/drive/MyDrive/orbit_runs')
    ap.add_argument('--name', default='ship_final')
    ap.add_argument('--split', default='random', choices=['random', 'official'])
    ap.add_argument('--model', default='yolov8s.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--negatives', type=float, default=0.5, help='share of images that get a ship-free tile (0 = off)')
    ap.add_argument('--convert-only', action='store_true')
    ap.add_argument('--allow-cpu', action='store_true')
    a, _ = ap.parse_known_args()
    yaml = convert(a.root, a.data, a.split, neg_ratio=a.negatives)
    if a.convert_only:
        return
    import torch
    if not torch.cuda.is_available() and (not a.allow_cpu):
        sys.exit('NO GPU. Runtime > Change runtime type > T4 GPU, restart, rerun.')
    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience, project=a.project, name=a.name, exist_ok=True, cos_lr=True, seed=0, save_period=5, workers=2, plots=True, device=0 if torch.cuda.is_available() else 'cpu')
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / 'weights' / 'best.pt'
    print('BEST WEIGHTS:', best)
    m = YOLO(str(best))
    val_curve = collect(m, a.data, 'val', a.imgsz)
    grid = [round(t, 2) for t in [0.05 * i for i in range(1, 19)]]
    f1s = {t: prf_at(val_curve, t)[5] for t in grid}
    conf = max(f1s, key=f1s.get)
    tp, fp, fn, P, R, F1 = prf_at(collect(m, a.data, 'test', a.imgsz), conf)
    r = m.val(data=str(yaml), split='test', imgsz=a.imgsz, conf=0.001, plots=True)
    res = {'model': a.model, 'imgsz': a.imgsz, 'split': a.split, 'epochs_requested': a.epochs, 'test_mAP50': float(r.box.map50), 'test_mAP50_95': float(r.box.map), 'recommended_conf_from_val': conf, 'test_at_that_conf': {'TP': tp, 'FP': fp, 'FN': fn, 'precision': P, 'recall': R, 'f1': F1}}
    (save_dir / 'detector_metrics.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    m.export(format='onnx', imgsz=a.imgsz)
    print('Saved in', save_dir)
if __name__ == '__main__':
    main()
