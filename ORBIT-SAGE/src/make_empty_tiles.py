# ORBIT-SAGE: build ship-free test crops
import argparse, random, shutil
from pathlib import Path
from PIL import Image

def free_windows(W, H, boxes, size, tries, rng, margin=24):
    out = []
    for _ in range(tries):
        x0 = rng.randint(0, W - size)
        y0 = rng.randint(0, H - size)
        x1, y1 = (x0 + size, y0 + size)
        if all((bx1 + margin < x0 or bx0 - margin > x1 or by1 + margin < y0 or (by0 - margin > y1) for bx0, by0, bx1, by1 in boxes)):
            out.append((x0, y0, x1, y1))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/content/hrsc_yolo')
    ap.add_argument('--out', default='/content/hrsc_empty')
    ap.add_argument('--size', type=int, default=416)
    ap.add_argument('--per_image', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    a, _ = ap.parse_known_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / 'images' / 'test').mkdir(parents=True)
    (out / 'labels' / 'test').mkdir(parents=True)
    n = 0
    for ip in sorted((Path(a.data) / 'images' / 'test').glob('*.jpg')):
        img = Image.open(ip).convert('RGB')
        W, H = img.size
        if min(W, H) < a.size:
            continue
        boxes = []
        for l in (Path(a.data) / 'labels' / 'test' / f'{ip.stem}.txt').read_text().splitlines():
            _, cx, cy, w, h = map(float, l.split())
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        for k, win in enumerate(free_windows(W, H, boxes, a.size, 60, rng)[:a.per_image]):
            name = f'{ip.stem}_empty{k}'
            img.crop(win).save(out / 'images' / 'test' / f'{name}.jpg', quality=95)
            (out / 'labels' / 'test' / f'{name}.txt').write_text('')
            n += 1
    print('ship-free tiles:', n, '->', out)
if __name__ == '__main__':
    main()
