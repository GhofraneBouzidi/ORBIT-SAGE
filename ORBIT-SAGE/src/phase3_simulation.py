# ORBIT-SAGE: contact-window, priority-queue and battery simulation
import argparse, json
from pathlib import Path
INFER_W, RADIO_W = (7.0, 3.0)
CRIT_SOC = 0.2
PRIORITY = {'M1': 0, 'M2': 1, 'M3': 2, 'M0': 3}
SCENARIOS = [('Nominal', 15, 50000, 1.0, 'all'), ('Link-constrained', 15, 10000, 1.0, 'all'), ('Energy-constrained', 3, 50000, 1.0, 'all'), ('Critical-energy', 1, 10000, 0.15, 'all'), ('High vessel density', 15, 50000, 1.0, 'dense'), ('No-vessel scenes', 15, 50000, 1.0, 'empty')]

def meta_bytes(r):
    return 120 + 50 * max(r['ships'], 1)

def run(rows, bw, wh, soc0, window_s, policy):
    battery_j = wh * 3600.0
    soc = soc0
    order = list(rows)
    if policy == 'priority':
        order.sort(key=lambda r: (PRIORITY[r['mode']], r['packet_bytes']))
    t = 0.0
    sent = deferred = stored = 0
    bytes_sent = 0
    energy = 0.0
    found = 0
    recov_bytes = 0
    for r in order:
        if policy != 'raw':
            energy += INFER_W * r['inference_ms'] / 1000.0
        crit = (soc0 * battery_j - energy) / battery_j < CRIT_SOC
        if policy == 'raw':
            size, got = (r['raw_bytes'], r['ships'])
        elif crit and r['mode'] == 'M3':
            stored += 1
            recov_bytes += r['packet_bytes']
            continue
        elif crit:
            size, got = (meta_bytes(r), r['ships_delivered'])
        else:
            size, got = (r['packet_bytes'], r['ships_delivered'])
        dt = size / bw
        if t + dt > window_s:
            deferred += 1
            continue
        t += dt
        energy += RADIO_W * dt
        bytes_sent += size
        found += got
        sent += 1
        if policy != 'raw' and r['mode'] == 'M2':
            recov_bytes += r['raw_bytes']
    total_ships = sum((r['ships'] for r in rows))
    return dict(images=len(rows), sent=sent, deferred_by_window=deferred, stored_safe_mode=stored, kb_sent=round(bytes_sent / 1024, 1), link_s=round(t, 1), energy_j=round(energy, 1), battery_used_pct=round(100 * energy / battery_j, 2), ships_delivered_pct=round(100 * found / total_ships, 1) if total_ships else None, recovery_if_requested_kb=round(recov_bytes / 1024, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/content/drive/MyDrive/orbit_runs/ship_final/pipeline_results.json')
    ap.add_argument('--window_s', type=float, default=120.0, help='contact window length (configurable prototype value)')
    ap.add_argument('--out', default='phase3_results.json')
    a = ap.parse_args()
    rows = json.loads(Path(a.results).read_text())['per_image']
    ships_sorted = sorted((r['ships'] for r in rows))
    dense_cut = ships_sorted[int(0.66 * len(rows))]
    out = {}
    print(f"{'scenario':20s}{'policy':10s}{'sent':>5s}{'defer':>6s}{'store':>6s}{'KB':>9s}{'link s':>8s}{'J':>8s}{'batt%':>7s}{'ships%':>8s}")
    for name, wh, bw, soc0, sub in SCENARIOS:
        sel = rows if sub == 'all' else [r for r in rows if r['ships'] >= dense_cut] if sub == 'dense' else [r for r in rows if r['ships'] == 0]
        if not sel:
            out[name] = 'no images of this type in the test set (HRSC2016 has no ship-free images)'
            print(f'{name:20s}n/a - no such images in the test set')
            continue
        out[name] = {}
        for pol in ('raw', 'fifo', 'priority'):
            m = run(sel, bw, wh, soc0, a.window_s, pol)
            out[name][pol] = m
            print(f"{name:20s}{pol:10s}{m['sent']:5d}{m['deferred_by_window']:6d}{m['stored_safe_mode']:6d}{m['kb_sent']:9.1f}{m['link_s']:8.1f}{m['energy_j']:8.1f}{m['battery_used_pct']:7.2f}{str(m['ships_delivered_pct']):>8s}")
    out['assumptions'] = dict(window_s=a.window_s, inference_w=INFER_W, radio_w=RADIO_W, critical_soc=CRIT_SOC, critical_scenario_start_soc=0.15, metadata_bytes='120 + 50 per ship (estimate)', priority='M1 > M2 > M3 > M0, then smaller packet first', not_implemented='novelty detection, embedded hardware')
    Path(a.out).write_text(json.dumps(out, indent=2))
    print('saved', a.out)
if __name__ == '__main__':
    main()
