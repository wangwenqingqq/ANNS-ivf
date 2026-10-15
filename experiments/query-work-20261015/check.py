#!/usr/bin/env python3
"""CPU-only SIFT query-work diagnostic; no GPU libraries or timing claims."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import itertools
import io
import json
from pathlib import Path
import platform
import random
import sys
import numpy as np

SEED, LISTS, D, B, STEPS = 20261015, 64, 128, 512, 64
PROBES = (1, 2, 4, 8, 16, 32, 64)
HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_prefix(path, count, offset=0):
    size = count * (D + 1) * 4
    with path.open('rb') as stream:
        stream.seek(offset * (D + 1) * 4)
        raw = stream.read(size)
    assert len(raw) == size
    records = np.frombuffer(raw, dtype='<i4').reshape(count, D + 1)
    assert np.all(records[:, 0] == D), 'bad fvecs dimension'
    vectors = records[:, 1:].copy().view('<f4')
    assert np.isfinite(vectors).all()
    assert np.all(vectors >= 0) and np.all(vectors <= 255)
    assert np.all(vectors == np.floor(vectors)), 'integer SIFT contract'
    return vectors, {'file': 'sift_base.fvecs' if offset or count > 1000 else 'sift_query.fvecs',
                     'offset_rows': offset, 'rows': count, 'prefix_bytes': size,
                     'prefix_sha256': digest(raw)}


def distance(a, b):
    # Exact on bounded integer SIFT inputs in FP64; centroids need FP64 as well.
    result = np.maximum(0, (a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2*a@b.T)
    assert np.isfinite(result).all(), 'nonfinite L2 matrix'
    return result


def assign(x, centers):
    return np.concatenate([np.argmin(distance(block.astype(np.float64), centers), axis=1)
                           for block in np.array_split(x, max(1, (len(x)+4095)//4096))])


def train(x):
    centers = x[np.random.default_rng(SEED).choice(len(x), LISTS, replace=False)].astype(np.float64)
    for _ in range(12):
        labels = assign(x, centers)
        for c in range(LISTS):
            if np.any(labels == c):
                centers[c] = x[labels == c].astype(np.float64).mean(0)
    return centers


def fused(old, removed, incoming):
    final = len(old) - sum(v in removed for v in old) + len(incoming)
    out = [None] * final
    donors = []
    for i, uid in enumerate(old):
        if uid not in removed:
            if i < final:
                out[i] = uid
            else:
                donors.append(uid)
    holes = [i for i, uid in enumerate(out) if uid is None]
    assert len(holes) == len(incoming) + len(donors)
    for pos, uid in zip(holes, incoming + donors):
        out[pos] = uid
    assert None not in out
    return out, len(donors)


def parent_fused(old, removed, incoming):
    # Literal final-prefix/hole/donor construction from the pinned parent planner.
    final = len(old) - sum(uid in removed for uid in old) + len(incoming)
    out = [None] * final
    donors = []
    for j, uid in enumerate(old):
        if uid not in removed:
            if j < final:
                out[j] = uid
            else:
                donors.append(j)
    holes = [j for j in range(final) if out[j] is None]
    fresh, holes = holes[:len(incoming)], holes[len(incoming):]
    assert len(holes) == len(donors)
    for dst, src in zip(holes, donors):
        out[dst] = old[src]
    for pos, uid in zip(fresh, incoming):
        out[pos] = uid
    return out, len(donors)


def reuse_holes(old, removed, incoming):
    out = [None if uid in removed else uid for uid in old]
    holes = [i for i, uid in enumerate(out) if uid is None]
    used = min(len(holes), len(incoming))
    for pos, uid in zip(holes[:used], incoming[:used]):
        out[pos] = uid
    out.extend(incoming[used:])
    while out and out[-1] is None:
        out.pop()
    return out


def validate(dense, sparse, expected, labels):
    for c in range(len(dense)):
        assert set(dense[c]) == {x for x in sparse[c] if x is not None}
        assert all(labels[x] == c for x in dense[c])
        assert not sparse[c] or sparse[c][-1] is not None
    flat = [x for row in dense for x in row]
    assert len(flat) == len(set(flat)) == len(expected)
    assert set(flat) == expected, 'live ID conservation'
    sparse_flat = [x for row in sparse for x in row if x is not None]
    assert len(sparse_flat) == len(set(sparse_flat)) == len(expected), 'sparse ID conservation'


def self_test():
    cases = 0
    for n in range(8):
        old = list(range(n))
        for bits in itertools.product((False, True), repeat=n):
            gone = {uid for uid, bit in zip(old, bits) if bit}
            for count in range(5):
                incoming = list(range(100, 100+count))
                actual, moves = fused(old, gone, incoming)
                assert (actual, moves) == parent_fused(old, gone, incoming)
                needed = set(old) - gone
                # Every surviving row outside the final dense prefix must move;
                # the fused plan attains that bound with arbitrary row ordering.
                bound = sum(uid not in gone and pos >= len(actual)
                            for pos, uid in enumerate(old))
                assert moves == bound
                sparse = reuse_holes(old, gone, incoming)
                assert set(actual) == {x for x in sparse if x is not None} == needed | set(incoming)
                cases += 1
    assert reuse_holes([0, None, 2, None], {2}, [3]) == [0, 3]
    labels = np.array([0, 0, 0])
    try:
        validate([[0, 2]], [[0, 1]], {0, 2}, labels)
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid ID mutation escaped')
    try:
        validate([[0, 1, 2]], [[0, 1, 2, 2]], {0, 1, 2}, labels)
    except AssertionError as error:
        assert str(error) == 'sparse ID conservation'
    else:
        raise AssertionError('duplicate sparse ID mutation escaped')
    rng = np.random.default_rng(SEED)
    x = rng.integers(0, 256, (65, D)).astype(np.float64)
    q = rng.integers(0, 256, (9, D)).astype(np.float64)
    direct = ((q[:, None, :].astype(np.int64) - x[None, :, :].astype(np.int64))**2).sum(2)
    assert np.array_equal(distance(q, x), direct)
    for seed in range(20):
        rng_py = random.Random(seed)
        dense, sparse, next_id = [[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 4, 5]], 6
        owner = [0, 0, 0, 1, 1, 1]
        expected = set(range(6))
        for _ in range(100):
            gone = set(rng_py.sample(sorted(expected), 2))
            new = [next_id, next_id+1]; next_id += 2
            owner.extend([rng_py.randrange(2), rng_py.randrange(2)])
            for c in range(2):
                added = [i for i in new if owner[i] == c]
                dense[c], _ = fused(dense[c], gone, added)
                sparse[c] = reuse_holes(sparse[c], gone, added)
            expected = (expected - gone) | set(new)
            validate(dense, sparse, expected, owner)
    toy_dense = [list(range(16))] + [[] for _ in range(LISTS-1)]
    toy_sparse = [[0, None] + list(range(1,16))] + [[] for _ in range(LISTS-1)]
    toy = snapshot(0, set(range(16)), toy_dense, toy_sparse, {'original':[0]},
                   np.arange(LISTS)[None,:], {0:np.arange(16,dtype=np.float64)}, io.StringIO(),
                   {'n':16, 'workload':'test', 'labels':np.zeros(16,dtype=np.int64)})
    assert toy['queries']['original']['curve'][0]['metadata_amplification'] == 17/16
    assert toy['queries']['original']['curve'][-1]['recall'] == 1.0
    print('PASS: CPU fused/hole/L2/negative-ID/query-path checks', cases)
    return {'exhaustive_fused_cases': cases, 'random_traces': 20,
            'events_per_random_trace': 100, 'direct_integer_l2': True,
            'invalid_id_mutation_rejected': True, 'physical_query_path': True, 'sparse_duplicate_rejected': True}


def quantiles(values):
    return {'mean': float(np.mean(values)), 'p50': float(np.quantile(values, .5, method='higher')),
            'p99': float(np.quantile(values, .99, method='higher')), 'max': int(np.max(values))}


def snapshot(step, live, dense, sparse, qids, routes, distances, raw, prefix):
    current = np.array(sorted(live), dtype=np.int64)
    counts = np.array([len(row) for row in dense])
    extents = np.array([len(row) for row in sparse])
    answer = {}
    for traffic, selection in qids.items():
        rows = []
        for qi in selection:
            scores = distances[qi][current]
            ordered = current[np.lexsort((current, scores))]
            ground = ordered[:10]
            ranks = np.empty(LISTS, dtype=np.int64)
            ranks[routes[qi]] = np.arange(LISTS)
            ordered_ranks = ranks[prefix['labels'][ordered]]
            live_work = np.cumsum(counts[routes[qi]])
            meta_work = np.cumsum(extents[routes[qi]])
            results, recalls = [], []
            for npb in PROBES:
                ids = ordered[ordered_ranks < npb][:10]
                # Explicitly reconstruct the same candidate set from physical holes.
                physical = np.array([uid for c in routes[qi, :npb]
                                     for uid in sparse[c] if uid is not None], dtype=np.int64)
                expected = physical[np.lexsort((physical, distances[qi][physical]))][:10]
                assert np.array_equal(ids, expected), 'physical selected-list top-k'
                assert len(set(ids.tolist())) == len(ids) and set(ids) <= live
                recall = len(set(ids) & set(ground)) / 10
                assert recall == sum(prefix['labels'][uid] in routes[qi, :npb] for uid in ground)/10
                results.append(ids.tolist()); recalls.append(recall)
            work = [int(live_work[p-1]) for p in PROBES]
            metadata = [int(meta_work[p-1]) for p in PROBES]
            assert all(m >= w for w, m in zip(work, metadata))
            row = {'n': prefix['n'], 'workload': prefix['workload'], 'step': step,
                   'traffic': traffic, 'query_row': int(qi), 'ground_truth': ground.tolist(),
                   'answers_by_nprobe': results, 'recall_by_nprobe': recalls,
                   'live_distances_by_nprobe': work, 'hole_metadata_by_nprobe': metadata,
                   'global_top10_probe_ranks': (ranks[prefix['labels'][ground]]+1).tolist(),
                   'all_live_prefix': live_work.tolist(), 'all_hole_prefix': meta_work.tolist()}
            raw.write(json.dumps(row, separators=(',', ':'))+'\n')
            rows.append(row)
        curve = []
        for i, npb in enumerate(PROBES):
            live_counts = [r['live_distances_by_nprobe'][i] for r in rows]
            meta_counts = [r['hole_metadata_by_nprobe'][i] for r in rows]
            curve.append({'nprobe': npb, 'recall': float(np.mean([r['recall_by_nprobe'][i] for r in rows])),
                          'live_distances': quantiles(live_counts), 'hole_metadata': quantiles(meta_counts),
                          'metadata_amplification': sum(meta_counts)/sum(live_counts),
                          'prefilter_distance_amplification': 1.0,
                          'late_validity_distance_amplification': sum(meta_counts)/sum(live_counts)})
        assert curve[-1]['recall'] == 1.0
        fine = []
        for npb in range(1, LISTS+1):
            fine.append({'nprobe': npb,
                'recall': float(np.mean([sum(rank <= npb for rank in r['global_top10_probe_ranks'])/10 for r in rows])),
                'mean_live_distances': float(np.mean([r['all_live_prefix'][npb-1] for r in rows])),
                'mean_hole_metadata': float(np.mean([r['all_hole_prefix'][npb-1] for r in rows]))})
            fine[-1]['metadata_amplification'] = fine[-1]['mean_hole_metadata']/fine[-1]['mean_live_distances']
        for point in curve:
            f = fine[point['nprobe']-1]
            assert f['recall'] == point['recall']
            assert f['mean_live_distances'] == point['live_distances']['mean']
            assert f['mean_hole_metadata'] == point['hole_metadata']['mean']
            assert f['metadata_amplification'] == point['metadata_amplification']
        answer[traffic] = {'curve': curve, 'matched_mean_recall': {
            str(target): next(r for r in curve if r['recall'] >= target) for target in (.90, .95)},
            'supplement_all_integer_curve': fine,
            'supplement_matched_mean_recall': {
                str(target): next(r for r in fine if r['recall'] >= target) for target in (.90, .95)}}
    return {'step': step, 'dense_rows': int(counts.sum()), 'hole_extent_rows': int(extents.sum()),
            'hole_rows': int(extents.sum()-counts.sum()), 'queries': answer}


def run(base_path, query_path, out):
    assert not out.exists(), 'preserve previous runs; choose a fresh output directory'
    out.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    tests = self_test()
    base, base_meta = read_prefix(base_path, 262144)
    learn, learn_meta = read_prefix(base_path, 8192, offset=262144)
    queries, query_meta = read_prefix(query_path, 1000)
    centers = train(learn)
    labels = assign(base, centers)
    routes = np.argsort(distance(queries.astype(np.float64), centers), axis=1, kind='stable')
    # Queries used below are selected before traces. Only these distances are materialized.
    selections = {}
    for n in (32768, 65536):
        reservoir = np.arange(n, len(base))
        hist = np.bincount(labels[reservoir], minlength=LISTS)
        hot = np.lexsort((np.arange(LISTS), -hist))[:16]
        hot_q = np.flatnonzero(np.isin(routes[:, 0], hot))[:64]
        assert len(hot_q) == 64
        selections[n] = (hot, {'original': list(range(64)), 'hot': hot_q.tolist()})
    active_q = sorted({q for _, selection in selections.values() for ids in selection.values() for q in ids})
    distances = {q: distance(queries[q:q+1].astype(np.float64), base.astype(np.float64))[0]
                 for q in active_q}
    # On real input, independently compare a deterministic subset to subtraction L2.
    ix = np.arange(0, len(base), 997)
    for qi in active_q[:4]:
        direct = ((base[ix].astype(np.int64)-queries[qi].astype(np.int64))**2).sum(1)
        assert np.array_equal(distances[qi][ix], direct)
    result = {'experiment': 'ivf_20261015_query_work_sift128_cpu', 'kind': 'CPU operation counts only',
              'tests': tests, 'source_sha256': digest(source),
              'python': platform.python_version(), 'numpy': np.__version__,
              'input_prefixes': [base_meta, learn_meta, query_meta],
              'centroids_f64_sha256': digest(centers.astype('<f8').tobytes()),
              'base_labels_sha256': digest(labels.astype('<i8').tobytes()),
              'probes': list(PROBES), 'cases': [], 'gpu_executed': False,
              'novel_mechanism_admitted': False}
    with (out/'QUERY.jsonl').open('w') as raw:
        for n in (32768, 65536):
            hot, qids = selections[n]
            for workload in ('random', 'fifo', 'cross-hot'):
                rng = np.random.default_rng(SEED)
                reservoir = np.arange(n, len(base))
                if workload == 'cross-hot':
                    reservoir = reservoir[np.isin(labels[reservoir], hot)]
                new_stream = rng.permutation(reservoir)[:B*STEPS]
                assert len(new_stream) == B*STEPS
                initial = [np.flatnonzero(labels[:n] == c).tolist() for c in range(LISTS)]
                dense, sparse = [r[:] for r in initial], [r[:] for r in initial]
                live = set(range(n)); epochs = []; snapshots = []
                peak_rows = sum(map(len, sparse)); high = [len(r) for r in sparse]
                prefix = {'n': n, 'workload': workload, 'labels': labels}
                snapshots.append(snapshot(0, live, dense, sparse, qids, routes, distances, raw, prefix))
                for step in range(1, STEPS+1):
                    candidates = sorted(live)
                    gone = set(candidates[:B] if workload == 'fifo' else rng.choice(candidates, B, replace=False).tolist())
                    added = new_stream[(step-1)*B:step*B].tolist()
                    assert len(gone) == B and len(set(added)) == B and not (set(added) & live)
                    moves = 0
                    for c in range(LISTS):
                        new = [uid for uid in added if labels[uid] == c]
                        dense[c], moved = fused(dense[c], gone, new); moves += moved
                        sparse[c] = reuse_holes(sparse[c], gone, new)
                    live = (live-gone) | set(added)
                    validate(dense, sparse, live, labels)
                    extent = sum(map(len, sparse))
                    peak_rows = max(peak_rows, extent)
                    high = [max(h, len(row)) for h, row in zip(high, sparse)]
                    epochs.append({'step': step, 'ack_replacements': B, 'failed': 0,
                                   'insert_payload_write_bytes': B*D*4,
                                   'dense_survivor_rows_moved': moves,
                                   'dense_survivor_payload_write_bytes': moves*D*4,
                                   'hole_survivor_rows_moved': 0,
                                   'logical_live': len(live), 'hole_extent_rows': extent})
                    if step in (16, 32, 64):
                        snapshots.append(snapshot(step, live, dense, sparse, qids, routes, distances, raw, prefix))
                case = {'n': n, 'workload': workload, 'hot_lists': hot.tolist(),
                        'insert_stream_sha256': digest(new_stream.astype('<i8').tobytes()),
                        'epochs': epochs, 'snapshots': snapshots, 'peak_hole_extent_rows': peak_rows,
                        'per_list_high_water_rows': high, 'ack_replacements': B*STEPS,
                        'dense_moved_survivors': sum(e['dense_survivor_rows_moved'] for e in epochs)}
                result['cases'].append(case)
                print(n, workload, 'moved/inserted', case['dense_moved_survivors'], '/', B*STEPS,
                      'final hole extent', len(live), '/', sum(map(len, sparse)), flush=True)
    assert Path(__file__).read_bytes() == source, 'source changed during run'
    (out/'RESULTS.json').write_text(json.dumps(result, indent=2)+'\n')
    inputs = {'mirror_revision': '0bc1ed67fc2626bc6f110c8794a70572681b7395',
              'base_url': 'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/0bc1ed67fc2626bc6f110c8794a70572681b7395/sift1M/sift_base.fvecs',
              'query_url': 'https://huggingface.co/datasets/hhy3/ann-datasets/resolve/0bc1ed67fc2626bc6f110c8794a70572681b7395/sift1M/sift_query.fvecs',
              'official_archive_equivalence_verified': False,
              'prefixes': result['input_prefixes']}
    (out/'INPUTS.json').write_text(json.dumps(inputs, indent=2)+'\n')
    print('PASS: full CPU work-count diagnostic; not a performance result')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--base', type=Path)
    p.add_argument('--queries', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.base and args.queries and args.out, '--base, --queries and --out required'
        run(args.base, args.queries, args.out)
