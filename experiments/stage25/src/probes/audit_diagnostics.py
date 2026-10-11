"""Independent saved-file audit of Stage25 A; never calls a model forward.

Run after both diagnostic JSON files exist. This reads immutable source evidence
and independently reconstructs the device gates, all vector predictions and
statistics. It does not import diagnostic.py, cache.py or the attribution core.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from scipy.ndimage import gaussian_filter

PROBES = Path(__file__).resolve().parent
ROOT = PROBES.parent
FILES = {}
CHECKS = []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_record(array):
    return {'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            'shape': list(array.shape), 'dtype': str(array.dtype)}


def check(condition, label):
    if not bool(condition):
        raise AssertionError(label)
    CHECKS.append(label)


def read_json(path):
    path = Path(path)
    FILES[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return json.loads(path.read_text())


def read_npz(path):
    path = Path(path)
    FILES[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    meta = json.loads(str(arrays.pop('metadata'))) if 'metadata' in arrays else None
    if meta and 'hashes' in meta:
        check(set(arrays) == set(meta['hashes']), f'{path.name}: exact hashed array set')
        for key, value in arrays.items():
            record = meta['hashes'][key]
            got = array_record(value)
            check(got == record if isinstance(record, dict) else got['sha256'] == record,
                  f'{path.name}/{key}: content SHA and declared shape/dtype')
            check(np.isfinite(value).all(), f'{path.name}/{key}: finite')
    return arrays, meta


def equal(actual, expected, label):
    check(actual.shape == expected.shape and actual.dtype == expected.dtype
          and np.array_equal(actual, expected)
          and array_record(actual)['sha256'] == array_record(expected)['sha256'],
          label + ': exact shape/dtype/content bytes (including signed zero)')


def stats(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    centered_a, centered_p = actual - actual.mean(), predicted - predicted.mean()
    denominator = max(float(np.linalg.norm(actual)), 1e-8)
    zero_variation = np.ptp(actual) == 0 or np.ptp(predicted) == 0
    pearson = None if zero_variation else float(
        np.dot(centered_a, centered_p) / (np.linalg.norm(centered_a) * np.linalg.norm(centered_p)))
    reason = 'Zero variation in actual and/or predicted vector' if zero_variation else None
    if pearson is not None and not np.isfinite(pearson):
        pearson, reason = None, 'Numerical correlation undefined despite nonconstant vectors'
    selected = np.maximum(np.abs(actual), np.abs(predicted)) > 1e-6
    included = int(selected.sum())
    match = np.sign(actual[selected]) == np.sign(predicted[selected])
    return {'count': len(actual), 'pearson': pearson, 'pearson_null_reason': reason,
            'relative_l2': float(np.linalg.norm(predicted - actual) / denominator),
            'relative_l2_denominator': denominator, 'mae': float(np.mean(np.abs(predicted - actual))),
            'sign_agreement': float(match.mean()) if included else None,
            'sign_null_reason': None if included else 'All pairs excluded by fixed near-zero criterion',
            'sign_threshold': 1e-6, 'sign_included_count': included,
            'sign_excluded_count': len(actual) - included, 'sign_mismatch_count': included - int(match.sum()),
            'actual_near_zero_count': int((np.abs(actual) <= 1e-6).sum()),
            'predicted_near_zero_count': int((np.abs(predicted) <= 1e-6).sum()),
            'actual_range': [float(actual.min()), float(actual.max())],
            'predicted_range': [float(predicted.min()), float(predicted.max())]}


def compare_structures(saved, computed, label):
    if isinstance(computed, dict):
        check(set(saved) == set(computed), label + ': field set')
        for key, value in computed.items():
            compare_structures(saved[key], value, label + '/' + key)
    elif isinstance(computed, list):
        check(len(saved) == len(computed), label + ': list length')
        for index, value in enumerate(computed):
            compare_structures(saved[index], value, label + f'/{index}')
    elif isinstance(computed, float):
        check(isinstance(saved, (int, float)) and math.isclose(saved, computed, rel_tol=1e-11, abs_tol=1e-13), label)
    else:
        check(saved == computed, label)


def five_logits(cpu, mps):
    close = np.isclose(cpu, mps, rtol=1e-3, atol=1e-4)
    return {'shape': list(cpu.shape), 'finite': bool(np.isfinite(cpu).all() and np.isfinite(mps).all()),
            'all_five_logits_allclose': bool(np.isfinite(cpu).all() and np.isfinite(mps).all() and close.all()),
            'mismatching_logit_count': int((~close).sum()),
            'max_abs_logit_difference': float(np.max(np.abs(cpu.astype(np.float64) - mps.astype(np.float64)))),
            'predicted_class_same': bool(np.array_equal(cpu.argmax(axis=-1), mps.argmax(axis=-1))),
            'rtol': 1e-3, 'atol': 1e-4}


def signed_top(values, fraction):
    flat = values.reshape(1, -1)
    kept = np.zeros_like(flat)
    order = np.argsort(np.abs(flat[0]), kind='stable')
    selected = order[-math.ceil(flat.shape[1] * fraction):]
    kept[0, selected] = flat[0, selected]
    return kept.reshape(values.shape)


def postprocess(raw, suffix):
    if suffix == 'post22':
        selected = signed_top(raw, .25)
        padded = torch_functional.pad(torch.from_numpy(selected), (2, 2, 2, 2), mode='reflect')
        smooth = torch_functional.avg_pool2d(padded, 5, stride=1).numpy()
        return np.ascontiguousarray(signed_top(smooth, .10), dtype=np.float32)
    amplitude = gaussian_filter(np.abs(raw[0]).astype(np.float64), sigma=(0, 1, 1), mode='reflect', truncate=4.)
    cutoff = float(np.quantile(amplitude, .85, method='linear'))
    gate = np.maximum(1. - cutoff / np.maximum(amplitude, 1e-12), 0.)
    return np.ascontiguousarray(raw[0].astype(np.float64) * gate, dtype=np.float32)[None]


def verify_probe(label, device, inputs, indices, prereg):
    arrays, meta = read_npz(PROBES / f'{device}_{label}.npz')
    saved_json = read_json(PROBES / f'{device}_{label}.json')
    check(meta == saved_json, f'{device}/{label}: NPZ metadata equals JSON')
    check(meta['status'] == 'COMPLETED', f'{device}/{label}: complete')
    equal(arrays['input'], inputs[label], f'{device}/{label}: complete input binding')
    equal(arrays['target'], inputs['target'], f'{device}/{label}: target binding')
    check(meta['identity']['device'] == device and meta['identity']['precision'] == 'fp32',
          f'{device}/{label}: explicit device and precision')
    check(meta['identity']['diagnostic_source_sha256'] == prereg['program_sha256']['diagnostic.py'],
          f'{device}/{label}: frozen source binding')
    check(meta['identity']['input'] == array_record(inputs[label])
          and meta['identity']['target'] == array_record(inputs['target'])
          and meta['identity']['baseline'] == float(inputs[label].min()), f'{device}/{label}: complete identity')
    check(meta['identity']['indices'] == {key: array_record(value) for key, value in indices.items()},
          f'{device}/{label}: all preregistered index identities')
    expected = {'input': (1, 3, 224, 224), 'original_logits': (1, 5),
                'single_logits': (32 if device == 'cpu' else 384, 5)}
    expected['tile_logits' if device == 'cpu' else 'group_logits'] = (128, 5)
    for key, shape in expected.items():
        check(arrays[key].shape == shape and arrays[key].dtype == np.float32,
              f'{device}/{label}/{key}: required full five-logit shape/FP32')
    equal(arrays['single_indices'], indices['single_indices'][:32] if device == 'cpu' else indices['single_indices'],
          f'{device}/{label}: singleton order')
    equal(arrays['tile_ids' if device == 'cpu' else 'group_indices'],
          indices['tile_ids' if device == 'cpu' else 'group_indices'], f'{device}/{label}: tile/group order')
    check(meta['planned_masks'] == meta['completed_masks'] == (160 if device == 'cpu' else 512),
          f'{device}/{label}: exact mask counts')
    for kind in ('single', 'tile' if device == 'cpu' else 'group'):
        position = 0
        records = [record for record in meta['batch_records'] if record['kind'] == kind]
        for record in records:
            count = record['count']
            check(record['start'] == position and record['inference_batch'] in (64, 32, 16)
                  and 0 < count <= record['inference_batch'], f'{device}/{label}/{kind}: contiguous batch record')
            check(array_record(arrays[kind + '_logits'][position:position + count])['sha256'] == record['logits_sha256'],
                  f'{device}/{label}/{kind}: batch content SHA')
            position += count
        check(position == len(arrays[kind + '_logits']), f'{device}/{label}/{kind}: complete batch record coverage')
    return arrays, meta


def audit_case(label, inputs, indices, prereg, acceptance):
    cpu, cm = verify_probe(label, 'cpu', inputs, indices, prereg)
    mps, mm = verify_probe(label, 'mps', inputs, indices, prereg)
    raw_path = Path(acceptance['raw_sources'][label]['path'])
    check(sha(raw_path) == acceptance['raw_sources'][label]['sha256'], label + ': raw source file SHA')
    raw4, rm = read_npz(raw_path)
    equal(raw4['input'], inputs[label], label + ': 4x4 input binding')
    equal(raw4['target'], inputs['target'], label + ': 4x4 target binding')
    for key, shape in (('raw', (1, 3, 224, 224)), ('logits', (1, 5)), ('masked_logits', (9408, 5))):
        check(raw4[key].shape == shape and raw4[key].dtype == np.float32, label + '/' + key + ': full 4x4 saved shape/dtype')
    target = int(inputs['target'][0])
    tile_drops = (np.float64(raw4['logits'][0, target]) - raw4['masked_logits'][:, target].astype(np.float64)) / 16
    rebuilt = np.repeat(np.repeat(tile_drops.astype(np.float32).reshape(3, 56, 56), 4, axis=1), 4, axis=2)[None]
    equal(raw4['raw'], rebuilt, label + ': 4x4 independent saved-logit rebuild bit exact')
    tile_ids = indices['tile_ids']
    cpu_drop = np.float64(cpu['original_logits'][0, target]) - cpu['tile_logits'][:, target].astype(np.float64)
    mps_drop = np.float64(raw4['logits'][0, target]) - raw4['masked_logits'][tile_ids, target].astype(np.float64)
    gate = read_json(PROBES / f'device_gate_{label}.json')
    gate_arrays, _ = read_npz(PROBES / f'device_gate_{label}.npz')
    pairs = {
        'original_cpu_vs_mps_small': (cpu['original_logits'], mps['original_logits']),
        'original_cpu_vs_mps_fullraw': (cpu['original_logits'], raw4['logits']),
        'single_first32': (cpu['single_logits'], mps['single_logits'][:32]),
        'tile_128': (cpu['tile_logits'], raw4['masked_logits'][tile_ids]),
    }
    comparisons = {name: five_logits(*values) for name, values in pairs.items()}
    compare_structures(gate['comparisons'], comparisons, label + ': independently recomputed all-five-logit gates')
    compare_structures(gate['tile_signed_drop'], stats(cpu_drop, mps_drop), label + ': independent signed-drop gate statistics')
    pass_gate = all(value['finite'] and value['all_five_logits_allclose'] and value['predicted_class_same']
                    for value in comparisons.values()) and stats(cpu_drop, mps_drop)['relative_l2'] <= .01
    check(gate['status'] == ('PASS' if pass_gate else 'FAIL'), label + ': gate decision matches frozen threshold')
    check(gate['acceptance']['logits_rtol'] == 1e-3 and gate['acceptance']['logits_atol'] == 1e-4
          and gate['acceptance']['tile_signed_drop_relative_l2_max'] == .01
          and gate['full_cpu_4x4_support_validation'] == 'NOT_PERFORMED_IN_THIS_STAGE', label + ': preregistered scope/thresholds')
    expected_gate_arrays = {
        'tile_ids': tile_ids, 'cpu_tile_signed_drop': cpu_drop, 'mps_tile_signed_drop': mps_drop,
        'cpu_original_logits': cpu['original_logits'], 'mps_original_logits': raw4['logits'],
        'cpu_single_logits': cpu['single_logits'], 'mps_single_logits': mps['single_logits'][:32],
        'cpu_tile_logits': cpu['tile_logits'], 'mps_tile_logits': raw4['masked_logits'][tile_ids]}
    check(set(gate_arrays) == set(expected_gate_arrays), label + ': exact gate vector set')
    for key, value in expected_gate_arrays.items():
        equal(gate_arrays[key], value, label + ': gate vector binding/' + key)
    historical, hm = read_npz(PROBES / f'historical_sources_{label}.npz')
    check(hm == read_json(PROBES / f'historical_sources_{label}.json'), label + ': historical evidence metadata binding')
    equal(historical['input'], inputs[label], label + ': historical input binding')
    equal(historical['target'], inputs['target'], label + ': historical target binding')
    gdd_rebuilt = (inputs[label] - np.float32(inputs[label].min())) * historical['gdd_gradient']
    equal(historical['gdd_black'], gdd_rebuilt, label + ': GDD raw saved-gradient rebuild')
    eight_drops = (np.float64(historical['occlusion8_original_logits'][0, target])
                   - historical['occlusion8_masked_logits'].astype(np.float64)) / 64
    eight_rebuilt = np.repeat(np.repeat(eight_drops.astype(np.float32).reshape(3, 28, 28), 8, axis=1), 8, axis=2)[None]
    equal(historical['black_occlusion8_signed'], eight_rebuilt, label + ': 8x8 raw saved-logit rebuild')
    for name, source in hm['sources'].items():
        source_path = Path(source['path'])
        check(sha(source_path) == source['file_sha256'], label + ': historical source file SHA/' + name)
        source_arrays, source_meta = read_npz(source_path)
        check(source_meta == source['metadata'], label + ': historical source metadata/' + name)
        equal(source_arrays['raw'], historical[name], label + ': historical raw source/' + name)
        if 'program_path' in source:
            check(sha(source['program_path']) == source['program_sha256'], label + ': historical actual program SHA/' + name)
    raw = {'gdd_black': historical['gdd_black'], 'black_occlusion8_signed': historical['black_occlusion8_signed'],
           'black_occlusion4_signed_v1': raw4['raw']}
    finals = {}
    for name, value in raw.items():
        for suffix in ('post22', 'soft15'):
            key = ('black_occlusion4_' + suffix) if name == 'black_occlusion4_signed_v1' else (name + '_' + suffix)
            finals[key] = postprocess(value, suffix)
            if key in historical:
                equal(historical[key], finals[key], label + ': independent final reconstruction/' + key)
    actual_single = np.float64(mps['original_logits'][0, target]) - mps['single_logits'][:, target].astype(np.float64)
    actual_group = np.float64(mps['original_logits'][0, target]) - mps['group_logits'][:, target].astype(np.float64)
    vectors, vector_meta = read_npz(PROBES / f'diagnostic_vectors_{label}.npz')
    diagnostic = read_json(PROBES / f'diagnostic_{label}.json')
    check(vector_meta == diagnostic, label + ': diagnostic vector metadata equals JSON')
    check(not diagnostic['diagnostic_is_official_faithfulness_replay']
          and not diagnostic['parameter_changes_based_on_diagnostic'], label + ': diagnostic interpretation scope')
    expected_vectors = {'single_indices': indices['single_indices'], 'group_indices': indices['group_indices'],
                        'true_single_drop': actual_single, 'true_group50_drop': actual_group}
    table = {'single_raw': {}, 'group50_raw': {}, 'group50_fixed_finals': {}}
    for name, value in raw.items():
        flat = value.reshape(-1).astype(np.float64)
        single = flat[indices['single_indices']]
        grouped = flat[indices['group_indices']].sum(axis=1, dtype=np.float64)
        expected_vectors['single_' + name] = single
        expected_vectors['group50_' + name] = grouped
        table['single_raw'][name] = stats(actual_single, single)
        table['group50_raw'][name] = stats(actual_group, grouped)
    for name, value in finals.items():
        grouped = value.reshape(-1).astype(np.float64)[indices['group_indices']].sum(axis=1, dtype=np.float64)
        expected_vectors['group50_final_' + name] = grouped
        table['group50_fixed_finals'][name] = stats(actual_group, grouped)
    check(set(vectors) == set(expected_vectors), label + ': exact diagnostic vector set')
    for key, value in expected_vectors.items():
        equal(vectors[key], value, label + ': independent diagnostic vector/' + key)
        check(diagnostic['vectors'][key] == array_record(value), label + ': diagnostic vector content record/' + key)
    compare_structures(diagnostic['tables'], table, label + ': independent diagnostic statistics')
    return {'device_gate': gate['status'], 'tile_drop_relative_l2': stats(cpu_drop, mps_drop)['relative_l2'],
            'mask_counts': {'cpu': cm['completed_masks'], 'mps': mm['completed_masks'], 'full4x4': 9408},
            'raw_source_sha256': sha(raw_path), 'diagnostic_file_sha256': sha(PROBES / f'diagnostic_{label}.json'),
            'single_raw': table['single_raw'], 'group50_raw': table['group50_raw'],
            'group50_fixed_finals': table['group50_fixed_finals']}


def write_result(result):
    path = PROBES / 'numerical_verification.json'
    descriptor, temporary = tempfile.mkstemp(prefix='.numerical_verification.', suffix='.tmp', dir=PROBES)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    started = time.monotonic()
    result = {'status': 'FAIL', 'model_forward_calls': 0, 'auditor_source_sha256': sha(Path(__file__))}
    try:
        inputs, _ = read_npz(PROBES / 'fixed_inputs.npz')
        indices, _ = read_npz(PROBES / 'frozen_indices.npz')
        prereg = read_json(ROOT / 'preregistered_method.json')
        sampling = read_json(PROBES / 'frozen_sampling.json')
        check(sha(PROBES / 'fixed_inputs.npz') == sampling['inputs_file_sha256']
              and sha(PROBES / 'frozen_indices.npz') == sampling['indices_file_sha256'], 'Frozen sampling source file SHAs')
        rng = np.random.Generator(np.random.PCG64(20261001))
        expected_single = np.concatenate([rng.choice(50176, 128, replace=False) + channel * 50176
                                          for channel in range(3)]).astype(np.int64)
        rng = np.random.Generator(np.random.PCG64(20261002))
        expected_groups = np.stack([rng.choice(150528, 50, replace=False) for _ in range(128)]).astype(np.int64)
        rng = np.random.Generator(np.random.PCG64(20261003))
        expected_tiles = rng.choice(9408, 128, replace=False).astype(np.int64)
        for key, expected in (('single_indices', expected_single), ('group_indices', expected_groups), ('tile_ids', expected_tiles)):
            equal(indices[key], expected, 'Independent frozen PCG64 reconstruction/' + key)
            check(array_record(expected) == sampling['arrays'][key], 'Frozen index full record/' + key)
        acceptance = read_json(PROBES / 'device_acceptance.json')
        result['cases'] = {label: audit_case(label, inputs, indices, prereg, acceptance)
                           for label in ('original', 'perturbed')}
        check(acceptance['status'] == ('PASS' if all(row['device_gate'] == 'PASS' for row in result['cases'].values()) else 'FAIL'),
              'Combined device acceptance matches independently reconstructed cases')
        result['status'] = 'PASS'
    except Exception as error:
        import traceback
        result['error'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc()
    result.update({'checks_passed': len(CHECKS), 'checks': CHECKS, 'source_files': FILES,
                   'audit_seconds': time.monotonic() - started})
    write_result(result)
    print(json.dumps({'status': result['status'], 'checks_passed': result['checks_passed'],
                      'audit_seconds': result['audit_seconds'], 'error': result.get('error')}, ensure_ascii=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
