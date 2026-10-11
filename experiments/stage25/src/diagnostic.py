"""Stage25 frozen finite-intervention probes and saved-vector diagnostics.

No global random generator is used.  A diagnostic correlation is not an
official FaithfulnessCorrelation replay or a criterion for changing the method.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT, WORKSPACE, WEIGHT_SHA, arrsha, atomic_json, atomic_npz, sha, sync, versions

IMAGE_SIZE = 224
ELEMENTS = 3 * IMAGE_SIZE * IMAGE_SIZE
TILE_COUNT = 3 * 56 * 56
INPUT_SHAS = {
    'original': 'ea4b08c3a5a18254ed0bb14e8aad0323b79ea07148c5f7ea6aba1401d12acc55',
    'perturbed': '7b524a39863423a3fdedf73b6612e37e6b89033de1755cb72922679f1869146b',
}
SOURCE_INPUT_SHA = 'f98772eb98aa2c50698d3f3ab4b513e785df03cc561f4de7ea28b86315ac6c64'
ACCEPTANCE = {'logits_rtol': 1e-3, 'logits_atol': 1e-4,
              'tile_signed_drop_relative_l2_max': .01, 'denominator_floor': 1e-8,
              'sign_near_zero_threshold': 1e-6, 'predicted_class_same': True,
              'all_logits_finite': True,
              'full_cpu_4x4_support_validation': 'NOT_PERFORMED_IN_THIS_STAGE'}


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: np.array(archive[k], copy=True) for k in archive.files}


def _array_record(array):
    return {'sha256': arrsha(array), 'shape': list(array.shape), 'dtype': str(array.dtype)}


def freeze_sampling(probe_dir=None):
    """Save the three independent, prescribed PCG64 samples before forward."""
    probe_dir = Path(probe_dir or ROOT / 'probes')
    probe_dir.mkdir(parents=True, exist_ok=True)
    source = WORKSPACE / 'output_mac_stage23/device/probe_inputs.npz'
    if sha(source) != SOURCE_INPUT_SHA:
        raise RuntimeError('Frozen Stage23 probe input file SHA differs')
    old = _load_npz(source)
    target = np.ascontiguousarray(old['target'], dtype=np.int64)
    if target.shape != (1,) or not 0 <= int(target[0]) < 5:
        raise RuntimeError('Frozen supplied target is invalid')
    inputs = {'target': target, 'noise': old['noise']}
    source_records = {'stage23_input_path': str(source), 'stage23_input_file_sha256': sha(source),
                      'target': _array_record(target), 'stage24_input_checks': {}}
    for label in ('original', 'perturbed'):
        x = np.ascontiguousarray(old[label], dtype=np.float32)[None]
        previous_path = WORKSPACE / f'output_mac_stage24/probes/mps_{label}.npz'
        previous = _load_npz(previous_path)
        meta = json.loads(str(previous['metadata']))
        if (x.shape != (1, 3, 224, 224) or arrsha(x) != INPUT_SHAS[label]
                or not np.array_equal(x, previous['input'])
                or int(meta['record']['target']) != int(target[0])
                or meta['hashes']['input'] != arrsha(x)):
            raise RuntimeError(f'Frozen Stage23/24 input or target differs: {label}')
        inputs[label] = x
        source_records['stage24_input_checks'][label] = {
            'path': str(previous_path), 'file_sha256': sha(previous_path),
            'input': _array_record(x), 'target': int(target[0]),
            'target_source': 'Stage23 complete int64 target; Stage24 record target equality',
            'device': meta['device'], 'result': 'PASS'}
    singleton_rng = np.random.Generator(np.random.PCG64(20261001))
    singles = np.concatenate([singleton_rng.choice(224 * 224, 128, replace=False)
                              + channel * 224 * 224 for channel in range(3)]).astype(np.int64)
    group_rng = np.random.Generator(np.random.PCG64(20261002))
    groups = np.stack([group_rng.choice(ELEMENTS, 50, replace=False)
                       for _ in range(128)]).astype(np.int64)
    tile_rng = np.random.Generator(np.random.PCG64(20261003))
    tiles = tile_rng.choice(TILE_COUNT, 128, replace=False).astype(np.int64)
    indices = {'single_indices': singles, 'group_indices': groups, 'tile_ids': tiles}
    sampling = {
        'schema': 1, 'frozen_before_new_forward': True,
        'sampling': {
            'single_indices': {'generator': 'Generator(PCG64(20261001))',
                'rule': 'c=0,1,2; 128 distinct HW positions per channel, preserving draw order'},
            'group_indices': {'generator': 'Generator(PCG64(20261002))',
                'rule': '128 sequential independent draws; each 50 distinct flattened CHW elements'},
            'tile_ids': {'generator': 'Generator(PCG64(20261003))',
                'rule': '128 distinct IDs in c,y,x order among 9408 nonoverlap 4x4 tiles'}},
        'arrays': {k: _array_record(v) for k, v in indices.items()},
        'acceptance': ACCEPTANCE, 'source_inputs': source_records,
        'independent_of_official_evaluation_seed': True,
        'diagnostic_source_sha256': sha(Path(__file__)),
    }
    # Repeated preparation is permitted only when the content is identical.
    for path, arrays in ((probe_dir / 'fixed_inputs.npz', inputs),
                         (probe_dir / 'frozen_indices.npz', indices)):
        if path.exists():
            saved = _load_npz(path)
            if set(saved) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise RuntimeError(f'Already frozen sampling differs: {path}')
        else:
            atomic_npz(path, **arrays)
    sampling['inputs_file_sha256'] = sha(probe_dir / 'fixed_inputs.npz')
    sampling['indices_file_sha256'] = sha(probe_dir / 'frozen_indices.npz')
    atomic_json(probe_dir / 'frozen_sampling.json', sampling)
    (probe_dir / 'frozen_sampling.sha256').write_text(
        sha(probe_dir / 'frozen_sampling.json') + '  frozen_sampling.json\n', encoding='utf-8')
    return inputs, indices, sampling


def read_sampling(probe_dir=None):
    probe_dir = Path(probe_dir or ROOT / 'probes')
    metadata = json.loads((probe_dir / 'frozen_sampling.json').read_text())
    for key, name in (('inputs_file_sha256', 'fixed_inputs.npz'),
                      ('indices_file_sha256', 'frozen_indices.npz')):
        if sha(probe_dir / name) != metadata[key]:
            raise RuntimeError(f'Frozen sampling file changed: {name}')
    inputs, indices = _load_npz(probe_dir / 'fixed_inputs.npz'), _load_npz(probe_dir / 'frozen_indices.npz')
    for key, array in indices.items():
        if _array_record(array) != metadata['arrays'][key]:
            raise RuntimeError(f'Frozen sampling array changed: {key}')
    return inputs, indices, metadata


def _tile_elements(tile_id):
    c, within = divmod(int(tile_id), 56 * 56)
    y, x = divmod(within, 56)
    return np.asarray([c * 224 * 224 + (y * 4 + dy) * 224 + x * 4 + dx
                       for dy in range(4) for dx in range(4)], dtype=np.int64)


def _clock_check(clock):
    if clock is not None:
        clock.check()


def _clock_update(clock, **fields):
    if clock is not None:
        clock.update(**fields)


def _check_model(model, device):
    if model.training:
        raise RuntimeError('Probe model must be eval')
    if next(model.parameters()).device.type != device:
        raise RuntimeError('Probe model device mismatch')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise RuntimeError('Probe model must be FP32')


def run_probes(model, x, target, device, indices, clock, batch=64,
               label='original', out_dir=None):
    """Run only the registered small probes; complete tile raw belongs to core.

    CPU: original five logits, first 32 singleton probes, 128 selected tiles.
    MPS: original five logits, all 384 singleton probes, 128 groups of 50.
    Partial batches are saved atomically and may resume only with exact identity.
    """
    if device not in ('cpu', 'mps') or batch not in (64, 32, 16):
        raise ValueError('Registered device and batch required')
    _check_model(model, device)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape == (3, 224, 224):
        x = x[None]
    target_array = np.ascontiguousarray(np.asarray(target, dtype=np.int64).reshape(1))
    t = int(target_array[0])
    if x.shape != (1, 3, 224, 224) or not np.isfinite(x).all() or not 0 <= t < 5:
        raise ValueError('Finite single NCHW input and supplied target required')
    out_dir = Path(out_dir or ROOT / 'probes')
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = f'{device}_{label}'
    identity = {'input': _array_record(x), 'target': _array_record(target_array),
                'device': device, 'precision': 'fp32', 'weight_sha256': WEIGHT_SHA,
                'requested_inference_batch': batch, 'packages': versions(),
                'diagnostic_source_sha256': sha(Path(__file__)),
                'indices': {k: _array_record(np.asarray(v)) for k, v in indices.items()},
                'baseline': float(x.min()), 'input_kind': label}
    checkpoint = out_dir / 'checkpoints' / f'{basename}.npz'
    completed = {}
    existing_original = None
    previous_batches = []
    if checkpoint.exists():
        saved = _load_npz(checkpoint)
        meta = json.loads(str(saved.pop('metadata')))
        if meta['identity'] != identity:
            raise RuntimeError('Probe checkpoint identity differs')
        if set(saved) != set(meta['hashes']) or any(
                _array_record(value) != meta['hashes'][key]
                or not np.isfinite(value).all() for key, value in saved.items()):
            raise RuntimeError('Probe checkpoint content shape/dtype/hash/finite check differs')
        if not np.array_equal(saved['input'], x) or not np.array_equal(saved['target'], target_array):
            raise RuntimeError('Probe checkpoint complete input/target differs')
        existing_original = saved['original_logits']
        completed = {k: v for k, v in saved.items() if k.endswith('_logits') and k != 'original_logits'}
        previous_batches = meta.get('batch_records', [])
    start = time.monotonic()
    _clock_check(clock)
    with torch.no_grad():
        if existing_original is None:
            logits = model(torch.from_numpy(x).to(device, dtype=torch.float32))
            if logits.dtype != torch.float32 or logits.device.type != device:
                raise RuntimeError('Probe original wrong dtype/device')
            original = np.ascontiguousarray(logits.detach().cpu().numpy(), dtype=np.float32)
        else:
            original = existing_original
        if original.shape != (1, 5) or original.dtype != np.float32 or not np.isfinite(original).all():
            raise RuntimeError('Invalid probe original five logits')
        singles = np.asarray(indices['single_indices'], dtype=np.int64)
        groups = np.asarray(indices['group_indices'], dtype=np.int64)
        tile_ids = np.asarray(indices['tile_ids'], dtype=np.int64)
        if singles.shape != (384,) or groups.shape != (128, 50) or tile_ids.shape != (128,):
            raise ValueError('Frozen indices shapes differ')
        jobs = [('single', singles[:32, None])] if device == 'cpu' else [('single', singles[:, None])]
        jobs += [('tile', np.stack([_tile_elements(i) for i in tile_ids]))] if device == 'cpu' else [('group', groups)]
        total = sum(len(job_indices) for _, job_indices in jobs)
        arrays = {'input': x, 'target': target_array, 'original_logits': original,
                  'single_indices': singles[:32] if device == 'cpu' else singles,
                  'tile_ids': tile_ids} if device == 'cpu' else {
                     'input': x, 'target': target_array, 'original_logits': original,
                     'single_indices': singles, 'group_indices': groups}
        arrays.update(completed)
        batch_records = list(previous_batches)
        errors = []
        actual_batch = batch

        def save_checkpoint(status):
            meta = {'identity': identity, 'status': status, 'batch_records': batch_records,
                    'errors': errors, 'planned_masks': total,
                    'completed_masks': sum(len(arrays.get(kind + '_logits', [])) for kind, _ in jobs),
                    'wall_seconds_this_invocation': time.monotonic() - start,
                    'hashes': {k: _array_record(v) for k, v in arrays.items()}}
            atomic_npz(checkpoint, **arrays, metadata=np.asarray(json.dumps(meta, sort_keys=True)))

        save_checkpoint('RUNNING')
        for kind, job_indices in jobs:
            key = kind + '_logits'
            outputs = arrays.get(key, np.empty((0, 5), dtype=np.float32))
            if outputs.ndim != 2 or outputs.shape[1] != 5 or len(outputs) > len(job_indices):
                raise RuntimeError('Invalid saved probe checkpoint shape')
            position = len(outputs)
            while position < len(job_indices):
                _clock_check(clock)
                count = min(actual_batch, len(job_indices) - position)
                masked = np.repeat(x, count, axis=0)
                flat = masked.reshape(count, -1)
                # Every row starts from the original input; each mask is independent.
                flat[np.arange(count)[:, None], job_indices[position:position + count]] = np.float32(x.min())
                try:
                    result = model(torch.from_numpy(masked).to(device, dtype=torch.float32))
                    if result.dtype != torch.float32 or result.device.type != device:
                        raise RuntimeError('Probe masked wrong dtype/device')
                    value = np.ascontiguousarray(result.detach().cpu().numpy(), dtype=np.float32)
                    if value.shape != (count, 5) or not np.isfinite(value).all():
                        raise RuntimeError('Probe masked logits invalid')
                except RuntimeError as error:
                    message = str(error)
                    errors.append({'kind': kind, 'start': position, 'batch': actual_batch,
                                   'exception': type(error).__name__, 'message': message})
                    save_checkpoint('RETRY_AFTER_OOM' if 'out of memory' in message.lower() else 'FAILED')
                    if 'out of memory' not in message.lower() or actual_batch == 16:
                        raise
                    actual_batch = {64: 32, 32: 16}[actual_batch]
                    if device == 'mps':
                        torch.mps.empty_cache()
                    continue
                outputs = np.concatenate([outputs, value], axis=0)
                arrays[key] = outputs
                batch_records.append({'kind': kind, 'start': position, 'count': count,
                                      'inference_batch': actual_batch, 'logits_sha256': arrsha(value)})
                position += count
                save_checkpoint('RUNNING')
                _clock_update(clock, phase='A_small_probes', method='finite_intervention_diagnostic',
                              device=device, input_kind=label, Id=1,
                              masks_completed=sum(len(arrays.get(j + '_logits', [])) for j, _ in jobs),
                              masks_planned=total, probe_kind=kind, actual_batch=actual_batch)
                _clock_check(clock)
    sync(device)
    metadata = {'identity': identity, 'status': 'COMPLETED', 'batch_records': batch_records,
                'errors': errors, 'requested_batch': batch, 'actual_batch': actual_batch,
                'planned_masks': total, 'completed_masks': total,
                'predicted_class': int(np.argmax(original[0])),
                'wall_seconds_this_invocation': time.monotonic() - start,
                'hashes': {k: _array_record(v) for k, v in arrays.items()}}
    atomic_npz(out_dir / f'{basename}.npz', **arrays,
               metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    atomic_json(out_dir / f'{basename}.json', metadata)
    save_checkpoint('COMPLETED')
    return arrays, metadata


def vector_statistics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Matching finite saved vectors required')
    denominator = max(float(np.linalg.norm(actual)), 1e-8)
    if np.ptp(actual) == 0 or np.ptp(predicted) == 0:
        pearson, reason = None, 'Zero variation in actual and/or predicted vector'
    else:
        pearson = float(np.corrcoef(actual, predicted)[0, 1])
        reason = None
        if not np.isfinite(pearson):
            pearson, reason = None, 'Numerical correlation undefined despite nonconstant vectors'
    active = np.maximum(np.abs(actual), np.abs(predicted)) > 1e-6
    matching = np.sign(actual[active]) == np.sign(predicted[active])
    count = int(active.sum())
    return {'count': len(actual), 'pearson': pearson, 'pearson_null_reason': reason,
            'relative_l2': float(np.linalg.norm(predicted - actual) / denominator),
            'relative_l2_denominator': denominator, 'mae': float(np.mean(np.abs(predicted - actual))),
            'sign_agreement': float(np.mean(matching)) if count else None,
            'sign_null_reason': None if count else 'All pairs excluded by fixed near-zero criterion',
            'sign_threshold': 1e-6, 'sign_included_count': count,
            'sign_excluded_count': int(len(actual) - count),
            'sign_mismatch_count': int(count - matching.sum()),
            'actual_near_zero_count': int((np.abs(actual) <= 1e-6).sum()),
            'predicted_near_zero_count': int((np.abs(predicted) <= 1e-6).sum()),
            'actual_range': [float(actual.min()), float(actual.max())],
            'predicted_range': [float(predicted.min()), float(predicted.max())]}


def _logit_comparison(cpu, mps):
    cpu, mps = np.asarray(cpu), np.asarray(mps)
    if cpu.shape != mps.shape:
        raise ValueError('CPU/MPS probe shape mismatch')
    finite = bool(np.isfinite(cpu).all() and np.isfinite(mps).all())
    close = np.isclose(cpu, mps, rtol=1e-3, atol=1e-4)
    same_classes = bool(np.array_equal(np.argmax(cpu, axis=-1), np.argmax(mps, axis=-1)))
    return {'shape': list(cpu.shape), 'finite': finite,
            'all_five_logits_allclose': bool(finite and close.all()),
            'mismatching_logit_count': int((~close).sum()),
            'max_abs_logit_difference': float(np.max(np.abs(cpu.astype(np.float64) - mps.astype(np.float64)))),
            'predicted_class_same': same_classes, 'rtol': 1e-3, 'atol': 1e-4}


def device_gate(cpu, mps, raw4_arrays, indices=None, out_path=None):
    """Check all five saved logits on the registered CPU/MPS subset."""
    t = int(np.asarray(cpu['target']).reshape(-1)[0])
    if (not np.array_equal(cpu['input'], mps['input'])
            or not np.array_equal(cpu['input'], raw4_arrays['input'])
            or not np.array_equal(cpu['target'], mps['target'])
            or not np.array_equal(cpu['target'], raw4_arrays['target'])):
        raise RuntimeError('Device gate input/target identities differ')
    if not np.array_equal(cpu['single_indices'], mps['single_indices'][:32]):
        raise RuntimeError('Device gate frozen singleton order differs')
    if indices is not None and not np.array_equal(cpu['single_indices'], indices['single_indices'][:32]):
        raise RuntimeError('Device gate singleton indices differ from preregistered sample')
    tile_ids = np.asarray(indices['tile_ids'] if indices is not None else cpu['tile_ids'], dtype=np.int64)
    if not np.array_equal(tile_ids, cpu['tile_ids']):
        raise RuntimeError('Device gate frozen tile order differs')
    raw_logits = np.asarray(raw4_arrays.get('logits', raw4_arrays.get('original_logits')))
    raw_masked = np.asarray(raw4_arrays['masked_logits'])
    if raw_logits.shape != (1, 5) or raw_masked.shape != (9408, 5):
        raise RuntimeError('Full raw must preserve every five-class masked logit')
    comparisons = {
        'original_cpu_vs_mps_small': _logit_comparison(cpu['original_logits'], mps['original_logits']),
        'original_cpu_vs_mps_fullraw': _logit_comparison(cpu['original_logits'], raw_logits),
        'single_first32': _logit_comparison(cpu['single_logits'], mps['single_logits'][:32]),
        'tile_128': _logit_comparison(cpu['tile_logits'], raw_masked[tile_ids]),
    }
    cpu_drop = cpu['original_logits'][0, t].astype(np.float64) - cpu['tile_logits'][:, t].astype(np.float64)
    mps_drop = raw_logits[0, t].astype(np.float64) - raw_masked[tile_ids, t].astype(np.float64)
    signed = vector_statistics(cpu_drop, mps_drop)
    passed = all(v['finite'] and v['all_five_logits_allclose'] and v['predicted_class_same']
                 for v in comparisons.values()) and signed['relative_l2'] <= .01
    result = {'status': 'PASS' if passed else 'FAIL', 'acceptance': ACCEPTANCE,
              'target': t, 'input_sha256': arrsha(cpu['input']), 'comparisons': comparisons,
              'tile_signed_drop': signed, 'tile_ids_sha256': arrsha(tile_ids),
              'full_cpu_4x4_support_validation': 'NOT_PERFORMED_IN_THIS_STAGE'}
    if out_path:
        atomic_json(out_path, result)
        atomic_npz(Path(out_path).with_suffix('.npz'), tile_ids=tile_ids,
                   cpu_tile_signed_drop=cpu_drop, mps_tile_signed_drop=mps_drop,
                   cpu_original_logits=cpu['original_logits'], mps_original_logits=raw_logits,
                   cpu_single_logits=cpu['single_logits'], mps_single_logits=mps['single_logits'][:32],
                   cpu_tile_logits=cpu['tile_logits'], mps_tile_logits=raw_masked[tile_ids])
    return result


def historical_sources(label, out_dir=None):
    """Validate and bind saved Stage23 8x8 and Stage24 GDD evidence."""
    from postprocess import process
    out_dir = Path(out_dir or ROOT / 'probes')
    inputs, indices, frozen = read_sampling(out_dir)
    x, t = inputs[label], int(inputs['target'][0])
    gdd_path = WORKSPACE / f'output_mac_stage24/probes/mps_{label}.npz'
    gdd = _load_npz(gdd_path)
    gm = json.loads(str(gdd.pop('metadata')))
    if (gm['device'] != 'mps' or gm['raw_spec']['weight_sha256'] != WEIGHT_SHA
            or int(gm['record']['target']) != t or not np.array_equal(gdd['input'], x)):
        raise RuntimeError('Historical GDD source identity differs')
    if any(arrsha(v) != gm['hashes'][k] for k, v in gdd.items()):
        raise RuntimeError('Historical GDD content hash differs')
    gdd_rebuilt = (gdd['input'] - np.float32(x.min())) * gdd['gradient']
    if not np.array_equal(gdd_rebuilt, gdd['raw']):
        raise RuntimeError('Historical GDD raw cannot reconstruct')
    candidates = []
    archived_programs = {
        sha(WORKSPACE / 'output_mac_stage23/device/occlusion8_gpu.py'):
            WORKSPACE / 'output_mac_stage23/device/occlusion8_gpu.py',
        sha(WORKSPACE / 'output_mac_stage23/device/occlusion8_probe_frozen.py'):
            WORKSPACE / 'output_mac_stage23/device/occlusion8_probe_frozen.py',
    }
    cache_dir = WORKSPACE / 'output_mac_stage23/device/cache/black_occlusion8_signed/mps'
    for path in sorted(cache_dir.rglob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            if meta.get('input_sha256') == arrsha(x) and int(meta.get('target', -1)) == t:
                # Use a current completed source, preferring the batch64 official path.
                if meta['spec']['program_sha256'] in archived_programs:
                    candidates.append((int(meta['spec']['batch_size'] != 64), str(path), meta))
    if not candidates:
        raise RuntimeError(f'No validated saved same-input MPS 8x8 raw: {label}')
    _, path_string, om = min(candidates)
    old_path = Path(path_string)
    old = _load_npz(old_path)
    old.pop('metadata')
    if (om['spec']['device'] != 'mps' or om['spec']['weight_sha256'] != WEIGHT_SHA
            or om['spec']['tile_size'] != 8 or om['record']['baseline'] != float(x.min())
            or any(arrsha(old[k]) != om[k + '_sha256'] for k in ('raw', 'final', 'masked_logits'))):
        raise RuntimeError('Historical 8x8 source specification/content differs')
    drops = (np.float64(om['record']['original_target_logit']) - old['masked_logits'].astype(np.float64)) / 64.
    rebuilt = np.repeat(np.repeat(drops.astype(np.float32).reshape(3, 28, 28), 8, axis=1), 8, axis=2)[None]
    if not np.array_equal(rebuilt, old['raw']):
        raise RuntimeError('Historical 8x8 signed raw cannot reconstruct')
    raw = {'gdd_black': gdd['raw'], 'black_occlusion8_signed': old['raw']}
    finals = {}
    for method, value in raw.items():
        for suffix in ('post22', 'soft15'):
            processed = process(value, 'black_occlusion4_' + suffix)
            finals[method + '_' + suffix] = processed['final']
            if method == 'gdd_black' and not np.array_equal(processed['final'], gdd['gdd_black_' + suffix + '_final']):
                raise RuntimeError('Historical GDD final mismatch')
            if method == 'black_occlusion8_signed' and suffix == 'post22' and not np.array_equal(processed['final'], old['final']):
                raise RuntimeError('Historical 8x8 fixed final mismatch')
    proof = {'input': _array_record(x), 'target': _array_record(inputs['target']),
             'sources': {'gdd_black': {'path': str(gdd_path), 'file_sha256': sha(gdd_path), 'metadata': gm},
                         'black_occlusion8_signed': {'path': str(old_path), 'file_sha256': sha(old_path),
                             'program_path': str(archived_programs[om['spec']['program_sha256']]),
                             'program_sha256': om['spec']['program_sha256'], 'metadata': om}},
             'raw_hashes': {k: arrsha(v) for k, v in raw.items()},
             'final_hashes': {k: arrsha(v) for k, v in finals.items()},
             'same_input_target_device_weight_validation': 'PASS'}
    arrays = {'input': x, 'target': inputs['target'], 'gdd_gradient': gdd['gradient'],
              'gdd_logits': gdd['logits'], 'occlusion8_masked_logits': old['masked_logits'],
              'occlusion8_original_logits': np.asarray([om['record']['original_logits']], dtype=np.float32),
              **raw, **finals}
    atomic_npz(out_dir / f'historical_sources_{label}.npz', **arrays,
               metadata=np.asarray(json.dumps(proof, sort_keys=True)))
    atomic_json(out_dir / f'historical_sources_{label}.json', proof)
    return raw, finals, proof


def build_diagnostics(label, raw4_arrays, out_dir=None, raw4_metadata=None, mps_arrays=None):
    """Save all actual and predicted finite-intervention vectors, without tuning."""
    from postprocess import process
    out_dir = Path(out_dir or ROOT / 'probes')
    inputs, indices, frozen = read_sampling(out_dir)
    x, t = inputs[label], int(inputs['target'][0])
    mps = mps_arrays if mps_arrays is not None else _load_npz(out_dir / f'mps_{label}.npz')
    if not np.array_equal(x, mps['input']) or not np.array_equal(x, raw4_arrays['input']):
        raise RuntimeError('Diagnostic source input identity differs')
    if (not np.array_equal(inputs['target'], mps['target'])
            or not np.array_equal(inputs['target'], raw4_arrays['target'])
            or not np.array_equal(indices['single_indices'], mps['single_indices'])
            or not np.array_equal(indices['group_indices'], mps['group_indices'])):
        raise RuntimeError('Diagnostic target or frozen mask identity differs')
    for key, shape in (('original_logits', (1, 5)), ('single_logits', (384, 5)),
                       ('group_logits', (128, 5))):
        value = np.asarray(mps[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f'Diagnostic actual-logit shape/dtype/finite differs: {key}')
    raw, finals, sources = historical_sources(label, out_dir)
    raw['black_occlusion4_signed_v1'] = raw4_arrays['raw']
    for suffix in ('post22', 'soft15'):
        finals['black_occlusion4_' + suffix] = process(raw4_arrays['raw'], 'black_occlusion4_' + suffix)['final']
    sources['sources']['black_occlusion4_signed_v1'] = {
        'metadata': raw4_metadata, 'raw_sha256': arrsha(raw4_arrays['raw']),
        'logits_sha256': arrsha(raw4_arrays['logits']),
        'masked_logits_sha256': arrsha(raw4_arrays['masked_logits'])}
    single_indices, group_indices = indices['single_indices'], indices['group_indices']
    original = np.float64(mps['original_logits'][0, t])
    actual_single = original - mps['single_logits'][:, t].astype(np.float64)
    actual_group = original - mps['group_logits'][:, t].astype(np.float64)
    vectors = {'single_indices': single_indices, 'group_indices': group_indices,
               'true_single_drop': actual_single, 'true_group50_drop': actual_group}
    rows = {'single_raw': {}, 'group50_raw': {}, 'group50_fixed_finals': {}}
    for method, value in raw.items():
        flat = value.reshape(-1).astype(np.float64)
        singles = flat[single_indices]
        groups = np.sum(flat[group_indices], axis=1, dtype=np.float64)
        vectors['single_' + method] = singles
        vectors['group50_' + method] = groups
        rows['single_raw'][method] = vector_statistics(actual_single, singles)
        rows['group50_raw'][method] = vector_statistics(actual_group, groups)
    for method, value in finals.items():
        groups = np.sum(value.reshape(-1).astype(np.float64)[group_indices], axis=1, dtype=np.float64)
        vectors['group50_final_' + method] = groups
        rows['group50_fixed_finals'][method] = vector_statistics(actual_group, groups)
    result = {'schema': 1, 'status': 'COMPLETED', 'input_kind': label,
              'input_sha256': arrsha(x), 'target': t, 'device': 'mps',
              'baseline': float(x.min()), 'frozen_sampling_sha256': sha(out_dir / 'frozen_sampling.json'),
              'definition': 'actual provided-target original logit minus independent black-replacement logit',
              'diagnostic_is_official_faithfulness_replay': False,
              'parameter_changes_based_on_diagnostic': False,
              'interpretation': '4x4 and 8x8 raw average joint tile effects; not exact single-element effects',
              'sources': sources, 'tables': rows,
              'vectors': {k: _array_record(v) for k, v in vectors.items()}}
    atomic_npz(out_dir / f'diagnostic_vectors_{label}.npz', **vectors,
               metadata=np.asarray(json.dumps(result, sort_keys=True)))
    atomic_json(out_dir / f'diagnostic_{label}.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--verify-historical', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        _, _, record = freeze_sampling()
        print(json.dumps({'sampling': record['arrays'], 'status': 'FROZEN'}, indent=2))
    if args.verify_historical:
        for label in ('original', 'perturbed'):
            _, _, record = historical_sources(label)
            print(label, record['same_input_target_device_weight_validation'])
