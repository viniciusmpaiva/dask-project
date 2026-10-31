import copy
import json
import os
import random
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import psutil

from .common import (cgroup_state, file_hash, identity, now, read_json, roots,
                     verify_limits, versions, write_json)
from .data import generate


def scenario_key(s):
    key = f"{s['volume']}-{s['approach']}-w{s['workers']}-p{s['partitions']}"
    workload = s.get('workload', 'merge')
    return key if workload == 'merge' else f'{key}-{workload}'


def override_dask_scenarios(config, workers=None, partitions=None):
    updated = copy.deepcopy(config)
    scenarios = []
    seen = set()
    for scenario in updated['scenarios']:
        if scenario['approach'] == 'dask':
            if workers is not None:
                scenario['workers'] = workers
            if partitions is not None:
                scenario['partitions'] = partitions
        key = scenario_key(scenario)
        if key not in seen:
            seen.add(key)
            scenarios.append(scenario)
    updated['scenarios'] = scenarios
    return updated


def source_id():
    files = sorted(Path('pilot').glob('*.py')) + sorted(Path('configs').glob('*.json'))
    files += [Path(p) for p in ('Dockerfile', 'compose.yaml', 'requirements.lock.txt') if Path(p).exists()]
    return identity({str(p): file_hash(p) for p in files})


def stop_tree(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for p in reversed(children):
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        psutil.wait_procs(children + [parent], timeout=10)
    except psutil.NoSuchProcess:
        pass
    proc.wait(timeout=15)


def run_config(config, experiment_id, allow_host=False):
    if not experiment_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in experiment_id):
        raise ValueError('Use letters, digits, hyphens or underscores for experiment ID')
    if allow_host and config['mode'] != 'host-check':
        raise ValueError('Host checks cannot be labelled pilot or smoke')
    if not allow_host and config['mode'] == 'host-check':
        raise ValueError('Host check needs explicit --allow-host-check')
    if config['blocks'] < 1 or config['max_attempts'] < 1:
        raise ValueError('blocks and max_attempts must be positive')
    for s in config['scenarios']:
        if s['approach'] not in ('pandas', 'dask') or s['volume'] not in config['volumes']:
            raise ValueError('Invalid scenario')
        if s.get('workload', 'merge') not in ('merge', 'merge_groupby'):
            raise ValueError('workload must be one of merge, merge_groupby')
        if s['approach'] == 'dask' and (
                s['workers'] not in (1, 2, 4, 8, 16) or
                s['partitions'] not in (1, 2, 4, 8, 16, 32, 64)):
            raise ValueError('Dask worker/partition values are outside the supported range')
        if s['approach'] == 'pandas' and (s['workers'] or s['partitions']):
            raise ValueError('Pandas must have workers=partitions=0')
    state = cgroup_state()
    limit_check = verify_limits(state, allow_host)
    data_root, result_root, scratch = roots()
    environment = os.environ.get('EXPERIMENT_ENV', 'PC local')
    if allow_host:
        environment += ' / verificação nativa Windows sem Docker (não é piloto)'
    resource_identity = {k: state[k] for k in ('cpu_limit', 'memory_limit_bytes', 'platform', 'storage')}
    contract = {'config': config, 'environment': environment, 'versions': versions(),
                'resources': resource_identity, 'source_id': source_id(),
                'worker_memory_mib': int(os.environ.get('WORKER_MEMORY_MIB', '768')),
                'source_commit': os.environ.get('SOURCE_COMMIT', 'unavailable')}
    fingerprint = identity(contract)
    folder = result_root / config['mode'] / experiment_id
    folder.mkdir(parents=True, exist_ok=True)
    lock = result_root / '.measurement.lock'
    result_root.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Another measurement or stale lock: {lock}; inspect before removing') from None
    os.write(lock_fd, json.dumps({'pid': os.getpid(), 'experiment': str(folder), 'at': now()}).encode())
    os.close(lock_fd)
    try:
        meta_path = folder / 'experiment.json'
        if meta_path.exists():
            metadata = read_json(meta_path)
            if metadata['fingerprint'] != fingerprint:
                raise ValueError('Resume refused: code/configuration/environment/versions/resources changed; use new ID')
        else:
            metadata = {'experiment_id': experiment_id, 'created_at': now(), 'fingerprint': fingerprint,
                        **contract, 'resource_check': limit_check, 'initial_resources': state,
                        'measurement_scope': 'read both tables through full client materialization',
                        'warmup': 'one full discarded workload per fresh process; OS cache not flushed',
                        'auxiliary_services': 'runner and instrumentation; results service must be stopped',
                        'host_interference': 'not eliminated; no other workload started by experiment'}
            write_json(meta_path, metadata)
        # Every input is generated and verified before measured attempts.
        datasets = {volume: generate(data_root, n, config['seeds'], config['physical_files'])
                    for volume, n in config['volumes'].items()}
        write_json(folder / 'inputs.json', datasets)
        schedule_path = folder / 'schedule.json'
        if schedule_path.exists():
            schedule = read_json(schedule_path)
        else:
            schedule = []
            rng = random.Random(config['order_seed'])
            for block in range(1, config['blocks'] + 1):
                scenarios = copy.deepcopy(config['scenarios'])
                rng.shuffle(scenarios)
                schedule.extend({'block': block, 'scenario': s} for s in scenarios)
            write_json(schedule_path, schedule)
        failures = []
        for slot in schedule:
            block, s = slot['block'], slot['scenario']
            key = scenario_key(s)
            slot_folder = folder / 'raw' / f'block-{block:02d}' / key
            slot_folder.mkdir(parents=True, exist_ok=True)
            existing = sorted(slot_folder.glob('*/record.json'))
            if any(read_json(p)['status'] == 'success' for p in existing):
                print(f'SKIP block={block} {key}: already validated', flush=True)
                continue
            # An interrupted process leaves spec and logs. Preserve as failed attempt.
            for p in slot_folder.glob('*/spec.json'):
                if not (p.parent / 'record.json').exists():
                    write_json(p.parent / 'record.json', {**read_json(p), 'status': 'failed',
                               'runtime_s': None, 'validation': None, 'ended_at': now(),
                               'failure_reason': 'interrupted attempt recovered on resume'})
            used = len(list(slot_folder.glob('*/record.json')))
            succeeded = False
            for attempt in range(used + 1, config['max_attempts'] + 1):
                run_id = f'{attempt:02d}-{uuid.uuid4().hex[:12]}'
                run_folder = slot_folder / run_id
                run_folder.mkdir()
                work = scratch / experiment_id / run_id
                work.mkdir(parents=True, exist_ok=True)
                spec = {**s, 'n': config['volumes'][s['volume']], 'block': block,
                        'attempt': attempt, 'replacement': attempt > 1,
                        'run_id': run_id, 'experiment_id': experiment_id,
                        'fingerprint': fingerprint, 'environment': environment, 'mode': config['mode'],
                        'dataset': datasets[s['volume']], 'scratch': str(work),
                        'resources': state, 'cpu_budget': state['cpu_limit'] or psutil.cpu_count(),
                        'measurement_root_pid': os.getpid(),
                        'sample_interval_s': config['sample_interval_s'],
                        'source_commit': contract['source_commit'], 'started_at': now()}
                write_json(run_folder / 'spec.json', spec)
                print(f'RUN block={block} {key} attempt={attempt}', flush=True)
                timed_out = False
                with (run_folder / 'process.log').open('w', encoding='utf-8') as log:
                    proc = subprocess.Popen([sys.executable, '-m', 'pilot', '_trial',
                                             str(run_folder / 'spec.json')],
                                            stdout=log, stderr=subprocess.STDOUT)
                    try:
                        proc.wait(timeout=config['timeout_s'])
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        stop_tree(proc)
                    except BaseException:
                        stop_tree(proc)
                        raise
                record_path = run_folder / 'record.json'
                if not record_path.exists():
                    write_json(record_path, {**spec, 'status': 'failed', 'runtime_s': None,
                               'validation': None, 'ended_at': now(),
                               'failure_reason': 'timeout' if timed_out else f'process exit {proc.returncode}'})
                rec = read_json(record_path)
                if rec['status'] == 'success' and proc.returncode != 0:
                    rec['status'] = 'failed'
                    rec['failure_reason'] = f'Process exited {proc.returncode} after validation'
                    write_json(record_path, rec)
                print(f"END {key}: {rec['status']} runtime={rec.get('runtime_s')} reason={rec.get('failure_reason')}", flush=True)
                if rec['status'] == 'success' and proc.returncode == 0:
                    succeeded = True
                    # Delete only this attempt's scratch, never shared inputs/results.
                    shutil.rmtree(work)
                    break
            if not succeeded:
                failures.append({'block': block, 'scenario': key, 'attempts_exhausted': True})
                # Repeated failure stops this configuration in later blocks.
                for later in schedule:
                    if later['block'] > block and scenario_key(later['scenario']) == key:
                        later['disabled'] = True
                write_json(folder / 'halted-configurations.json', failures)
                # Abort the session: investigate instead of silently shrinking matrix.
                raise RuntimeError(f'{key}: attempts exhausted; inspect {slot_folder}')
        write_json(folder / 'completed.json', {'at': now(), 'slots': len(schedule),
                                             'fingerprint': fingerprint})
        return folder
    finally:
        lock.unlink(missing_ok=True)


def pilot_with_gates(config, experiment_id, smoke_config=None):
    smoke = smoke_config or read_json('configs/smoke.json')
    run_config(smoke, experiment_id + '-smoke')
    preflight = copy.deepcopy(config)
    preflight.update(mode='preflight', blocks=1)
    largest = max(config['volumes'], key=config['volumes'].get)
    preflight['scenarios'] = [s for s in config['scenarios'] if s['volume'] == largest]
    preflight['scenarios'] += [s for s in config['scenarios'] if s['partitions'] == 64 and s['volume'] != largest]
    path = run_config(preflight, experiment_id + '-preflight')
    for record in path.glob('raw/*/*/*/record.json'):
        rec = read_json(record)
        if rec['status'] != 'success':
            continue
        metrics = rec.get('metrics', {})
        peak = metrics.get('container_memory_sampled_peak_bytes')
        budget = rec['resources']['memory_limit_bytes']
        if peak and budget and peak > .90 * budget:
            raise MemoryError('Preflight exceeded 90% of memory budget; revise all volumes before pilot')
    return run_config(config, experiment_id)
