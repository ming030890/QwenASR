#!/usr/bin/env python3
"""Production-policy Android experiments; raw output is retained before scoring.

prepare uploads isolated assets/models. run executes one named configuration;
compare performs a strictly paired, per-corpus CER gate without rounded floats.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACK = Path('/Volumes/SSD/qwen3_asr_17b_staged_qat_v18_pack')
VARIANTS = {
    'b0': {}, 'e1-4': {'QWEN_EXPERIMENT_LM_ROWS': '4'},
    'e1-8': {'QWEN_EXPERIMENT_LM_ROWS': '8'},
    'e2': {'QWEN_EXPERIMENT_CONV_REUSE': '1'},
    'packed': {}, 'e3': {'QWEN_EXPERIMENT_QKV_REUSE': '1'},
}
POLICY = dict(language='Cantonese', prompt='', chinese_variant='', segment_sec=0, threads=1)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tokens(text):
    # Identical to staged A2S-SFT metrics.mixed_cer_tokens, after comparison_text.
    return [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'._-]*|.", text.casefold(), re.DOTALL)
            if not t.isspace()]


def distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def score(rows, predictions):
    scored = []
    for row in rows:
        result = dict(row)
        prediction = predictions.get(row['file'])
        if prediction is None:
            result.update(error='missing prediction / inference failed', prediction=None)
        else:
            ref = tokens(row['reference'])
            result.update(prediction=prediction, edits=distance(ref, tokens(prediction)), ref_units=len(ref))
        scored.append(result)
    failures = sum('error' in r for r in scored)
    edits = sum(r.get('edits', 0) for r in scored)
    units = sum(r.get('ref_units', 0) for r in scored)
    summary = dict(rows=len(rows), failures=failures, edits=edits, ref_units=units,
                   corpus_cer=edits / units if units and not failures else None,
                   valid=not failures and units > 0)
    return scored, summary


def cer_gate(baseline, candidate):
    if baseline['policy'] != candidate['policy'] or baseline['inputs'] != candidate['inputs']:
        raise ValueError('Different policy, manifests, or audio bytes')
    verdict = {}
    for corpus in ('pure_cantonese', 'mixed_script'):
        b, c = baseline['corpora'][corpus], candidate['corpora'][corpus]
        verdict[corpus] = bool(b['valid'] and c['valid'] and b['rows'] == c['rows']
                              and b['ref_units'] == c['ref_units'] and c['edits'] <= b['edits'])
    return dict(passed=all(verdict.values()), corpora=verdict)


def adb(args, *parts, **kwargs):
    return subprocess.run([args.adb, *map(str, parts)], check=True, **kwargs)


def thermal_ready(args):
    deadline = time.monotonic() + 900
    while True:
        result = adb(args, 'shell', 'dumpsys thermalservice', capture_output=True, text=True).stdout
        match = re.search(r'Thermal Status:\s*(\d+)', result)
        if match and match[1] == '0':
            return result
        if time.monotonic() > deadline:
            raise RuntimeError('Thermal status did not reach zero; refusing contaminated measurement')
        print('Waiting for thermal status zero', flush=True)
        time.sleep(15)


def prepare(args):
    local = args.output / 'assets'
    local.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for corpus, count in [('pure_cantonese', 363), ('mixed_script', 126)]:
        manifest = args.pack / 'staging/eval' / f'{corpus}.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(rows) != count or len({r['id'] for r in rows}) != count:
            raise ValueError(f'{corpus}: unexpected count or duplicate IDs')
        directory = local / corpus
        directory.mkdir(exist_ok=True)
        mapped = []
        for index, row in enumerate(rows):
            source = args.pack / 'staging/audio' / Path(row['audio']).name
            if not source.is_file():
                raise FileNotFoundError(source)
            name = f'{index:04d}.wav'
            shutil.copyfile(source, directory / name)
            reference = re.sub(r'^\s*language\s+Cantonese\s*<asr_text>\s*', '', row['text']).strip()
            mapped.append(dict(id=row['id'], file=name, reference=reference, audio_sha256=sha256(source)))
        inventory[corpus] = dict(manifest_sha256=sha256(manifest), rows=mapped)
    perf = local / 'performance'
    perf.mkdir(exist_ok=True)
    for name in [f'cv_yue_{i:03d}.wav' for i in range(5)] + ['ptt_guardrail.wav']:
        shutil.copyfile(ROOT / 'bench/android-assets' / name, perf / name)
    (args.output / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2))
    adb(args, 'shell', f'mkdir -p {shlex.quote(args.remote)}')
    adb(args, 'push', local, args.remote + '/')
    adb(args, 'push', args.binary, args.remote + '/bench')
    adb(args, 'shell', f'chmod 755 {shlex.quote(args.remote + "/bench")}')
    adb(args, 'push', str(args.model) + '/.', args.remote + '/b0')
    provenance = dict(policy=POLICY, model=str(args.model), model_sha256=sha256(args.model / 'model.safetensors'),
                      binary_sha256=sha256(args.binary), binary_size_bytes=args.binary.stat().st_size,
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))


def run_device(args, directory, destination, profile=False):
    thermal = thermal_ready(args)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'thermal.txt').write_text(thermal)
    model = 'packed' if args.variant in ('packed', 'e3') else 'b0'
    env = {'QWEN_BENCH_STREAM': '1', 'QWEN_EXPERIMENT_LM_ROWS': '2',
           'QWEN_EXPERIMENT_CONV_REUSE': '0', 'QWEN_EXPERIMENT_QKV_REUSE': '0', **VARIANTS[args.variant]}
    parts = [args.remote + '/bench', '--model-dir', args.remote + '/' + model,
             '--input', args.remote + '/assets/' + directory, '--runs', '1',
             '--threads', '1', '--language', 'Cantonese', '--prompt', '']
    if profile:
        parts.append('--profile')
    command = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items()) + ' ' + shlex.join(parts)
    (destination / 'command.txt').write_text(command)
    predictions, payload = {}, None
    with (destination / 'stderr.log').open('w') as err, (destination / 'stdout.log').open('w') as raw:
        proc = subprocess.Popen([args.adb, 'shell', command], stdout=subprocess.PIPE, stderr=err, text=True)
        try:
            for line in proc.stdout:
                raw.write(line)
                raw.flush()
                if line.startswith('QWEN_FILE '):
                    record = json.loads(line[len('QWEN_FILE '):])
                    name = Path(record['audio_path']).name
                    if name in predictions:
                        raise ValueError('duplicate result: ' + name)
                    predictions[name] = record['transcript']
                    print(f'{args.variant}/{directory}: {len(predictions)} files, {record["inference_ms"]:.0f} ms', flush=True)
                elif line.startswith('{'):
                    payload = json.loads(line)
            returncode = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    if payload is not None:
        (destination / 'result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    valid = returncode == 0 and payload is not None and 'OpenBLAS' in payload.get('optimization_flags', [])
    (destination / 'status.json').write_text(json.dumps(dict(returncode=returncode, valid=valid)))
    return predictions, payload, valid


def run(args):
    dest = args.output / args.label
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    state = adb(args, 'shell', 'settings get global stay_on_while_plugged_in', capture_output=True, text=True).stdout.strip()
    try:
        adb(args, 'shell', 'svc power stayon true')
        adb(args, 'shell', 'cmd power set-fixed-performance-mode-enabled true')
        if args.suite == 'eval':
            inventory = json.loads((args.output / 'inventory.json').read_text())
            report = dict(variant=args.variant, policy=POLICY, inputs=inventory, corpora={})
            for corpus, entry in inventory.items():
                predictions, _, valid = run_device(args, corpus, dest / corpus)
                rows, summary = score(entry['rows'], predictions)
                summary['valid'] &= valid
                report['corpora'][corpus] = summary
                (dest / f'{corpus}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
                (dest / 'cer.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(report['corpora'], ensure_ascii=False), flush=True)
        else:
            _, _, valid = run_device(args, 'performance', dest / 'performance', args.suite == 'profile')
            if not valid:
                raise RuntimeError('Invalid device run; inspect retained logs')
    finally:
        adb(args, 'shell', 'cmd power set-fixed-performance-mode-enabled false')
        if state.isdigit():
            adb(args, 'shell', 'settings put global stay_on_while_plugged_in ' + state)
        else:
            adb(args, 'shell', 'settings delete global stay_on_while_plugged_in')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'compare'])
    p.add_argument('--pack', type=Path, default=DEFAULT_PACK)
    p.add_argument('--model', type=Path, default=DEFAULT_PACK / 'int8_exports/stage2a_qat_projector_checkpoint-500')
    p.add_argument('--binary', type=Path, default=ROOT / 'target/aarch64-linux-android/release/qwen-asr-bench-android')
    p.add_argument('--output', type=Path, default=ROOT / 'bench/android-results/v18-loop')
    p.add_argument('--remote', default='/data/local/tmp/qwen-v18-loop')
    p.add_argument('--adb', default=shutil.which('adb'))
    p.add_argument('--variant', choices=VARIANTS, default='b0')
    p.add_argument('--suite', choices=['eval', 'performance', 'profile'], default='performance')
    p.add_argument('--label', default='b0')
    p.add_argument('--baseline', type=Path)
    p.add_argument('--candidate', type=Path)
    args = p.parse_args()
    if args.action == 'compare':
        verdict = cer_gate(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
        print(json.dumps(verdict, indent=2))
        raise SystemExit(0 if verdict['passed'] else 1)
    if args.action == 'prepare':
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
