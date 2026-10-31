import html
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import now, read_json, write_json
from .runner import scenario_key


def analyze(folder):
    folder = Path(folder)
    meta = read_json(folder / 'experiment.json')
    config = meta['config']
    records = [read_json(p) for p in sorted(folder.glob('raw/*/*/*/record.json'))]
    valid = [r for r in records if r['status'] == 'success' and r.get('validation', {}).get('ok')]
    # Reject cross-experiment/config contamination, including edited raw files.
    if any(r.get('fingerprint') != meta['fingerprint'] for r in records):
        raise ValueError('Raw fingerprint mismatch')
    slots = [(r['block'], scenario_key(r)) for r in valid]
    if len(set(slots)) != len(slots):
        raise ValueError('Duplicate valid slot; no silent double-counting')
    processed, output = folder / 'processed', folder / 'results'
    processed.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    rows = []
    for s in config['scenarios']:
        key = scenario_key(s)
        attempts = [r for r in records if scenario_key(r) == key]
        success = [r for r in valid if scenario_key(r) == key]
        times = [r['runtime_s'] for r in success]
        row = {**s, 'workload': s.get('workload', 'merge'), 'n': config['volumes'][s['volume']], 'configuration': key,
               'successes': len(success), 'failures': len(attempts) - len(success),
               'planned': config['blocks'], 'cp3_complete': len(success) >= 5 and config['mode'] in ('pilot', 'scenario'),
               'collection_complete': len(success) == config['blocks'],
               'mean_runtime_s': float(np.mean(times)) if times else None,
               'sample_std_runtime_s': float(np.std(times, ddof=1)) if len(times) > 1 else None,
               'speedup': None, 'relative_efficiency': None, 'paired_blocks': 0}
        if s['approach'] == 'dask':
            pairs = []
            for r in success:
                baseline = next((b for b in valid if b['volume'] == s['volume'] and
                                 b['block'] == r['block'] and b['approach'] == 'pandas' and
                                 b.get('workload', 'merge') == s.get('workload', 'merge')), None)
                if baseline:
                    pairs.append((baseline['runtime_s'], r['runtime_s']))
            if pairs:
                # Ratio of means on common volume/block pairs, not mean of ratios.
                row['speedup'] = float(np.mean([p[0] for p in pairs]) / np.mean([p[1] for p in pairs]))
                row['relative_efficiency'] = row['speedup'] / s['workers']
                row['paired_blocks'] = len(pairs)
        for metric in ('process_cpu_seconds', 'cpu_budget_percent', 'cgroup_cpu_seconds',
                       'cgroup_cpu_budget_percent', 'rss_sum_peak_bytes', 'container_memory_sampled_peak_bytes'):
            values = [r['metrics'][metric] for r in success
                      if r.get('metrics', {}).get(metric) is not None]
            row['mean_' + metric] = float(np.mean(values)) if values else None
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(processed / 'summary.csv', index=False)
    pd.json_normalize(records).to_csv(processed / 'attempts.csv', index=False)
    write_json(processed / 'summary.json', rows)
    plot_rows(summary, output, meta['environment'])
    successes, failures = len(valid), len(records) - len(valid)
    report = make_report(meta, rows, successes, failures)
    (output / 'report.md').write_text(report, encoding='utf-8')
    images = ''.join(f'<figure><img src="{p.name}" alt="{html.escape(p.stem)}"></figure>'
                     for p in sorted(output.glob('*.png')))
    display = summary[['configuration', 'n', 'successes', 'failures', 'mean_runtime_s',
                       'sample_std_runtime_s', 'speedup', 'relative_efficiency']].copy()
    for col in ('mean_runtime_s', 'sample_std_runtime_s', 'speedup', 'relative_efficiency'):
        display[col] = display[col].map(lambda value: 'indisponível' if pd.isna(value) else f'{value:.5g}')
    display['configuration'] = [
        (f"Pandas · {r['volume']}" if r['approach'] == 'pandas' else
         f"Dask · {r['volume']} · {r['workers']} worker(s) · {r['partitions']} partições") +
        ('' if r.get('workload', 'merge') == 'merge' else f" · {r['workload']}")
        for r in rows]
    display = display.rename(columns={'configuration': 'Configuração', 'n': 'Linhas/tabela',
        'successes': 'Válidas', 'failures': 'Falhas', 'mean_runtime_s': 'Média (s)',
        'sample_std_runtime_s': 'DP amostral (s)', 'speedup': 'Speedup', 'relative_efficiency': 'Eficiência relativa'})
    page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dask | Grupo 05 | {html.escape(meta['experiment_id'])}</title>
<style>body{{font:16px system-ui;background:#f3f5f8;color:#172537;max-width:1200px;margin:40px auto;padding:0 24px}}
h1,h2{{color:#143d65}}.card{{background:white;padding:24px;border-radius:12px;margin:20px 0;overflow:auto}}
table{{border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ccd7e1;text-align:right}}
th{{background:#e5edf5}}img{{width:100%;max-width:1000px}}pre{{white-space:pre-wrap;line-height:1.5}}a{{color:#075f98}}</style>
<h1>Quando distribuir a computação compensa?</h1><p>SSC0954 · Grupo 05 · CP3</p>
<div class="card"><strong>{html.escape(meta['environment'])}</strong><p>Experimento: {html.escape(meta['experiment_id'])}
 · modo: {html.escape(config['mode'])} · criado em {html.escape(meta['created_at'])}</p>
<p>{successes} sucessos · {failures} falhas · {len(config['scenarios'])*config['blocks']} ensaios planejados.</p>
<p>{'Coleta completa para este modo.' if all(r['collection_complete'] for r in rows) else 'Coleta incompleta.'}
 {'Requisito de cinco execuções válidas por configuração atendido.' if all(r['cp3_complete'] for r in rows) else 'Requisito CP3 de cinco execuções válidas por configuração NÃO atendido.'}</p></div>
<div class="card"><h2>Resultados reais</h2>{display.to_html(index=False)}<p>Recursos e configurações completas estão nas tabelas CSV exportadas.</p></div>
<div class="card">{images}</div><div class="card"><h2>Relatório e limites</h2><pre>{html.escape(report)}</pre></div>
<div class="card"><a href="report.md">Relatório</a> · <a href="../processed/summary.csv">Tabela CSV</a> ·
<a href="../processed/attempts.csv">Todas as tentativas</a> · <a href="../experiment.json">Ambiente e configuração</a> ·
<a href="../inputs.json">Identificação das entradas</a> · <a href="../raw/">Logs e dados brutos</a></div></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    return {'folder': str(folder), 'configurations': len(rows), 'successes': successes, 'failures': failures,
            'cp3_complete': all(r['cp3_complete'] for r in rows), 'analyzed_at': now()}


def plot_rows(df, output, environment):
    if not df['successes'].sum():
        return
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    environment = textwrap.fill(environment, width=85)
    # Mantemos uma figura por workload para nao sobrepor series no mesmo eixo.
    workloads = sorted(df['workload'].dropna().unique()) if 'workload' in df.columns else ['merge']
    for wl in workloads:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for approach, workers, label in [('pandas', 0, 'Pandas'), ('dask', 1, 'Dask 1 worker'), ('dask', 2, 'Dask 2 workers')]:
            sub = df[(df.approach == approach) & (df.workers == workers) & (df.successes > 0) & (df.workload == wl)]
            if approach == 'dask':
                default_p = 16 if (sub.partitions == 16).any() else 4
                sub = sub[sub.partitions == default_p]
            sub = sub.sort_values('n')
            if len(sub):
                ax.errorbar(sub.n * 48 / 2**20, sub.mean_runtime_s,
                            yerr=sub.sample_std_runtime_s.to_numpy(dtype=float), marker='o', capsize=4,
                            label=label + ' (n=' + ','.join(str(x) for x in sub.successes) + ')')
        ax.set(xlabel='Valores das duas entradas (MiB); arquivos e RSS são distintos',
               ylabel='Runtime (s)', title=f'{environment}\nWorkload: {wl}; barras = desvio padrão amostral')
        ax.legend()
        fig.tight_layout()
        suffix = '' if wl == 'merge' else f'-{wl}'
        fig.savefig(output / f'volume-runtime{suffix}.png', dpi=160)
        plt.close(fig)
    sub = df[(df.volume == 'medium') & (df.approach == 'dask') & (df.workers == 2) & (df.successes > 0) & (df.workload == 'merge')].sort_values('partitions')
    if len(sub):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(sub.partitions, sub.mean_runtime_s, yerr=sub.sample_std_runtime_s.to_numpy(dtype=float), marker='o', capsize=4)
        ax.set(xlabel='Partições nas entradas e saída', ylabel='Runtime (s)',
               title=f'{environment} | volume médio, 2 workers\nBarras = desvio padrão amostral; n={list(sub.successes)}')
        ax.set_xticks(sub.partitions)
        fig.tight_layout()
        fig.savefig(output / 'partition-runtime.png', dpi=160)
        plt.close(fig)
    sub = df[(df.successes > 0) & df.mean_cpu_budget_percent.notna() & df.mean_rss_sum_peak_bytes.notna()]
    if len(sub):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        labels = sub.configuration + ' (n=' + sub.successes.astype(str) + ')'
        axes[0].bar(labels, sub.mean_cpu_budget_percent.astype(float))
        axes[0].set(ylabel='CPU / orçamento (%)', title=f'{environment}\nCPU agregada de processos; memória = soma de RSS amostrado')
        axes[1].bar(labels, sub.mean_rss_sum_peak_bytes.astype(float) / 2**20)
        axes[1].set(ylabel='Média dos picos RSS (MiB)')
        axes[1].tick_params(axis='x', rotation=60)
        fig.tight_layout()
        fig.savefig(output / 'resources.png', dpi=160)
        plt.close(fig)
    sub = df[df.speedup.notna()]
    if len(sub):
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = sub.configuration + ' (n=' + sub.paired_blocks.astype(str) + ' blocos)'
        ax.barh(labels, sub.speedup)
        ax.axvline(1, color='black', linestyle='--')
        ax.set(xlabel='Speedup: média Pandas / média Dask nos blocos comuns',
               title=f'{environment}\nAnálise descritiva; sem teste de significância')
        fig.tight_layout()
        fig.savefig(output / 'speedup.png', dpi=160)
        plt.close(fig)


def make_report(meta, rows, successes, failures):
    lines = [f"# Experimento {meta['experiment_id']}", '', f"Ambiente: {meta['environment']}",
             f"Modo: {meta['config']['mode']}. Sucessos: {successes}. Falhas: {failures}.",
             f"Criado: {meta['created_at']}. Fonte: {meta['source_id']}.", '',
             'Objetivo: comparar inner merge por chave entre duas tabelas iguais em tamanho, com chaves únicas e embaralhadas.',
             'Runtime inclui leitura, grafo, reparticionamento, shuffle, merge e coleta completa no cliente.',
             'Inicialização, aquecimento completo descartado, validação exata e exportação ficam fora do cronômetro.', '',
             '## Observações descritivas']
    for r in rows:
        mean, std = r['mean_runtime_s'], r['sample_std_runtime_s']
        lines.append(f"- {r['configuration']}: {r['successes']}/{r['planned']} válidas; {r['failures']} falhas; " +
                     (f"média={mean:.6f} s; DP amostral=" + (f"{std:.6f} s" if std is not None else 'indisponível (n<2)')
                      if mean is not None else 'sem runtime válido') +
                     (f"; speedup={r['speedup']:.4f}; eficiência relativa={r['relative_efficiency']:.4f}; blocos pareados={r['paired_blocks']}" if r['speedup'] is not None else '') +
                     ('; CP3 incompleto' if not r['cp3_complete'] else ''))
    lines += ['', '## Interpretação e limitações',
              'Speedup é a razão das médias de Pandas e Dask no mesmo volume e blocos comuns. A eficiência divide esse speedup pelos workers; não mede aproveitamento físico diretamente.',
              'Barras representam desvio padrão amostral, não intervalo de confiança. n=1 não permite estimar desvio padrão.',
              'Não há confirmação inferencial de break-even ou significância. Resultados do PC não substituem medições na VM.',
              'Leitura, coordenação, shuffle, coleta, pressão de memória e instrumentação podem influenciar os tempos. Não foram isoladas causas exclusivas.',
              'Processos novos não limpam o cache do sistema operacional. Todos recebem um aquecimento completo e descartado.',
              'Instrumentação ativa: amostrador de processos/cgroup e task stream Dask; dashboard ativo durante Dask. Serviço de resultados deve permanecer desligado nas medições.',
              'CPU em segundos agregados e % do orçamento (100% = orçamento inteiro). Soma de RSS conta páginas compartilhadas repetidamente. Picos entre amostras podem ser perdidos.',
              'Eventos de recepção comprovam transferências entre workers. O campo total do log Dask soma get_nbytes das tarefas (estimativa de tamanho dos objetos em memória), não mede payload serializado nem interfaces de rede. Logs finitos têm ressalva de completude. Nas verificações nativas 001/002/003 o campo histórico interworker_payload_bytes_logged possui essa mesma semântica de estimativa, apesar do nome; não deve ser citado como bytes físicos.',
              'Com um worker há shuffle lógico sem transferência entre workers distintos. Comunicação é entre processos na mesma máquina.',
              'Spill é registrado antes/depois; snapshots podem não detectar pico transitório. Identidade dos workers é comparada; logs preservam diagnóstico.',
              'Volume serializado e overhead exclusivo do scheduler permanecem null: instrumentação específica pendente.',
              'Interferência externa do host não foi eliminada. Git commit é unavailable quando não fornecido; hashes das fontes identificam o código.',
              'O artigo descreve dez repetições iniciais, ampliação e inferência, 4/8 workers e microbenchmark. Este CP3 usa cinco blocos, 1/2 workers e análise descritiva, conforme pedido. O artigo não foi alterado.']
    return '\n'.join(lines) + '\n'


def export(folder, destination):
    folder, destination = Path(folder), Path(destination)
    target = destination / folder.parent.name / folder.name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(folder, target, dirs_exist_ok=True)
    # Snapshot archive preserves previous exports; inputs themselves stay on named volume.
    archive_dir = destination / 'archives'
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = now().replace(':', '').replace('+', '_').replace('.', '_')
    shutil.make_archive(str(archive_dir / f'{folder.name}-{stamp}'), 'zip', root_dir=folder)
    links = []
    for page in destination.glob('*/*/results/index.html'):
        links.append(f'<li><a href="{page.relative_to(destination).as_posix()}">{html.escape(str(page.parent.parent.relative_to(destination)))}</a></li>')
    (destination / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Grupo 05</title>'
                                          '<h1>Experimentos Grupo 05</h1><ul>' + ''.join(sorted(links)) + '</ul>', encoding='utf-8')
    return target


def empty_page(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'index.html'
    if not path.exists():
        path.write_text('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Grupo 05</title>'
                        '<h1>Dask Distributed — Grupo 05</h1><p>Experimento ainda não executado.</p>'
                        '<p>Sem medições Docker ou VM disponíveis. Nenhum resultado fictício.</p></html>', encoding='utf-8')
