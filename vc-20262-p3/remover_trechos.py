#!/usr/bin/env python3
"""
Remove de tracks.csv os quadros de trechos indesejados (replays, closes, cortes).

Os quadros mantêm a numeração original: homografia.py lê o vídeo sequencialmente e
casa cada quadro com o CSV pelo número, então renumerar desalinharia as etapas
seguintes. As lacunas são tratadas adiante: metricas.py ignora velocidades com
lacuna > 3 quadros e descarta passes mais longos que --dur-max.

Idempotente: o original é guardado em tracks_completo.csv na primeira execução e
as seguintes sempre partem dele, então dá para trocar a lista de trechos.

Uso
---
python remover_trechos.py --saida ../saidas/jogo1 --excluir 0:40-1:00 1:30-1:48 2:05-2:19 3:30-3:43

Tempos em m:ss (ou segundos), contados desde o início do clipe processado.
Para desfazer: --excluir sem argumentos restaura o tracks.csv completo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _segundos(texto: str) -> float:
    """'1:30' -> 90.0 ; '75' -> 75.0"""
    minutos, _, segs = texto.rpartition(":")
    return (float(minutos) * 60 if minutos else 0.0) + float(segs)


def _parse_trecho(texto: str) -> tuple[float, float]:
    ini, sep, fim = texto.partition("-")
    if not sep:
        raise argparse.ArgumentTypeError(f"Trecho inválido '{texto}': use INICIO-FIM, ex.: 0:40-1:00")
    a, b = _segundos(ini), _segundos(fim)
    if b <= a:
        raise argparse.ArgumentTypeError(f"Trecho inválido '{texto}': o fim deve ser maior que o início")
    return a, b


def remover(pasta: Path, trechos: list[tuple[float, float]]) -> None:
    meta_arq = pasta / "meta.json"
    meta = json.loads(meta_arq.read_text())
    fps = float(meta["fps"])

    original = pasta / "tracks_completo.csv"
    atual = pasta / "tracks.csv"
    if not original.exists():
        atual.replace(original)          # primeira execução: guarda o CSV completo
    tracks = pd.read_csv(original)

    excluir = pd.Series(False, index=tracks.index)
    for ini, fim in trechos:
        excluir |= (tracks["frame"] >= round(ini * fps)) & (tracks["frame"] < round(fim * fps))

    tracks[~excluir].to_csv(atual, index=False)

    quadros_antes = tracks["frame"].nunique()
    quadros_depois = tracks.loc[~excluir, "frame"].nunique()
    meta["trechos_excluidos_s"] = [[ini, fim] for ini, fim in trechos]
    meta_arq.write_text(json.dumps(meta, indent=2))

    print(f"Trechos excluídos: {[f'{i:g}-{f:g}s' for i, f in trechos] or 'nenhum (restaurado)'}")
    print(f"Quadros com detecção: {quadros_antes} -> {quadros_depois}")
    print(f"Linhas: {len(tracks)} -> {int((~excluir).sum())}  (original em {original.name})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Remove trechos (replays/cortes) do tracks.csv.")
    p.add_argument("--saida", required=True, help="Pasta de saída do detectar_rastrear.py")
    p.add_argument("--excluir", nargs="*", default=[], type=_parse_trecho, metavar="INI-FIM",
                   help="Trechos em m:ss, ex.: 0:40-1:00 1:30-1:48 (vazio = restaura o completo)")
    a = p.parse_args()
    remover(Path(a.saida), a.excluir)
