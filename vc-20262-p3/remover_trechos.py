#!/usr/bin/env python3
"""
Remove de tracks.csv os quadros de trechos indesejados (replays, closes, cortes)
e, opcionalmente, corta o começo do clipe para calibrar em um quadro com marcos.

Sem --inicio os quadros mantêm a numeração original: homografia.py lê o vídeo
sequencialmente e casa cada quadro com o CSV pelo número, então renumerar sozinho
desalinharia as etapas seguintes. As lacunas são tratadas adiante: metricas.py
ignora velocidades com lacuna > 3 quadros e descarta passes mais longos que
--dur-max.

Com --inicio (e --destino) os quadros anteriores são descartados e o resto é
renumerado a partir de 0, junto com t_s, meta.json (inicio_s, n_frames) e o
frame0.jpg, que passa a ser o quadro do novo início. Serve quando o primeiro
quadro do trecho não mostra marcos suficientes do campo para calibrar.

Idempotente: o original é guardado em tracks_completo.csv na primeira execução
(modo no lugar) e as seguintes sempre partem dele, então dá para trocar a lista.

Uso
---
# No lugar (mantém a numeração):
python remover_trechos.py --saida ../saidas/jogo1 --excluir 0:40-1:00 1:30-1:48 2:05-2:19 3:30-3:43

# Nova pasta começando em 0:05 (reaproveita o rastreamento; não refaz o YOLO):
python remover_trechos.py --saida ../saidas/jogo1 --destino ../saidas/jogo2 --inicio 0:05 \\
    --excluir 0:40-1:00 1:30-1:48 2:05-2:19 3:30-3:43

Tempos em m:ss (ou segundos), contados desde o início do clipe de --saida.
Para desfazer o modo no lugar: --excluir sem argumentos restaura o tracks.csv completo.
Rode a partir da raiz do projeto: meta.json guarda o caminho do vídeo relativo a ela.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
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


def _salvar_frame0(meta: dict, destino: Path) -> None:
    """Grava o quadro em meta['inicio_s'], lido como homografia.py lê (POS_MSEC)."""
    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_MSEC, meta["inicio_s"] * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Não consegui ler o vídeo {meta['video']} em {meta['inicio_s']}s "
                         "(rode a partir da raiz do projeto).")
    cv2.imwrite(str(destino / "frame0.jpg"), frame)


def remover(saida: Path, trechos: list[tuple[float, float]], destino: Path | None, inicio_s: float) -> None:
    if inicio_s > 0 and destino is None:
        raise SystemExit("--inicio exige --destino (não sobrescrevo o rastreamento original).")
    no_lugar = destino is None or destino.resolve() == saida.resolve()
    destino = saida if no_lugar else destino
    destino.mkdir(parents=True, exist_ok=True)

    meta = json.loads((saida / "meta.json").read_text())
    fps = float(meta["fps"])

    original = saida / "tracks_completo.csv"
    if no_lugar and not original.exists():
        (saida / "tracks.csv").replace(original)   # primeira execução: guarda o CSV completo
    tracks = pd.read_csv(original if original.exists() else saida / "tracks.csv")

    deslocamento = round(inicio_s * fps)
    excluir = tracks["frame"] < deslocamento
    for ini, fim in trechos:
        excluir |= (tracks["frame"] >= round(ini * fps)) & (tracks["frame"] < round(fim * fps))

    saida_tracks = tracks[~excluir].copy()
    saida_tracks["frame"] -= deslocamento
    saida_tracks["t_s"] -= inicio_s
    saida_tracks.to_csv(destino / "tracks.csv", index=False)

    novos = [(max(i - inicio_s, 0.0), f - inicio_s) for i, f in trechos if f > inicio_s]
    meta["inicio_s"] = float(meta.get("inicio_s", 0.0)) + inicio_s
    meta["n_frames"] = int(meta["n_frames"]) - deslocamento
    meta["trechos_excluidos_s"] = [[i, f] for i, f in novos]
    (destino / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if not no_lugar:
        _salvar_frame0(meta, destino)

    print(f"Trechos excluídos (tempo de {destino.name}): "
          f"{[f'{i:g}-{f:g}s' for i, f in novos] or 'nenhum'}")
    if inicio_s > 0:
        print(f"Início cortado em {inicio_s:g}s -> {meta['n_frames']} quadros; novo frame0.jpg gerado")
    print(f"Quadros com detecção: {tracks['frame'].nunique()} -> {saida_tracks['frame'].nunique()}")
    print(f"Linhas: {len(tracks)} -> {len(saida_tracks)}  (saída em {destino / 'tracks.csv'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Remove trechos (replays/cortes) do tracks.csv.")
    p.add_argument("--saida", required=True, help="Pasta de saída do detectar_rastrear.py (fonte)")
    p.add_argument("--destino", default=None, help="Nova pasta para o resultado (padrão: no lugar)")
    p.add_argument("--inicio", default="0", metavar="M:SS",
                   help="Descarta o começo do clipe e renumera (exige --destino)")
    p.add_argument("--excluir", nargs="*", default=[], type=_parse_trecho, metavar="INI-FIM",
                   help="Trechos em m:ss, ex.: 0:40-1:00 1:30-1:48 (vazio = só restaura/copia)")
    a = p.parse_args()
    remover(Path(a.saida), a.excluir, Path(a.destino) if a.destino else None, _segundos(a.inicio))
