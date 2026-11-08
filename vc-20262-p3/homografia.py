#!/usr/bin/env python3
"""
Do pixel ao campo: homografia + compensação do movimento da câmera.

Por que isto é necessário? (Ahmad Yousef et al., 2025, Fig. 8)
A câmera lateral introduz distorção de perspectiva: 1 pixel perto da câmera vale
poucos centímetros, 1 pixel do outro lado do campo vale quase meio metro.
Distâncias, velocidades e o eXP precisam ser calculados em METROS.

Ideia
-----
1. Calibração: clicamos >= 4 marcos conhecidos do campo (ver campo.MARCOS) num quadro
   q e estimamos H_q tal que  [x_m, y_m, 1]ᵀ ~ H_q · [u, v, 1]ᵀ  (pixels de q → metros).
2. Câmera: a transmissão faz pan/zoom e corta entre câmeras. Dividimos o vídeo em
   SEGMENTOS (planos contínuos). Dentro de um segmento estimamos G_t, que leva o quadro
   t ao primeiro quadro do segmento, casando pontos ORB do gramado (sem jogadores e
   placar) com um quadro-chave e, se preciso, com os quadros anteriores. Um corte ou um
   quadro que não casa com nada abre um segmento novo.
3. Projeção: o quadro t usa o quadro calibrado q mais próximo DO MESMO segmento:
   ponto → campo por  H_q · G_q⁻¹ · G_t · [u, v, 1]ᵀ. O encadeamento da câmera acumula
   erro de perspectiva ao longo de um pan (no jogo2, ~2-4 m em 20 s), por isso num pan
   longo vale calibrar quadros nas duas pontas. Segmento sem quadro calibrado (close,
   replay, outro ângulo) fica sem posição em metros (NaN).

Subcomandos (ordem de uso)
--------------------------
python homografia.py calibrar --saida ../saidas/jogo1 --pontos meio_sup,circulo_sup,circulo_inf,circulo_esq,circulo_dir
python homografia.py camera   --saida ../saidas/jogo1
python homografia.py calibrar --saida ../saidas/jogo1 --quadro 600 --pontos ga_dir_sup,ga_dir_inf,pa_dir_sup,pa_dir_inf
python homografia.py conferir --saida ../saidas/jogo1
python homografia.py projetar --saida ../saidas/jogo1

Cada quadro clicado precisa de >= 4 marcos (não colineares). Clicar de novo num quadro
substitui os cliques dele.

Sem interface gráfica (ex.: Colab)? Crie <saida>/calibracao.json à mão com
{"pixels": [[u, v], ...], "marcos": ["meio_sup", ...], "quadros": [0, ...]}
("quadros" é opcional; sem ele, todos os cliques são do quadro 0) e rode
"calibrar --sem-janela".
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from campo import COMPRIMENTO, LARGURA, MARCOS


# ── Utilitários ────────────────────────────────────────────────────────────

def aplicar_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica a homografia H (3×3) a pontos (N×2) em coordenadas cartesianas."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _imagem(pasta: Path, meta: dict, quadro: int) -> np.ndarray:
    """Quadro `quadro` do trecho (0 = frame0.jpg), lido como camera() lê o vídeo."""
    if quadro == 0 and (pasta / "frame0.jpg").exists():
        return cv2.imread(str(pasta / "frame0.jpg"))
    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_MSEC, (meta["inicio_s"] + quadro / meta["fps"]) * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Não consegui ler o quadro {quadro} de {meta['video']}.")
    return img


def _camera(pasta: Path) -> tuple[np.ndarray, np.ndarray]:
    """G (n×3×3, quadro → referência do segmento) e ref (n,), o quadro de referência do
    segmento de cada quadro (-1 = sem câmera)."""
    G = np.load(pasta / "camera.npy")
    info = json.loads((pasta / "camera_cortes.json").read_text())
    if "segmentos" not in info:
        raise SystemExit("camera.npy é de uma versão antiga: rode 'homografia.py camera' de novo.")
    ref = np.full(len(G), -1)
    for ini, fim in info["segmentos"]:
        ref[ini:fim + 1] = ini
    return G, ref


def _homografias(pasta: Path) -> dict[int, np.ndarray]:
    """H_q (pixels do quadro q → metros) de cada quadro calibrado."""
    cal = json.loads((pasta / "calibracao.json").read_text())
    if "homografias" not in cal:
        raise SystemExit("calibracao.json sem homografias: rode 'homografia.py calibrar --sem-janela'.")
    return {int(q): np.array(H) for q, H in cal["homografias"].items()}


def _tiles(paineis: list[np.ndarray], colunas: int = 3) -> np.ndarray:
    while len(paineis) % colunas:
        paineis.append(np.zeros_like(paineis[0]))
    return np.vstack([np.hstack(paineis[i:i + colunas]) for i in range(0, len(paineis), colunas)])


# ── 1. Calibração ──────────────────────────────────────────────────────────

def _clicar_pontos(img: np.ndarray, marcos: list[str]) -> list[list[float]]:
    """Janela OpenCV: clique nos marcos na ordem pedida. 'u' desfaz, 'q' termina."""
    pts: list[list[float]] = []
    janela = "Calibracao - clique nos marcos (u = desfazer, q = sair)"

    def ao_clicar(evento, x, y, *_):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pts) < len(marcos):
            pts.append([float(x), float(y)])

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, ao_clicar)
    while True:
        vis = img.copy()
        for (x, y), nome in zip(pts, marcos):
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(vis, nome, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        prox = marcos[len(pts)] if len(pts) < len(marcos) else "pronto! tecle q"
        cv2.putText(vis, f"Proximo: {prox}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(janela, vis)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("u") and pts:
            pts.pop()
        elif tecla == ord("q"):
            break
    cv2.destroyAllWindows()
    return pts


def calibrar(pasta: Path, marcos: list[str], sem_janela: bool, quadro: int):
    arq = pasta / "calibracao.json"
    meta = json.loads((pasta / "meta.json").read_text())
    cal = json.loads(arq.read_text()) if arq.exists() else {"pixels": [], "marcos": []}
    pixels, nomes = cal["pixels"], cal["marcos"]
    quadros = cal.get("quadros", [0] * len(pixels))

    if not sem_janela:
        desconhecidos = [m for m in marcos if m not in MARCOS]
        if desconhecidos or not marcos:
            raise SystemExit(f"Marcos desconhecidos/ausentes: {desconhecidos}. Opções: {list(MARCOS)}")
        novos = _clicar_pontos(_imagem(pasta, meta, quadro), marcos)
        fica = [i for i, q in enumerate(quadros) if q != quadro]   # refazer um quadro substitui os cliques dele
        pixels = [pixels[i] for i in fica] + novos
        nomes = [nomes[i] for i in fica] + marcos[:len(novos)]
        quadros = [quadros[i] for i in fica] + [quadro] * len(novos)
    if not pixels:
        raise SystemExit("Nenhum clique em calibracao.json.")

    # Uma homografia por quadro clicado. Juntar cliques de quadros diferentes do mesmo
    # segmento (via G) herdaria a deriva da câmera entre eles.
    homografias, erro = {}, np.full(len(pixels), np.nan)
    for q in sorted(set(quadros)):
        idx = [i for i, qi in enumerate(quadros) if qi == q]
        if len(idx) < 4:
            print(f"Quadro {q}: só {len(idx)} ponto(s), são necessários 4 (não colineares).")
            continue
        src = np.array([pixels[i] for i in idx], dtype=np.float64)
        dst = np.array([MARCOS[nomes[i]] for i in idx], dtype=np.float64)
        # RANSAC só faz sentido com pontos de sobra: o modelo mínimo tem 4 pontos e sempre os
        # ajusta com erro 0, então com 5-7 cliques ele só descartaria o pior. Abaixo disso,
        # mínimos quadrados usam todos os cliques.
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(src) >= 8 else 0, 1.0)
        if H is None:
            print(f"Quadro {q}: pontos degenerados (colineares?), sem calibração.")
            continue
        homografias[str(q)] = H.tolist()
        erro[idx] = np.linalg.norm(aplicar_h(H, src) - dst, axis=1)

    json.dump({"pixels": pixels, "marcos": nomes, "quadros": quadros, "homografias": homografias,
               "erro_reprojecao_m": [None if np.isnan(e) else float(e) for e in erro]},
              open(arq, "w"), indent=2)
    print(f"Calibração salva em {arq} ({len(homografias)} quadro(s) calibrado(s))")
    for m, q, e in zip(nomes, quadros, erro):
        print(f"  {m:18s} quadro {q:5d}  erro = {'-' if np.isnan(e) else f'{e:.2f} m'}")

    # Conferência: grade de 5 m sobre cada quadro calibrado
    paineis = []
    for q, H in sorted((int(k), np.array(v)) for k, v in homografias.items()):
        img = _imagem(pasta, meta, q)
        _desenhar(img, np.linalg.inv(H), _linhas_grade(), (0, 255, 255))
        cv2.putText(img, f"quadro {q}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        paineis.append(cv2.resize(img, (640, 360)))
    if paineis:
        cv2.imwrite(str(pasta / "calibracao_grade.jpg"), _tiles(paineis, min(3, len(paineis))))
        print(f"Confira a grade de 5 m em {pasta/'calibracao_grade.jpg'}")


# ── 2. Movimento da câmera ─────────────────────────────────────────────────

# Casamento de pontos. Medido no jogo2 (quadros seguidos de câmera lateral): sem CLAHE e
# com FAST 10, a máscara do gramado rendia ~90-600 pontos e mediana de 13-139 inliers,
# pouco para encadear milhares de quadros; com CLAHE + FAST 5, ~2000-2700 pontos e mediana
# de 220-450 inliers. Esses pontos (textura da grama) só se repetem entre quadros próximos:
# depois de um corte não dá para reconectar ao trecho anterior, e um segmento novo começa.
ORB_PONTOS, ORB_FAST = 3000, 5
MIN_INLIERS_SEGUIDO = 40   # contra os quadros anteriores (sobreposição quase total)
MIN_INLIERS_CHAVE = 60     # contra o quadro-chave: um erro aqui contamina o resto do segmento
RECENTES = 6               # quadros anteriores tentados antes de abrir um segmento novo
MIN_PONTOS_SEGMENTO = 50   # quadro com menos pontos não abre segmento (preto, close extremo)


def _mascara(frame: np.ndarray, caixas: np.ndarray, margem_placar: float) -> np.ndarray:
    """Máscara dos pixels úteis para estimar o movimento da câmera.

    Mantém só o GRAMADO (com as linhas): a torcida se mexe e os painéis de LED
    são animados, o que engana o RANSAC e faz a homografia "derivar".
    Remove também os jogadores (se movem por conta própria) e o placar.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    # Fechamento morfológico: inclui as linhas brancas que cortam o gramado
    verde = cv2.morphologyEx(verde, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    verde = cv2.erode(verde, np.ones((15, 15), np.uint8))   # afasta da borda com as placas
    verde[: int(h * margem_placar), :] = 0
    for x1, y1, x2, y2 in caixas.astype(int):
        verde[max(y1 - 5, 0):y2 + 5, max(x1 - 5, 0):x2 + 5] = 0
    return verde


def _plausivel(M: np.ndarray | None, w: int, h: int, area_min: float, area_max: float,
               desloc_max: float | None = None) -> bool:
    """Rejeita homografias degeneradas: o retângulo do quadro, transformado por M, tem
    de continuar um quadrilátero convexo, na mesma orientação (sem espelhar), à frente
    da câmera, com área entre area_min e area_max da original e, se pedido, com cantos
    deslocados no máximo desloc_max pixels. RANSAC aceita com facilidade matrizes quase
    singulares; encadeadas, elas colapsam todos os quadros seguintes em um ponto."""
    if M is None or not np.isfinite(M).all():
        return False
    cantos = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    hom = np.c_[cantos, np.ones(4)] @ M.T
    if (hom[:, 2] <= 1e-9).any():
        return False
    q = hom[:, :2] / hom[:, 2:]
    arestas = np.roll(q, -1, axis=0) - q
    seguintes = np.roll(arestas, -1, axis=0)
    if (arestas[:, 0] * seguintes[:, 1] - arestas[:, 1] * seguintes[:, 0] <= 0).any():
        return False                                  # côncavo, torcido ou espelhado
    area = 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    if not area_min <= area / (w * h) <= area_max:
        return False
    return desloc_max is None or float(np.abs(q - cantos).max()) <= desloc_max


def _casar(kp, des, kp_ref, des_ref, bf):
    """Homografia (quadro atual → referência) por ORB + teste da razão de Lowe + RANSAC."""
    if des is None or des_ref is None or len(kp) < 10 or len(kp_ref) < 10:
        return None, 0
    pares = bf.knnMatch(des, des_ref, k=2)
    bons = [m for m, *o in pares if o and m.distance < 0.75 * o[0].distance]
    if len(bons) < 10:
        return None, 0
    p = np.float32([kp[m.queryIdx].pt for m in bons])
    q = np.float32([kp_ref[m.trainIdx].pt for m in bons])
    M, ok = cv2.findHomography(p, q, cv2.RANSAC, 2.0)
    return M, int(ok.sum()) if ok is not None else 0


def movimento_camera(pasta: Path, limiar_corte: float, margem_placar: float):
    """Divide o trecho em segmentos de câmera e estima G_t dentro de cada um.

    Para cada quadro tenta, em ordem: o quadro-chave do segmento (menos deriva) e os
    últimos quadros válidos, do mais recente ao mais antigo (o que casar vira a nova chave). Cada matriz
    passa por _plausivel. Se nada casa, ou se há corte, o quadro abre um segmento novo
    (G = identidade). Trechos excluídos em meta.json ficam sem câmera (NaN).
    """
    meta = json.loads((pasta / "meta.json").read_text())
    tracks = pd.read_csv(pasta / "tracks.csv")
    caixas_por_frame = {f: g[["x1", "y1", "x2", "y2"]].to_numpy()
                        for f, g in tracks.groupby("frame")}
    fps, n, w = meta["fps"], meta["n_frames"], meta["largura"]
    h = meta["altura"]
    excluido = np.zeros(n, dtype=bool)
    for ini, fim in meta.get("trechos_excluidos_s", []):
        excluido[round(ini * fps):round(fim * fps)] = True

    cap = cv2.VideoCapture(meta["video"])
    cap.set(cv2.CAP_PROP_POS_MSEC, meta["inicio_s"] * 1000)
    orb = cv2.ORB_create(nfeatures=ORB_PONTOS, fastThreshold=ORB_FAST)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def casar(kp, des, ref, minimo, area_min, area_max, desloc_max=None):
        M, inliers = _casar(kp, des, ref[1], ref[2], bf)
        return M if inliers >= minimo and _plausivel(M, w, h, area_min, area_max, desloc_max) else None

    G = np.full((n, 3, 3), np.nan)
    segmentos: list[list[int]] = []   # [primeiro, último] quadro; o primeiro é a referência
    recentes: deque = deque(maxlen=RECENTES)   # (quadro, keypoints, descritores, G) válidos
    chave = None
    cortes, n_chaves = [], 0
    ant_hist = None

    for f in tqdm(range(n), desc="Câmera"):
        ok, frame = cap.read()
        if not ok:
            G = G[:f]
            break
        if excluido[f]:                      # trecho removido (replay): sem câmera
            recentes.clear()
            ant_hist = None
            continue

        # Corte de câmera: queda brusca na correlação dos histogramas
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if ant_hist is not None and cv2.compareHist(ant_hist, hist, cv2.HISTCMP_CORREL) < limiar_corte:
            cortes.append(f)
            recentes.clear()
        ant_hist = hist

        cinza = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        kp, des = orb.detectAndCompute(cinza, _mascara(frame, caixas_por_frame.get(f, np.empty((0, 4))), margem_placar))

        Gf = None
        if recentes:
            M = casar(kp, des, chave, MIN_INLIERS_CHAVE, 0.3, 3.5)
            if M is not None:
                Gf = chave[3] @ M
            else:
                for cand in reversed(recentes):
                    M = casar(kp, des, cand, MIN_INLIERS_SEGUIDO, 0.7, 1.4, desloc_max=0.25 * w)
                    if M is not None:
                        Gf, chave = cand[3] @ M, cand
                        n_chaves += 1
                        break
        if Gf is None:                       # corte ou nada casou: segmento novo
            if des is None or len(kp) < MIN_PONTOS_SEGMENTO:
                recentes.clear()
                continue
            Gf = np.eye(3)
            chave = (f, kp, des, Gf)
            segmentos.append([f, f])
        else:
            Gf = Gf / Gf[2, 2]               # evita que o produto de matrizes estoure/zere
            segmentos[-1][1] = f
        G[f] = Gf
        recentes.append((f, kp, des, Gf))
    cap.release()

    np.save(pasta / "camera.npy", G)
    json.dump({"cortes_frames": cortes, "segmentos": segmentos}, open(pasta / "camera_cortes.json", "w"))
    uteis = int((~excluido[:len(G)]).sum())
    longos = [s for s in segmentos if s[1] - s[0] + 1 >= 2 * fps]
    em_longos = sum(s[1] - s[0] + 1 for s in longos)
    print(f"Homografias salvas em {pasta/'camera.npy'} ({len(G)} quadros, {len(segmentos)} segmentos, "
          f"{n_chaves} trocas de quadro-chave)")
    print(f"  {len(longos)} segmento(s) de >= 2 s cobrem {em_longos} de {uteis} quadros fora dos trechos excluídos:")
    for a, b in longos:
        print(f"    quadros {a:5d}-{b:5d}  ({a / fps:6.1f}-{(b + 1) / fps:6.1f}s)")
    print("Calibre cada segmento de câmera lateral com 'calibrar --quadro N' (N dentro do segmento)"
          " e confira com 'conferir'.")


# ── Desenho das linhas do campo (calibrar e conferir) ──────────────────────

def _linhas_grade(passo: float = 5.0) -> list[np.ndarray]:
    """Grade de `passo` metros cobrindo o campo."""
    xs = [np.array([[x, 0], [x, LARGURA]]) for x in np.arange(0, COMPRIMENTO + 0.1, passo)]
    ys = [np.array([[0, y], [COMPRIMENTO, y]]) for y in np.arange(0, LARGURA + 0.1, passo)]
    return xs + ys


def _linhas_campo() -> list[np.ndarray]:
    """Marcações reais do campo (linhas, círculo central, áreas), como polilinhas em metros."""
    def ret(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    t = np.linspace(0, 2 * np.pi, 73)
    meio = COMPRIMENTO / 2
    linhas = [ret(0, 0, COMPRIMENTO, LARGURA),
              np.array([[meio, 0], [meio, LARGURA]]),
              np.c_[meio + 9.15 * np.cos(t), LARGURA / 2 + 9.15 * np.sin(t)]]
    for prof, larg in ((16.5, 40.32), (5.5, 18.32)):
        y0, y1 = LARGURA / 2 - larg / 2, LARGURA / 2 + larg / 2
        linhas += [ret(0, y0, prof, y1), ret(COMPRIMENTO - prof, y0, COMPRIMENTO, y1)]
    return linhas


def _desenhar(img: np.ndarray, H_campo_img: np.ndarray, linhas: list[np.ndarray], cor) -> None:
    """Desenha polilinhas do campo (metros) na imagem. Cada polilinha é amostrada a cada
    ~0,5 m e só os pontos à frente da câmera (w > 0) são ligados."""
    for lin in linhas:
        pts = [lin[0]]
        for a, b in zip(lin[:-1], lin[1:]):
            k = max(int(np.hypot(*(b - a)) / 0.5), 1)
            pts += list(a + (b - a) * np.linspace(0, 1, k + 1)[1:, None])
        hom = np.c_[np.array(pts), np.ones(len(pts))] @ H_campo_img.T
        frente = hom[:, 2] > 1e-9
        xy = np.clip(hom[:, :2] / np.where(frente, hom[:, 2], 1.0)[:, None], -1e4, 1e4)
        for i in range(len(xy) - 1):
            if frente[i] and frente[i + 1]:
                cv2.line(img, tuple(int(v) for v in xy[i]), tuple(int(v) for v in xy[i + 1]), cor, 2)


# ── 3. Projeção ────────────────────────────────────────────────────────────

class _Projecao:
    """Pixels do quadro f → metros, usando o quadro calibrado mais próximo do mesmo segmento."""

    def __init__(self, Hs: dict[int, np.ndarray], G: np.ndarray, ref: np.ndarray):
        self.G, self.ref = G, ref
        self.por_segmento: dict[int, list[tuple[int, np.ndarray]]] = {}
        for q, H in sorted(Hs.items()):
            if q < len(G) and ref[q] >= 0 and np.isfinite(G[q]).all():
                # H_q · G_q⁻¹ leva o quadro de referência do segmento ao campo
                self.por_segmento.setdefault(int(ref[q]), []).append((q, H @ np.linalg.inv(G[q])))
        sem_camera = sorted(set(Hs) - {q for lst in self.por_segmento.values() for q, _ in lst})
        if sem_camera:
            print(f"ATENÇÃO: quadro(s) calibrado(s) sem câmera estimada (ignorados): {sem_camera}")

    def __call__(self, f: int) -> np.ndarray | None:
        if f >= len(self.G) or not np.isfinite(self.G[f]).all():
            return None
        candidatos = self.por_segmento.get(int(self.ref[f]))
        if not candidatos:
            return None
        _, H_ref = min(candidatos, key=lambda c: abs(c[0] - f))
        return H_ref @ self.G[f]


def projetar(pasta: Path, sem_camera: bool):
    Hs = _homografias(pasta)
    if sem_camera:                           # câmera fixa: a calibração do quadro 0 vale para tudo
        n = json.loads((pasta / "meta.json").read_text())["n_frames"]
        G, ref = np.repeat(np.eye(3)[None], n, axis=0), np.zeros(n, dtype=int)
        Hs = {0: Hs[0]}
    else:
        G, ref = _camera(pasta)
    transf = _Projecao(Hs, G, ref)

    def para_campo(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        xy = np.full((len(df), 2), np.nan)
        for f, idx in df.groupby("frame").indices.items():
            Ht = transf(f)
            if Ht is not None:               # sem câmera calibrada: posição em metros fica NaN
                xy[idx] = aplicar_h(Ht, df.iloc[idx][["px", "py"]].to_numpy())
        df["x_m"], df["y_m"] = xy[:, 0], xy[:, 1]
        # Descarta pontos muito fora do campo (erros de projeção/detecções na arquibancada)
        fora = (df["x_m"] < -5) | (df["x_m"] > COMPRIMENTO + 5) | (df["y_m"] < -5) | (df["y_m"] > LARGURA + 5)
        df.loc[fora, ["x_m", "y_m"]] = np.nan
        return df

    calibrados = sum(transf(f) is not None for f in range(len(G)))
    print(f"Quadros com câmera calibrada: {calibrados} de {len(G)} ({100 * calibrados / len(G):.1f}%)")
    if not sem_camera:
        fps = json.loads((pasta / "meta.json").read_text())["fps"]
        faltam = [f"{a / fps:.1f}-{(b + 1) / fps:.1f}s (ex.: --quadro {(a + b) // 2})"
                  for a, b in json.loads((pasta / "camera_cortes.json").read_text())["segmentos"]
                  if b - a + 1 >= 2 * fps and a not in transf.por_segmento]
        if faltam:
            print(f"  Segmentos de >= 2 s sem calibração (calibre os de câmera lateral): {faltam}")
    tracks = pd.read_csv(pasta / "tracks.csv")
    jog = para_campo(tracks[tracks["classe"] != "bola"])
    jog.to_csv(pasta / "tracks_campo.csv", index=False)
    print(f"Jogadores projetados: {jog['x_m'].notna().mean()*100:.1f}% válidos → tracks_campo.csv")

    if (pasta / "bola.csv").exists():
        bola = para_campo(pd.read_csv(pasta / "bola.csv"))
        bola.to_csv(pasta / "bola_campo.csv", index=False)
        print(f"Bola projetada: {bola['x_m'].notna().mean()*100:.1f}% válidos → bola_campo.csv")


# ── 4. Conferência visual ──────────────────────────────────────────────────

def conferir(pasta: Path, n_amostras: int):
    """Desenha as marcações do campo em quadros espalhados pelos trechos calibrados. Se a
    câmera e a calibração estão boas, as linhas desenhadas caem sobre as linhas reais do
    gramado em todos os quadros."""
    meta = json.loads((pasta / "meta.json").read_text())
    G, ref = _camera(pasta)
    transf = _Projecao(_homografias(pasta), G, ref)
    validos = [f for f in range(len(G)) if transf(f) is not None]
    if not validos:
        raise SystemExit("Nenhum quadro com câmera calibrada.")
    amostras = [validos[i] for i in np.linspace(0, len(validos) - 1, n_amostras).round().astype(int)]
    paineis = []
    for f in amostras:
        img = _imagem(pasta, meta, f)
        _desenhar(img, np.linalg.inv(transf(f)), _linhas_campo(), (0, 255, 255))
        img = cv2.resize(img, (640, 360))
        cv2.putText(img, f"{f / meta['fps']:.1f}s (q{f})", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        paineis.append(img)
    cv2.imwrite(str(pasta / "camera_conferencia.jpg"), _tiles(paineis))
    print(f"Conferência em {pasta / 'camera_conferencia.jpg'} ({len(amostras)} de {len(validos)} "
          f"quadros com câmera calibrada)")


def main():
    p = argparse.ArgumentParser(description="Homografia pixel → campo (metros)")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrar")
    c.add_argument("--saida", required=True)
    c.add_argument("--pontos", default="", help="Marcos separados por vírgula (ver campo.MARCOS)")
    c.add_argument("--quadro", type=int, default=0, help="Quadro clicado (0 = frame0.jpg; > 0 exige camera)")
    c.add_argument("--sem-janela", action="store_true", help="Lê os pixels de calibracao.json")
    m = sub.add_parser("camera")
    m.add_argument("--saida", required=True)
    m.add_argument("--limiar-corte", type=float, default=0.5, help="Correlação mínima entre quadros")
    m.add_argument("--margem-placar", type=float, default=0.12, help="Fração superior ignorada")
    j = sub.add_parser("projetar")
    j.add_argument("--saida", required=True)
    j.add_argument("--sem-camera", action="store_true", help="Assume câmera fixa (usa só a calibração do quadro 0)")
    k = sub.add_parser("conferir")
    k.add_argument("--saida", required=True)
    k.add_argument("--amostras", type=int, default=12, help="Quadros desenhados na conferência")
    a = p.parse_args()

    pasta = Path(a.saida)
    if a.cmd == "calibrar":
        calibrar(pasta, [s for s in a.pontos.split(",") if s], a.sem_janela, a.quadro)
    elif a.cmd == "camera":
        movimento_camera(pasta, a.limiar_corte, a.margem_placar)
    elif a.cmd == "conferir":
        conferir(pasta, a.amostras)
    else:
        projetar(pasta, a.sem_camera)


if __name__ == "__main__":
    main()
