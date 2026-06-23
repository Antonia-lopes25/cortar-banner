#!/usr/bin/env python3
"""
deteccao_cv.py
==============

Detecção de banners e de borda por VISÃO COMPUTACIONAL (OpenCV), com limiar
AUTOMÁTICO por imagem (Otsu) — sem número mágico de cor/limiar.

Pensado para ser mais abrangente que a heurística de limiar fixo:
  - adapta-se ao contraste de cada imagem;
  - acha os N blocos de conteúdo (para separar os banners empilhados/lado a lado);
  - acha a borda apertada de cada banner (corte seco, removendo fundo + sombra);
  - quando a imagem é ambígua (sem borda clara), corta o melhor possível e
    devolve assim mesmo (nunca recusa).

Funções principais:
  mascara_conteudo(arr)             -> máscara binária (conteúdo=255, fundo=0)
  blocos_cv(arr, eixo)              -> [(ini,fim), ...] blocos ao longo do eixo
  bbox_conteudo(banner)             -> (x0,y0,x1,y1) borda apertada do conteúdo
"""

import cv2
import numpy as np


def _cor_fundo(arr, frac=0.02):
    """Mediana dos pixels de borda como cor de fundo (robusta a canto destoante)."""
    a = arr.astype(np.float32)
    H, W = a.shape[:2]
    cy = max(1, int(H * frac))
    cx = max(1, int(W * frac))
    borda = np.concatenate([
        a[:cy, :].reshape(-1, 3), a[H - cy:, :].reshape(-1, 3),
        a[:, :cx].reshape(-1, 3), a[:, W - cx:].reshape(-1, 3),
    ], axis=0)
    return np.median(borda, axis=0)


def mascara_conteudo(arr):
    """
    Constrói uma máscara binária onde o CONTEÚDO é 255 e o FUNDO (incluindo a
    sombra suave do mockup) é 0.

    Como: calcula a distância de cada pixel à cor de fundo, normaliza para
    0..255 e aplica Otsu — que escolhe AUTOMATICAMENTE o limiar que melhor
    separa as duas populações (fundo+sombra vs conteúdo) naquela imagem
    específica. Depois usa morfologia para fechar buracos e remover ruído.
    """
    fundo = _cor_fundo(arr)
    dist = np.sqrt(((arr.astype(np.float32) - fundo) ** 2).sum(axis=2))
    dist = np.clip(dist, 0, 255).astype(np.uint8)

    # Otsu: limiar automático por imagem
    _, m = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # consolida: fecha buracos dentro do conteúdo e remove pontos isolados
    H, W = arr.shape[:2]
    k = max(3, int(min(H, W) * 0.01))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return m


def blocos_cv(arr, eixo, n_alvo=None, min_bloco_frac=0.04):
    """
    Separa a imagem em blocos de conteúdo ao longo de `eixo`
    (0 = empilhado vertical, 1 = lado a lado horizontal), projetando a máscara.

    Diferente do limiar fixo: o "vazio" entre banners é definido pela máscara
    Otsu, então faixas de fundo de QUALQUER cor (clara, escura, sombreada)
    são separadores. Retorna lista de (ini, fim).
    """
    m = mascara_conteudo(arr)
    perfil = (m > 0).mean(axis=1) if eixo == 0 else (m > 0).mean(axis=0)
    comprimento = perfil.shape[0]

    dentro = perfil > 0.10  # 10% da linha/coluna com conteúdo
    blocos = []
    i = 0
    while i < comprimento:
        if dentro[i]:
            j = i
            while j < comprimento and dentro[j]:
                j += 1
            blocos.append((i, j))
            i = j
        else:
            i += 1

    # funde gaps pequenos (ruído interno do banner)
    min_gap = max(1, int(comprimento * 0.005))
    fundidos = []
    for b in blocos:
        if fundidos and b[0] - fundidos[-1][1] <= min_gap:
            fundidos[-1] = (fundidos[-1][0], b[1])
        else:
            fundidos.append(list(b))
    min_bloco = int(comprimento * min_bloco_frac)
    fundidos = [tuple(b) for b in fundidos if (b[1] - b[0]) >= min_bloco]
    return fundidos


def bbox_conteudo(banner, margem=1, max_trim_frac=0.08):
    """
    Acha a borda apertada do conteúdo de UM banner via maior contorno na máscara.
    Devolve (x0, y0, x1, y1).

    SEGURANÇA CONTRA CORTE DE DESIGN: o trim de cada lado é limitado a
    `max_trim_frac` da dimensão (padrão 8%). Assim só molduras/sombras FINAS são
    removidas; uma área clara grande (ex.: fundo branco que faz parte do design,
    comum em banners de moda) é preservada, pois cortá-la passaria do teto.

    Se não houver contorno utilizável (imagem ambígua / banner sangrando até a
    borda), devolve a imagem inteira — corta o melhor possível, nunca recusa.
    """
    H, W = banner.shape[:2]
    m = mascara_conteudo(banner)

    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0, 0, W, H

    area_min = 0.01 * H * W
    caixas = [cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) >= area_min]
    if not caixas:
        c = max(contornos, key=cv2.contourArea)
        caixas = [cv2.boundingRect(c)]

    x0 = min(b[0] for b in caixas)
    y0 = min(b[1] for b in caixas)
    x1 = max(b[0] + b[2] for b in caixas)
    y1 = max(b[1] + b[3] for b in caixas)

    # margem para dentro
    x0 = min(x0 + margem, W - 1); y0 = min(y0 + margem, H - 1)
    x1 = max(x1 - margem, x0 + 1); y1 = max(y1 - margem, y0 + 1)

    # TETO DE TRIM: nunca aparar mais que max_trim_frac de cada lado.
    # Se o conteúdo só começa muito para dentro, é fundo de design -> preserva.
    max_x = int(W * max_trim_frac)
    max_y = int(H * max_trim_frac)
    x0 = min(x0, max_x)
    y0 = min(y0, max_y)
    x1 = max(x1, W - max_x)
    y1 = max(y1, H - max_y)

    return x0, y0, x1, y1
