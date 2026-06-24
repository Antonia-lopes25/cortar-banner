#!/usr/bin/env python3
"""
deteccao_fronteiras.py
======================

Detecta as fronteiras entre N banners pela MUDANÇA DE CONTEÚDO entre faixas
de pixels vizinhas — o sinal mais estável, que independe da cor/tipo da
separação (linha branca, sombra, corte seco, fundo de design).

Estratégia:
  1. Decide a orientação pela forma (mais largo que alto -> horizontal).
  2. Mede, ao longo do eixo, o quanto cada faixa difere da anterior.
  3. Escolhe as (N-1) fronteiras como os maiores picos de mudança, MAS
     ancorados perto das posições ideais (comprimento*k/N), porque sabemos
     que são N banners. Isso evita cortar no meio de um banner.
  4. Para cada fronteira candidata, refina para o pico local exato.

Funciona tanto para banners de tamanhos iguais quanto ligeiramente diferentes,
pois a âncora só guia a busca — o pico real define a posição.
"""

import numpy as np
from PIL import Image


def _perfil_mudanca(arr, eixo):
    """Diferença de cor média entre cada faixa e a anterior, ao longo do eixo."""
    a = arr.astype(np.float32)
    if eixo == 0:
        faixas = a.mean(axis=1)   # média por linha -> (H, C)
    else:
        faixas = a.mean(axis=0)   # média por coluna -> (W, C)
    dif = np.zeros(faixas.shape[0], dtype=np.float32)
    dif[1:] = np.sqrt(((faixas[1:] - faixas[:-1]) ** 2).sum(axis=1))
    return dif


def detectar(arr, n=4, orientacao="auto", janela_frac=0.25):
    """
    Retorna (orientacao, cortes) onde cortes são as (n-1) posições de fronteira.

    janela_frac: largura da janela de busca em torno de cada posição ideal,
    como fração do tamanho ideal de um banner. Maior = tolera banners mais
    desiguais; menor = mais preso à divisão uniforme.
    """
    H, W = arr.shape[:2]

    if orientacao == "vertical":
        eixo = 0
    elif orientacao == "horizontal":
        eixo = 1
    else:
        # auto: se a forma é claramente alongada, usa a forma.
        # se for quase quadrada, testa os dois eixos e escolhe o que tem
        # fronteiras mais fortes (a orientação real corta entre cenas distintas).
        proporcao = W / H
        if proporcao >= 1.15:
            eixo = 1
        elif proporcao <= 0.87:
            eixo = 0
        else:
            forca = {}
            for e in (0, 1):
                comp = H if e == 0 else W
                d = _perfil_mudanca(arr, e)
                passo = comp / n
                jan = int(passo * janela_frac) + 1
                soma = 0.0
                for k in range(1, n):
                    ideal = int(round(passo * k))
                    lo = max(1, ideal - jan); hi = min(comp - 1, ideal + jan)
                    if hi > lo:
                        soma += float(d[lo:hi].max())
                forca[e] = soma
            eixo = 0 if forca[0] >= forca[1] else 1

    comprimento = H if eixo == 0 else W
    dif = _perfil_mudanca(arr, eixo)

    passo = comprimento / n
    janela = int(passo * janela_frac) + 1

    cortes = []
    for k in range(1, n):
        ideal = int(round(passo * k))
        lo = max(1, ideal - janela)
        hi = min(comprimento - 1, ideal + janela)
        # pico de mudança dentro da janela ancorada na posição ideal
        local = dif[lo:hi]
        if local.size == 0:
            cortes.append(ideal)
        else:
            cortes.append(lo + int(np.argmax(local)))

    ori = "vertical" if eixo == 0 else "horizontal"
    return ori, sorted(cortes)


def cortar_fronteiras(arr, n=4, orientacao="auto", descartar_faixa=True,
                      janela_frac=0.25):
    """
    Detecta e corta. Se `descartar_faixa`, remove uma fina divisória lisa em
    torno de cada corte (linha branca, etc.); caso contrário corta exato.
    Retorna (banners, orientacao, cortes).
    """
    ori, cortes = detectar(arr, n=n, orientacao=orientacao, janela_frac=janela_frac)
    eixo = 0 if ori == "vertical" else 1
    comprimento = arr.shape[eixo]

    # opcional: expandir cada corte para descartar uma divisória lisa fina
    def faixa_lisa(i):
        if i < 0 or i >= comprimento:
            return False
        linha = arr[i, :, :] if eixo == 0 else arr[:, i, :]
        return float(linha.astype(np.float32).var(axis=0).mean()) <= 120.0

    intervalos = []
    prev = 0
    for c in cortes:
        ini, fim = c, c
        if descartar_faixa and faixa_lisa(c):
            max_exp = int((comprimento / n) * 0.05)
            while ini - 1 > prev and faixa_lisa(ini - 1) and (c - (ini - 1)) <= max_exp:
                ini -= 1
            while fim + 1 < comprimento and faixa_lisa(fim + 1) and ((fim + 1) - c) <= max_exp:
                fim += 1
        intervalos.append((prev, ini))
        prev = fim + 1 if fim > ini else c
    intervalos.append((prev, comprimento))

    banners = []
    for (a0, a1) in intervalos:
        if a1 <= a0:
            a1 = a0 + 1
        banners.append(arr[a0:a1, :, :] if eixo == 0 else arr[:, a0:a1, :])
    return banners, ori, cortes
