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
    """
    Perfil de fronteira ao longo do eixo. Para cada posição, mede a FRAÇÃO da
    largura/altura perpendicular que muda fortemente em relação à faixa anterior.

    Esse critério é muito mais robusto que a média: a divisa real entre dois
    banners é uma linha onde QUASE TODA a extensão perpendicular troca de
    conteúdo de uma vez (fração alta). Já uma mudança interna a um banner (uma
    borda de foto, um bloco de cor) muda só parte da extensão (fração baixa),
    então não engana mais o detector.
    """
    a = arr.astype(np.float32)
    if eixo == 0:
        difpix = np.sqrt(((a[1:] - a[:-1]) ** 2).sum(axis=2))  # (H-1, W)
        frac = (difpix > 60).mean(axis=1)
    else:
        difpix = np.sqrt(((a[:, 1:] - a[:, :-1]) ** 2).sum(axis=2))  # (H, W-1)
        frac = (difpix > 60).mean(axis=0)
    perfil = np.zeros(a.shape[eixo], dtype=np.float32)
    perfil[1:] = frac
    return perfil


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

    a = arr.astype(np.float32)

    def linha(i):
        return a[i, :, :] if eixo == 0 else a[:, i, :]

    def eh_separador(i):
        """Linha é separador se for LISA (baixa variância) E CLARA (faixa branca,
        o caso comum). Exigir clareza evita confundir conteúdo escuro liso com
        divisória. Faixas de cor sólida muito diferentes do banner também passam
        pelo teste de variância na transição, mas a clareza cobre o caso real."""
        if i < 0 or i >= comprimento:
            return False
        L = linha(i)
        lisa = float(L.var(axis=0).mean()) <= 200.0
        clara = float(L.mean()) >= 225.0
        return lisa and clara

    intervalos = []
    prev = 0
    for c in cortes:
        ini, fim = c, c
        if descartar_faixa:
            # procura o separador numa janela ao redor do corte (o corte pode
            # cair na borda da faixa, não no centro). Limite de busca generoso.
            busca = max(4, int((comprimento / n) * 0.06))
            # acha um ponto separador perto de c
            centro = None
            for d in range(busca + 1):
                for cand in (c + d, c - d):
                    if eh_separador(cand):
                        centro = cand
                        break
                if centro is not None:
                    break
            if centro is not None:
                ini = fim = centro
                max_exp = int((comprimento / n) * 0.10)
                while ini - 1 > prev and eh_separador(ini - 1) and (centro - (ini - 1)) <= max_exp:
                    ini -= 1
                while fim + 1 < comprimento and eh_separador(fim + 1) and ((fim + 1) - centro) <= max_exp:
                    fim += 1
        intervalos.append((prev, ini))
        prev = fim + 1 if fim > ini else (fim if fim > c else c)
    intervalos.append((prev, comprimento))

    banners = []
    for (a0, a1) in intervalos:
        if a1 <= a0:
            a1 = a0 + 1
        banners.append(arr[a0:a1, :, :] if eixo == 0 else arr[:, a0:a1, :])
    return banners, ori, cortes
