#!/usr/bin/env python3
"""
cortar_banners.py
=================

Corta uma imagem composta por EXATAMENTE 4 banners em 4 arquivos,
detectando a posição REAL de cada banner — independente de:

  - orientação:  banners empilhados (vertical) OU lado a lado (horizontal);
  - separação:   faixa clara/branca, faixa escura, faixa colorida, ou
                 corte seco (banners colados, sem faixa nenhuma).

Como N é fixo em 4, o script procura as 3 fronteiras que melhor dividem a
imagem, validando que resultam em 4 blocos coerentes. Não depende da cor da
faixa nem da posição — trabalha pela DESCONTINUIDADE visual entre faixas de
pixels vizinhas e pela UNIFORMIDADE local (assinatura de uma divisória lisa).

Uso:
    python3 cortar_banners.py entrada.png
    python3 cortar_banners.py entrada.png --saida ./banners --w 784 --h 1344
    python3 cortar_banners.py entrada.png --orientacao auto --fmt jpg

Parâmetros:
    --n            número de banners (padrão 4)
    --orientacao   auto | vertical | horizontal   (padrão auto)
    --saida        pasta de saída (padrão ./banners_out)
    --w --h        redimensiona cada banner final p/ esse tamanho (opcional)
    --fmt          png | jpg   (padrão png)
    --debug        imprime os candidatos de fronteira encontrados
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------
# Núcleo: perfil de corte ao longo de um eixo
# --------------------------------------------------------------------------
def _perfil_fronteiras(arr, eixo):
    """
    Constrói, para cada posição ao longo de `eixo`, um score de "quão boa
    fronteira esta linha/coluna é". Combina dois sinais:

      1) DESCONTINUIDADE: diferença média entre a faixa de pixels imediatamente
         antes e depois desta posição. Fronteiras entre dois banners diferentes
         têm conteúdo distinto dos dois lados -> diferença alta.

      2) UNIFORMIDADE: o quão lisa (baixa variância ao longo da própria faixa)
         é a linha/coluna. Uma divisória (faixa de cor sólida) é muito uniforme.

    eixo=0 -> fronteiras horizontais (corta empilhamento vertical): varremos
              linha por linha (cada y).
    eixo=1 -> fronteiras verticais (corta layout horizontal): varremos coluna
              por coluna (cada x).

    Retorna (descont, unif) já normalizados em [0,1].
    """
    a = arr.astype(np.float32)

    if eixo == 0:
        # média de cor por linha -> (H, C)
        linhas = a.mean(axis=1)
        # variância ao longo da largura, por linha -> uniformidade (baixa = liso)
        var_local = a.var(axis=1).mean(axis=1)  # (H,)
    else:
        # média de cor por coluna -> (W, C)
        linhas = a.mean(axis=0)
        var_local = a.var(axis=0).mean(axis=1)  # (W,)

    # descontinuidade: distância entre faixa atual e a anterior
    dif = np.zeros(linhas.shape[0], dtype=np.float32)
    dif[1:] = np.sqrt(((linhas[1:] - linhas[:-1]) ** 2).sum(axis=1))

    def _norm(v):
        v = v - v.min()
        m = v.max()
        return v / m if m > 1e-6 else v

    descont = _norm(dif)
    # uniformidade alta = variância baixa -> invertemos
    unif = 1.0 - _norm(var_local)
    return descont, unif


def _candidatos(arr, eixo, n_banners, margem_frac=0.12, debug=False):
    """
    Encontra as (n_banners - 1) fronteiras ao longo de `eixo`.

    Score de cada posição = mistura de descontinuidade e uniformidade.
    A divisória ideal é tanto um ponto de mudança de conteúdo quanto uma faixa
    lisa. Mas em corte seco NÃO há faixa lisa — então a descontinuidade sozinha
    já carrega o sinal. Por isso somamos os dois (peso na descontinuidade) em
    vez de exigir os dois.

    Restrições:
      - fronteiras não podem cair perto demais das bordas (margem_frac);
      - duas fronteiras não podem ficar coladas (distância mínima entre elas),
        garantindo 4 blocos de tamanho razoável.
    """
    comprimento = arr.shape[0] if eixo == 0 else arr.shape[1]
    descont, unif = _perfil_fronteiras(arr, eixo)

    # score combinado: descontinuidade domina, uniformidade reforça
    score = 0.70 * descont + 0.30 * unif

    # zona proibida nas bordas
    margem = int(comprimento * margem_frac)
    score[:margem] = -1
    score[comprimento - margem:] = -1

    # distância mínima entre fronteiras: ~ metade do tamanho ideal de um bloco
    passo_ideal = comprimento / n_banners
    dist_min = int(passo_ideal * 0.55)

    # seleção gulosa com supressão de vizinhança
    escolhidas = []
    s = score.copy()
    for _ in range(n_banners - 1):
        idx = int(np.argmax(s))
        if s[idx] < 0:
            break
        escolhidas.append(idx)
        lo = max(0, idx - dist_min)
        hi = min(comprimento, idx + dist_min)
        s[lo:hi] = -1

    escolhidas.sort()

    if debug:
        ori = "horizontais (eixo y)" if eixo == 0 else "verticais (eixo x)"
        print(f"  [debug] fronteiras {ori}: {escolhidas} "
              f"(comprimento={comprimento}, passo_ideal={passo_ideal:.0f})")

    return escolhidas, score


def _qualidade(escolhidas, comprimento, n_banners):
    """
    Mede o quão 'equilibrado' ficou o conjunto de fronteiras, para decidir
    a orientação no modo auto. Penaliza blocos muito desiguais (o layout real
    tem 4 banners de tamanho parecido). Retorna score: maior = melhor.
    """
    if len(escolhidas) != n_banners - 1:
        return -1.0
    cortes = [0] + list(escolhidas) + [comprimento]
    tamanhos = np.diff(cortes).astype(np.float32)
    ideal = comprimento / n_banners
    # desvio relativo médio em relação ao tamanho ideal
    desvio = np.abs(tamanhos - ideal).mean() / ideal
    return 1.0 - desvio


# --------------------------------------------------------------------------
# Detecção de faixa lisa em torno da fronteira (para descartar a calha)
# --------------------------------------------------------------------------
def _expandir_faixa(arr, eixo, pos, tol_var=18.0, max_frac=0.06):
    """
    Dada uma fronteira em `pos`, verifica se ela está no meio de uma faixa lisa
    (divisória de cor sólida) e, em caso afirmativo, devolve (ini, fim) da faixa
    a ser DESCARTADA. Se for corte seco (sem faixa lisa), devolve (pos, pos) —
    nada a descartar, corte exato na posição.
    """
    comprimento = arr.shape[0] if eixo == 0 else arr.shape[1]
    max_exp = int(comprimento * max_frac)

    def _linha_lisa(i):
        if not (0 <= i < comprimento):
            return False
        if eixo == 0:
            faixa = arr[i, :, :].astype(np.float32)
        else:
            faixa = arr[:, i, :].astype(np.float32)
        return faixa.var(axis=0).mean() <= tol_var

    # A fronteira detectada pode cair no início, meio ou fim da calha.
    # Procuramos uma linha lisa numa pequena janela ao redor de `pos` para
    # ancorar a faixa; se nenhuma for lisa, é corte seco.
    janela = max(3, int(comprimento * 0.01))
    ancora = None
    for d in range(janela + 1):
        for cand in (pos + d, pos - d):
            if _linha_lisa(cand):
                ancora = cand
                break
        if ancora is not None:
            break

    if ancora is None:
        return pos, pos  # corte seco — nada a descartar

    ini = fim = ancora
    while _linha_lisa(ini - 1) and (ancora - (ini - 1)) <= max_exp:
        ini -= 1
    while _linha_lisa(fim + 1) and ((fim + 1) - ancora) <= max_exp:
        fim += 1
    return ini, fim


# --------------------------------------------------------------------------
# Recorte
# --------------------------------------------------------------------------
def _recortar(arr, eixo, fronteiras, descartar_faixa=True):
    """
    Recorta a imagem em blocos a partir das fronteiras, descartando faixas
    lisas (calhas) quando existirem. Retorna lista de arrays (banners).
    """
    comprimento = arr.shape[0] if eixo == 0 else arr.shape[1]
    limites = [0]
    descartes = []
    for p in fronteiras:
        if descartar_faixa:
            ini, fim = _expandir_faixa(arr, eixo, p)
        else:
            ini, fim = p, p
        descartes.append((ini, fim))

    # constrói os intervalos de cada banner
    banners = []
    cortes = []
    prev_fim = 0
    for (ini, fim) in descartes:
        cortes.append((prev_fim, ini))   # banner: de prev_fim até início da faixa
        prev_fim = fim if fim == ini else fim + 1  # próximo começa após a faixa
    cortes.append((prev_fim, comprimento))

    for (a0, a1) in cortes:
        if a1 <= a0:
            a1 = a0 + 1
        if eixo == 0:
            banners.append(arr[a0:a1, :, :])
        else:
            banners.append(arr[:, a0:a1, :])
    return banners


# --------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------
def cortar(caminho, n=4, orientacao="auto", debug=False):
    img = Image.open(caminho).convert("RGB")
    arr = np.asarray(img)
    H, W = arr.shape[:2]

    def _processa(eixo):
        comprimento = H if eixo == 0 else W
        fronteiras, _ = _candidatos(arr, eixo, n, debug=debug)
        q = _qualidade(fronteiras, comprimento, n)
        return fronteiras, q

    if orientacao == "vertical":
        eixo = 0
        fronteiras, _ = _candidatos(arr, eixo, n, debug=debug)
    elif orientacao == "horizontal":
        eixo = 1
        fronteiras, _ = _candidatos(arr, eixo, n, debug=debug)
    else:  # auto: testa os dois e escolhe o mais equilibrado,
           # com leve viés pela proporção (imagem alta -> provavelmente vertical)
        fv, qv = _processa(0)
        fh, qh = _processa(1)
        # viés suave pela forma da imagem
        if H >= W:
            qv += 0.05
        else:
            qh += 0.05
        if debug:
            print(f"  [debug] qualidade vertical={qv:.3f} | horizontal={qh:.3f}")
        if qv >= qh:
            eixo, fronteiras = 0, fv
        else:
            eixo, fronteiras = 1, fh

    banners = _recortar(arr, eixo, fronteiras, descartar_faixa=True)
    ori_txt = "vertical" if eixo == 0 else "horizontal"
    return banners, ori_txt, fronteiras


def main():
    ap = argparse.ArgumentParser(description="Corta uma imagem em 4 banners pela posição real.")
    ap.add_argument("entrada")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--orientacao", choices=["auto", "vertical", "horizontal"], default="auto")
    ap.add_argument("--saida", default="./banners_out")
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--fmt", choices=["png", "jpg"], default="png")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.entrada):
        print(f"Arquivo não encontrado: {args.entrada}", file=sys.stderr)
        sys.exit(1)

    banners, ori, fronteiras = cortar(args.entrada, n=args.n,
                                      orientacao=args.orientacao, debug=args.debug)

    os.makedirs(args.saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.entrada))[0]

    print(f"Orientação detectada: {ori} | fronteiras em: {fronteiras}")
    for i, b in enumerate(banners, 1):
        im = Image.fromarray(b)
        if args.w and args.h:
            im = im.resize((args.w, args.h), Image.LANCZOS)
        nome = f"{base}_banner{i}.{args.fmt}"
        destino = os.path.join(args.saida, nome)
        if args.fmt == "jpg":
            im.save(destino, quality=95)
        else:
            im.save(destino)
        print(f"  salvo: {destino}  ({im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    main()
