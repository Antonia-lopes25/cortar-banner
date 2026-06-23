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

try:
    from deteccao_cv import bbox_conteudo as _cv_bbox, blocos_cv as _cv_blocos
    _CV_OK = True
except Exception:
    _CV_OK = False


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


def _blocos_por_projecao(arr, eixo, fundo, tol, min_bloco_frac=0.04, min_gap_frac=0.005):
    """
    Encontra blocos de CONTEÚDO ao longo de `eixo`, separando-os por faixas de
    FUNDO. Usa conteúdo FORTE (bem distante do fundo) para que a sombra/penumbra
    — que conecta cartões num mockup — não funda tudo num bloco só.
    """
    a = arr.astype(np.float32)
    dist = np.sqrt(((a - fundo) ** 2).sum(axis=2))
    forte = dist > (tol * 2.0)   # só conteúdo forte conta

    if eixo == 0:
        perfil = forte.mean(axis=1)
    else:
        perfil = forte.mean(axis=0)

    comprimento = perfil.shape[0]
    limiar = 0.10  # 10% da linha/coluna precisa ser conteúdo forte
    dentro = perfil > limiar

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

    min_gap = max(1, int(comprimento * min_gap_frac))
    fundidos = []
    for b in blocos:
        if fundidos and b[0] - fundidos[-1][1] <= min_gap:
            fundidos[-1] = (fundidos[-1][0], b[1])
        else:
            fundidos.append(list(b))
    min_bloco = int(comprimento * min_bloco_frac)
    fundidos = [tuple(b) for b in fundidos if (b[1] - b[0]) >= min_bloco]
    return fundidos


def _cor_fundo(arr, cantos_frac=0.04):
    """Estima a cor de fundo da imagem pela mediana dos quatro cantos."""
    a = arr.astype(np.float32)
    H, W = a.shape[:2]
    cy = max(1, int(H * cantos_frac))
    cx = max(1, int(W * cantos_frac))
    cantos = np.concatenate([
        a[:cy, :cx].reshape(-1, 3), a[:cy, W - cx:].reshape(-1, 3),
        a[H - cy:, :cx].reshape(-1, 3), a[H - cy:, W - cx:].reshape(-1, 3),
    ], axis=0)
    return np.median(cantos, axis=0)


# --------------------------------------------------------------------------
# Auto-trim: remove moldura uniforme (de qualquer cor) ao redor do banner
# --------------------------------------------------------------------------
def aparar_moldura(banner, tol=14.0, cantos_frac=0.06, max_trim_frac=0.45,
                   frac_linha=0.5, margem_seca=2):
    """
    Remove moldura + sombra/penumbra ao redor de UM banner, deixando CORTE SECO.

    Método primário (se OpenCV disponível): bbox do conteúdo via máscara com
    limiar AUTOMÁTICO (Otsu) — adapta-se ao contraste de cada imagem, sem número
    mágico. Fallback (sem OpenCV): perfil de intensidade com limiar absoluto.
    """
    H, W = banner.shape[:2]

    if _CV_OK:
        # o bbox já aplica teto de trim fino (max_trim_frac interno),
        # preservando fundo de design. Aqui só usamos o resultado.
        x0, y0, x1, y1 = _cv_bbox(banner, margem=margem_seca)
        if x1 > x0 and y1 > y0:
            return banner[y0:y1, x0:x1]
        return banner

    # ---- fallback sem OpenCV (limiar absoluto) ----
    a = banner.astype(np.float32)
    H, W = a.shape[:2]

    # cor de fundo robusta: mediana por canal sobre TODOS os pixels de borda —
    # assim um canto destoante (ex.: arroxeado do conteúdo) não desloca a estimativa.
    cy = max(1, int(H * 0.02))
    cx = max(1, int(W * 0.02))
    borda_px = np.concatenate([
        a[:cy, :].reshape(-1, 3),      # faixa topo
        a[H - cy:, :].reshape(-1, 3),  # faixa base
        a[:, :cx].reshape(-1, 3),      # faixa esquerda
        a[:, W - cx:].reshape(-1, 3),  # faixa direita
    ], axis=0)
    fundo = np.median(borda_px, axis=0)

    dist = np.sqrt(((a - fundo) ** 2).sum(axis=2))
    forte = dist > (tol * 2.0)
    if forte.mean() < 0.02:
        return banner

    # perfil de INTENSIDADE média por linha/coluna (capta a rampa da sombra)
    pr_lin = dist.mean(axis=1)
    pr_col = dist.mean(axis=0)

    # limiar absoluto: a borda do conteúdo é onde a intensidade média cruza
    # `lim_corte`. Abaixo disso é fundo ou sombra (penumbra). Validado em
    # imagens reais de mockup. Ajustável se a sombra for muito densa.
    lim_corte = 70.0

    def _borda(perfil):
        n = perfil.shape[0]
        i = 0
        while i < n and perfil[i] < lim_corte:
            i += 1
        j = n - 1
        while j > i and perfil[j] < lim_corte:
            j -= 1
        if i >= j:  # nada cruzou o limiar: não apara
            return 0, n
        return i, j + 1

    y0, y1 = _borda(pr_lin)
    x0, x1 = _borda(pr_col)

    # margem seca extra para garantir zero resíduo de transição
    y0 = min(y0 + margem_seca, H - 1); x0 = min(x0 + margem_seca, W - 1)
    y1 = max(y1 - margem_seca, y0 + 1); x1 = max(x1 - margem_seca, x0 + 1)

    # trava de segurança
    y0 = min(y0, int(H * max_trim_frac)); x0 = min(x0, int(W * max_trim_frac))
    y1 = max(y1, int(H * (1 - max_trim_frac))); x1 = max(x1, int(W * (1 - max_trim_frac)))

    return banner[y0:y1, x0:x1]


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
def cortar(caminho, n=4, orientacao="auto", debug=False, aparar=True):
    img = Image.open(caminho).convert("RGB")
    arr = np.asarray(img)
    H, W = arr.shape[:2]

    fundo = _cor_fundo(arr)
    # tolerância de fundo: relativa ao contraste da imagem
    tol_fundo = 24.0

    def _proj(eixo):
        return _blocos_por_projecao(arr, eixo, fundo, tol_fundo)

    # ---- Método primário A: blocos via OpenCV (limiar automático Otsu) ----
    escolha = None  # (eixo, blocos)
    if _CV_OK:
        if orientacao in ("auto", "vertical"):
            bv = _cv_blocos(arr, 0)
            if debug:
                print(f"  [debug] opencv vertical: {len(bv)} blocos")
            if len(bv) == n:
                escolha = (0, bv)
        if escolha is None and orientacao in ("auto", "horizontal"):
            bh = _cv_blocos(arr, 1)
            if debug:
                print(f"  [debug] opencv horizontal: {len(bh)} blocos")
            if len(bh) == n:
                escolha = (1, bh)

    # ---- Método primário B: projeção por limiar (cobre corte seco sem fundo) ----
    if escolha is None and orientacao in ("auto", "vertical"):
        bv = _proj(0)
        if debug:
            print(f"  [debug] projeção vertical: {len(bv)} blocos -> {bv}")
        if len(bv) == n:
            escolha = (0, bv)
    if escolha is None and orientacao in ("auto", "horizontal"):
        bh = _proj(1)
        if debug:
            print(f"  [debug] projeção horizontal: {len(bh)} blocos -> {bh}")
        if len(bh) == n:
            escolha = (1, bh)

    if escolha is not None:
        eixo, blocos = escolha
        banners = []
        for (a0, a1) in blocos:
            if eixo == 0:
                banners.append(arr[a0:a1, :, :])
            else:
                banners.append(arr[:, a0:a1, :])
        if aparar:
            banners = [aparar_moldura(b) for b in banners]
        ori_txt = "vertical" if eixo == 0 else "horizontal"
        if debug:
            print(f"  [debug] método: projeção | orientação: {ori_txt}")
        return banners, ori_txt, [b[0] for b in blocos[1:]]

    # ---- Fallback: detecção de fronteiras (corte seco sem moldura clara) ----
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
    else:
        fv, qv = _processa(0)
        fh, qh = _processa(1)
        if H >= W:
            qv += 0.05
        else:
            qh += 0.05
        if debug:
            print(f"  [debug] fallback fronteiras — qualidade v={qv:.3f} h={qh:.3f}")
        eixo, fronteiras = (0, fv) if qv >= qh else (1, fh)

    banners = _recortar(arr, eixo, fronteiras, descartar_faixa=True)
    if aparar:
        banners = [aparar_moldura(b) for b in banners]
    ori_txt = "vertical" if eixo == 0 else "horizontal"
    if debug:
        print(f"  [debug] método: fronteiras (fallback) | orientação: {ori_txt}")
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
    ap.add_argument("--sem-aparar", action="store_true",
                    help="não remove a moldura/fundo ao redor de cada banner")
    args = ap.parse_args()

    if not os.path.isfile(args.entrada):
        print(f"Arquivo não encontrado: {args.entrada}", file=sys.stderr)
        sys.exit(1)

    banners, ori, fronteiras = cortar(args.entrada, n=args.n,
                                      orientacao=args.orientacao, debug=args.debug,
                                      aparar=not args.sem_aparar)

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
