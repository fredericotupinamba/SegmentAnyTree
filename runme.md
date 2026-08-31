# TupiSAT — guia de execução

Segmentação de árvores individuais em nuvens de pontos TLS/MLS, separação
madeira/folha, e cálculo de variáveis dendrométricas por árvore.

Este documento cobre **o que rodar, em que ordem, e o que cada etapa faz**.
Se você só quer o resultado padrão, leia a seção 1 e pare. Se precisa da
base de copa precisa (a diferença é grande — veja a seção 3), leia até a 4.

---

## 0. O que o pipeline faz

```
  nuvem .laz bruta
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ ETAPA 1  —  TupiSAT (automática, 1 comando) │
  │   segmentação semântica  → PredSemantic     │
  │   segmentação instância  → PredInstance     │
  │   métricas florestais    → CSVs + LAZ       │
  └─────────────────────────────────────────────┘
        │
        ▼  <nome>_SAT_output/<nome>.laz
        │
  ┌─────────────────────────────────────────────┐
  │ ETAPA 2  —  PointsToWood  (MANUAL, hoje)    │
  │   madeira vs folha       → prediction, pwood│
  └─────────────────────────────────────────────┘
        │
        ▼  <nome>_pwood.laz
        │
  ┌─────────────────────────────────────────────┐
  │ ETAPA 3  —  métricas de novo (MANUAL)       │
  │   agora com a regra de copa por madeira     │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ ETAPA 4  —  relatórios visuais (opcional)   │
  └─────────────────────────────────────────────┘
```

> **Importante:** as etapas 2 e 3 **ainda não estão dentro do container do
> TupiSAT**. O orquestrador ([`tupisat_inference/batch_orchestrator.py`](tupisat_inference/batch_orchestrator.py))
> tem 9 passos e nenhum deles chama o PointsToWood. Integrar isso numa
> imagem única é trabalho pendente — os dois projetos usam Python e PyTorch
> incompatíveis (3.8 + torch 1.9 contra 3.11 + torch 2.8), então precisam
> de ambientes separados. Enquanto isso, as etapas 2 e 3 são comandos
> manuais, documentados abaixo.

---

## 0.1 Atalho — rodar tudo de uma vez

As três etapas para uma pasta inteira, num comando:

```powershell
cd E:\GITHUB\SegmentAnyTree
.\run_all_stages.ps1
```

| flag | efeito |
|---|---|
| *(nenhuma)* | pula o que já terminou — seguro reexecutar após interrupção |
| `-Force` | refaz tudo |
| `-SkipStage1` | segmentação já feita, só madeira/folha + métricas |
| `-SkipStage4` | não gera as páginas por árvore (economiza horas e ~1 GB) |
| `-Lang en\|pt\|es` | idioma das páginas da etapa 4 (padrão `en`) |
| `-Only P01,P02` | apenas as parcelas cujo nome contenha esses trechos |
| `-InputDir`, `-SatOut`, `-PwoodDir`, `-MetricsDir` | caminhos alternativos |

O script **recusa a rodar** se a imagem do TupiSAT não tiver o código atual
(veja a seção 1), em vez de devolver resultados que parecem certos mas usam
a regra de copa antiga. Uma parcela que falha não derruba as outras: o erro
é registrado e a fila continua.

**Tempo para 16 parcelas de 16 m numa RTX 4060 Ti:** ~6 h na etapa 1,
~2 h na etapa 2, minutos na etapa 3, e ~4 h na etapa 4 se você gerar a
página de **todas** as árvores (~30 por parcela, ~30 s cada) — cerca de
**12 horas no total**, ou 8 h com `-SkipStage4`. As páginas ocupam ~1,6 MB
cada, algo como 800 MB no fim.

As seções abaixo explicam cada etapa individualmente, para quando você
precisar rodar uma só, entender o que saiu, ou depurar.

---

## 1. Etapa 1 — rodar o TupiSAT

### A imagem

```powershell
docker pull fredericotupinamba/tupisat:latest
```

`tupisat:latest` (local) e `fredericotupinamba/tupisat:latest` (publicada)
são o mesmo conteúdo; os comandos deste guia usam o nome local.

A etapa 1 roda o código **assado dentro da imagem**, não o do seu disco.
Se você alterar qualquer coisa em `tupisat_inference/`, reconstrua antes de
rodar a etapa 1:

```powershell
docker build -f Dockerfile.pandas-fix -t tupisat:latest .
```

Para conferir se uma imagem já tem as mudanças de base de copa e de
diâmetro (deve responder `2`):

```powershell
docker run --rm --entrypoint bash tupisat:latest -c `
  "grep -c crown_wood_frac_threshold /home/nibio/mutable-outside-world/tupisat_inference/forest_metrics/config.py"
```

> As etapas 3 e 4 **não** precisam de reconstrução: elas montam o
> repositório por cima (`-v ...:/w -w /w`), então sempre executam o código
> do disco e a imagem só fornece o Python e as dependências. É por isso que
> dá para iterar no código sem reconstruir nada — mas só nessas etapas.

### Rodar

Coloque os `.laz`/`.las`/`.ply` numa pasta de entrada e rode:

```powershell
docker run -d --gpus all `
    --name tupisat `
    --mount type=bind,source="E:\GITHUB\SegmentAnyTree\data\03-Clipped16m",target=/home/nibio/mutable-outside-world/bucket_in_folder `
    --mount type=bind,source="E:\GITHUB\SegmentAnyTree\data\04-OUTPUT",target=/home/nibio/mutable-outside-world/bucket_out_folder `
    tupisat:latest --force
```

Acompanhe com `docker logs -f tupisat`.

| flag | efeito |
|---|---|
| *(nenhuma)* | processa só o que ainda não terminou (retomável) |
| `--force` | reprocessa tudo, ignorando o estado anterior |
| `--skip-forest-metrics` | só segmenta — use quando o PointsToWood vem depois |
| `--stop-on-error` | aborta o lote na primeira falha em vez de seguir |
| `--max-attempts N` | tentativas antes de marcar o arquivo como falha permanente |

**Resultado:** uma pasta `<nome>_SAT_output/` por nuvem de entrada, com a
nuvem segmentada `<nome>.laz` (`PredSemantic`, `PredInstance`).

Sem `--skip-forest-metrics`, a etapa 1 também calcula as métricas
(`_tree_metrics.csv`, `_taper.csv`, `_plot_summary`, e os `.laz` de
visualização) e grava `IsCrown` na nuvem. **Isso só faz sentido se você
não for rodar o PointsToWood.** Se for, essas métricas usam a regra de
copa antiga, são refeitas pela etapa 3, e ficam em `04-OUTPUT` parecidas
com as boas de `06-METRICS` — foi por isso que o `run_all_stages.ps1`
passa `--skip-forest-metrics`.

Os 9 passos internos, na ordem, aparecem no log como `step=...`:
`fix_naming` → `utm2local` → `prepare_eval_config` → `clear_cache` →
`inference` → `rename_results` → `merge` → `forest_metrics` → `finalize`.

O estado fica em `bucket_out_folder/.sat_state/`, por isso o pipeline é
retomável: reiniciar o container só reprocessa o que não terminou.

**Tempo:** ~22–25 min por parcela de 16 m (≈34 M pontos) numa RTX 4060 Ti.

---

## 2. Etapa 2 — PointsToWood (madeira vs folha)

### Por que fazer isso

A etapa 1 já entrega uma base de copa, mas ela é estimada por intensidade
LiDAR e área de projeção, sobre **todos** os pontos. Com o rótulo
madeira/folha o erro cai pela metade:

| método | erro (validação cruzada entre parcelas) |
|---|---|
| densidade de pontos *(versão antiga)* | 3,37 m |
| intensidade + área *(etapa 1 sozinha)* | 1,09 m |
| **fração de madeira** *(com esta etapa)* | **0,71 m** |

Isso importa porque a base de copa define a **altura comercial**, que trunca
a curva de afilamento e portanto muda o volume de fuste de cada árvore.

### Como rodar

Requer o repositório [PointsToWood](https://github.com/philwilkes/pointstowood)
e sua imagem. A flag `--no-ptw-output` usada abaixo é uma adição local — se
o comando reclamar dela, reconstrua aquela imagem primeiro (`docker compose
build` no repositório do PointsToWood). Sem ela o pipeline também grava um
`.ply` de ~1 GB por parcela cujo conteúdo já está no `.laz` de saída.

```powershell
docker run -d --gpus all --name ptw `
  -v E:\GITHUB\SegmentAnyTree\data:/app/sat_data `
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True `
  --entrypoint python `
  pointstowood:latest preinstance_pipeline.py `
    /app/sat_data/04-OUTPUT/<nome>_SAT_output/<nome>.laz `
    --preinstance-field PredInstance --region eu `
    --memory-fraction 0.45 --tta 1 --no-ptw-output `
    --output /app/sat_data/05-PWOOD/<nome>_pwood.laz
```

### As três flags que não são opcionais

Estas foram determinadas experimentalmente e **sem elas o processo falha
em silêncio** numa placa de 16 GB:

**`-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — o
`point_budget_collate` monta lotes de tamanho variável, o que fragmenta o
alocador de cache do PyTorch. A memória *reservada* sobe até ~15,9 GB de
16,4 GB enquanto a *alocada* fica estável, e aí o WDDM do Windows começa a
derramar para a RAM em vez de dar OOM. O processo **não levanta exceção,
não sai do ar, e o container continua "Up"** — a GPU marca 91–100% de
utilização puxando só ~40 W. Com a flag, a VRAM fica em ~10 GB e estável.

**`--tta 1`** — com a flag acima mas TTA 2×, o segundo passe morre com
`SIGSEGV` puro, sem log de erro CUDA, por volta de 80%. Um passe só evita
esse estado e corta o tempo pela metade. **O modelo de base de copa foi
calibrado sobre saída de passe único**, então mudar isso exige recalibrar.

**`--memory-fraction 0.45`** — folga adicional.

**Como saber se está saudável:** utilização ~65–80% a ~57–70 W. Utilização
alta com potência baixa (~40 W) significa *thrashing*, não trabalho.

**Tempo:** 7,1 min por parcela, pico de 7,2 GB de GPU.

**Resultado:** `<nome>_pwood.laz` — a nuvem original completa, em
coordenadas originais, com duas colunas novas: `prediction` (0 = folha,
1 = madeira) e `pwood` (probabilidade 0–1).

> `prediction = 1` é **madeira**, ou seja tronco **e galhos**. Não é
> "tronco". Separar fuste de galho é feito depois, geometricamente.

---

## 3. Etapa 3 — recalcular as métricas com o rótulo de madeira

```powershell
docker run --rm `
  -v E:\GITHUB\SegmentAnyTree:/w -w /w -e PYTHONPATH=/w `
  --entrypoint python3.8 tupisat:latest `
  tupisat_inference/forest_metrics/forest_metrics.py `
    --input-las data/05-PWOOD/<nome>_pwood.laz `
    --output-dir data/06-METRICS `
    --stem <nome> --verbose
```

O log diz qual regra foi usada:

```
Found PointsToWood wood/leaf labels -- using the wood-fraction crown rule
```
ou, se a coluna `prediction` não existir:
```
No 'prediction' column -- falling back to the intensity+area crown rule
```

**O fallback é automático.** Nuvens antigas, sem PointsToWood, continuam
funcionando — só com a precisão menor da tabela da seção 2.

> ⚠️ Este comando **modifica o arquivo de entrada**: ele grava o campo
> `IsCrown` de volta no `_pwood.laz`, no lugar (por escolha de projeto, para
> não duplicar um arquivo de centenas de MB).

---

## 4. Etapa 4 — relatórios visuais de validação

Uma página por árvore, em alta resolução, para conferir a medição contra os
pontos que a geraram — em vez de confiar no CSV.

```powershell
docker run --rm `
  -v E:\GITHUB\SegmentAnyTree:/w -w /w -e PYTHONPATH=/w -e MPLCONFIGDIR=/tmp/mpl `
  --entrypoint python3.8 tupisat:latest `
  tupisat_inference/forest_metrics/tree_report.py `
    --input-las data/05-PWOOD/<nome>_pwood.laz `
    --metrics-dir data/06-METRICS --stem <nome> `
    --tree-ids 33 28 2 15 `
    --output-dir data/07-RELATORIO --dpi 200 --lang en
```

Para a parcela inteira, troque `--tree-ids ...` por `--all-trees`. A nuvem
é lida e o DTM reconstruído **uma vez por parcela**, não por árvore, então
30 páginas custam muito menos que 30 execuções separadas.

| flag | efeito |
|---|---|
| `--tree-ids` | lista de árvores (os `tree_id` do `tree_metrics.csv`) |
| `--all-trees` | todas as árvores da parcela, em vez de uma lista |
| `--lang` | `en` (padrão), `pt`, `es` |
| `--dpi` | 200 gera ~3200×4200 px; 300 para impressão |

Cada página traz quatro blocos:

1. **12 seções de diâmetro** — os pontos reais de cada fatia com o círculo
   ajustado sobreposto. A fatia é reconstruída com a mesma regra do
   algoritmo (`|hag − h| ≤ espessura/2`), então o que você vê é o que foi
   medido, não uma aproximação.
2. **Perfil de afilamento** — diâmetro medido e corrigido contra a altura.
3. **Painel numérico** — sortimentos, volumes, qualidade, tortuosidade,
   conicidade.
4. **Copa em 3D** — a árvore inteira e a copa isolada, com a malha de
   voxels cujo volume *é* o `crown_volume_m3`. Só as faces externas são
   desenhadas, então dá para ver o envelope da copa e não um bloco opaco.

O DTM é reconstruído aqui, não aproximado pelo `z_base`: toda altura na
página é altura acima do solo, e um atalho deslocaria as seções em relação
às que foram medidas.

---

## 5. Entendendo os resultados

### `tree_metrics.csv` — uma linha por árvore

| coluna | significado |
|---|---|
| `tree_id` | id da instância (= `PredInstance` na nuvem) |
| `height_m` | altura total (percentil 99,5 da altura acima do solo) |
| `dbh_cm`, `dbh_cci` | DAP e sua cobertura angular (0–1; 1 = anel completo) |
| `crown_base_height_m` | base de copa = **altura comercial** |
| `live_crown_ratio` | proporção da altura ocupada por copa |
| `crown_volume_m3` | volume da copa por contagem de voxels de 0,25 m |
| `stem_volume_taper_m3` | volume de fuste integrando a curva de afilamento |
| `stem_volume_conic_m3` | referência cônica independente (para comparação) |
| `volume_sawlog_m3`, `volume_pulpwood_m3` | sortimentos |
| `stem_lean_deg` | inclinação do eixo do fuste |
| `quality_flags` | sinalizações de qualidade, vazio quando não há |

### `taper.csv` — uma linha por seção

| coluna | significado |
|---|---|
| `height_m` | altura da seção acima do solo |
| `diameter_cm` | diâmetro medido |
| `diameter_corrected_cm` | após a correção monotônica (usado no volume) |
| `cci` | cobertura angular do anel (0–1) |
| `center_x`, `center_y` | centro do círculo ajustado |
| `fit_source` | **como esta seção foi medida** — veja abaixo |
| `axis_residual_m` | distância do centro ao eixo do fuste |
| `tilt_outlier_prob` | score de inclinação anômala |

O `fit_source` é a coluna para auditar qualquer número:

| valor | significa |
|---|---|
| `slice` | ajuste original, não tocado |
| `axis` | reajustado com restrição ao eixo do fuste |
| `cylinder` | reajustado por janela vertical (camada 3) |
| `rejected_off_axis` | descartado: centro fora do fuste, ou reajuste implausível |
| `rejected_tilt` | descartado pelo detector de inclinação |

---

## 6. Como a base de copa é encontrada

Duas regras, escolhidas automaticamente conforme a nuvem tenha ou não o
rótulo madeira/folha.

**Com `prediction` (preferida).** Varre faixas de 0,25 m de baixo para
cima e marca a base de copa na primeira corrida de **6 faixas consecutivas
(1,5 m)** em que menos de **50%** dos pontos são madeira, somando **+1,00 m**.

Por que funciona: sem folhas, um fuste limpo é ~100% madeira e uma faixa de
copa é ~0% (mediana 0,999 contra 0,000, AUC 0,990).

Por que só uma variável: acrescentar área ou espalhamento da madeira
**piorou** o resultado (1,06 e 1,08 m) e derrubou a cobertura. Na copa quase
não sobra madeira para ter área ou espalhamento — a madeira não se abre
acima da base de copa, ela **some**. Uma regressão logística com três
variáveis empatou (0,75 m) sem superar.

Por que o deslocamento de +1,00 m: a transição é gradual (~3,1 m de largura
por árvore), e o ponto onde a fração de madeira cruza o limiar fica
sistematicamente ~1 m **abaixo** de onde um anotador marca a base. É viés
fixo, ajustado só nas árvores de treino.

**Sem `prediction` (fallback).** Intensidade LiDAR mediana abaixo de 40% da
linha de base do próprio fuste **e** área de projeção acima de 2,5× dela.

Os parâmetros das duas regras estão em
[`config.py`](tupisat_inference/forest_metrics/config.py), documentados com
os números da calibração.

---

## 7. Como os diâmetros são medidos

Um ajuste de círculo irrestrito enxerga todos os pontos da fatia, então um
galho ao lado do fuste pode capturar o ajuste — e o círculo inflado fica
*redondo* justamente porque abraça os dois, passando nos testes de
qualidade. Medido nas parcelas de calibração: **8,6% das seções** (232 de
2688), em **68% das árvores**.

Três camadas corrigem isso:

**Camada 1 — eixo do fuste.** Regressão robusta (Theil-Sen) pelos centros
das seções. **Não se assume eixo vertical**: inclinar é a norma, não a
exceção (mediana 3,0°, p90 5,2°, máximo 13,4°; 63% das árvores passam de
2°). Um teste contra a vertical reprovaria os ajustes **bons** das árvores
tortas.

**Camada 2 — reajuste restrito.** Seções fora do eixo, ou gordas demais
para a vizinhança, são remedidas usando só os pontos dentro de uma janela
em volta de onde o eixo diz que o fuste está. O galho fica fora da janela e
nunca chega ao ajustador.

**Camada 3 — coerência cilíndrica.** O que a camada 2 não resolve vira uma
janela vertical de 0,6 m, com os pontos "desinclinados" ao longo do eixo. O
ajuste só é aceito se os pontos que sustentam o anel ocuparem a maior parte
da altura da janela — um fuste ocupa, um galho atravessando não.

### Três salvaguardas (sem elas isto faz mal)

Na primeira versão, o reajuste derrubou o DAP de uma árvore de 36,4 para
**12,1 cm** — trocou "diâmetro inflado" por "diâmetro confiantemente errado
para menos", que é pior porque não parece errado. Causa: fustes têm
curvatura basal, uma reta é modelo ruim nos primeiros metros, e a janela
ficava centrada 27 cm fora do fuste, capturando só a borda.

1. **Banda de plausibilidade** — o reajuste só é aceito entre 0,70 e 1,40×
   do raio esperado. Fora disso a janela cortou o fuste. Falhar para "sem
   medição" é seguro; a correção monotônica interpola.
2. **Gate de confiança** — se menos de 50% das seções medidas estão sobre o
   eixo, a árvore não tem eixo confiável e nada é corrigido.
3. **DAP só é resgatado, nunca sobrescrito** — a altura do peito é a fatia
   mais bem escaneada; ali os pontos valem mais que o eixo.

**Efeito medido (P01+P02, 59 árvores):** DAP com variação mediana de
**+0,00%** e nenhuma árvore fora de ±5%; seções medidas de 2688 para 2984;
e a dispersão do volume por afilamento contra a referência cônica caindo de
0,37 para 0,27 em P01. Correção, não distorção.

---

## 8. Recalibrar o modelo de base de copa

Só é necessário se você mudar de sítio, de sensor, ou voltar para TTA 2×.

```powershell
# 1. features por faixa de altura, a partir dos .laz com prediction
docker run --rm -v E:\GITHUB\SegmentAnyTree:/w -w /w -e PYTHONPATH=/w `
  --entrypoint python3.8 tupisat:latest `
  tupisat_inference/forest_metrics/calibration/extract_cbh_features.py `
    --pwood P01=data/05-PWOOD/P01_pwood.laz `
    --pwood P02=data/05-PWOOD/P02_pwood.laz `
    --output tupisat_inference/forest_metrics/calibration/cbh_bin_features_wood.csv

# 2. ajuste + validação cruzada + escolha do modelo
docker run --rm -v E:\GITHUB\SegmentAnyTree:/w -w /w -e PYTHONPATH=/w `
  --entrypoint python3.8 tupisat:latest `
  tupisat_inference/forest_metrics/calibration/calibrate_cbh.py
```

O segundo comando compara 7 regras sobre madeira/folha e 2 modelos
logísticos contra o método em produção, **todos sob o mesmo protocolo**, e
escreve o vencedor em
`cbh_model.json`. Ele imprime o ajuste completo e o de validação cruzada
lado a lado — **cite sempre o de validação cruzada**: o ajuste completo
afina e pontua nas mesmas árvores, e é otimista por construção.

Os rótulos ficam em
[`calibration/cbh_tree_labels.csv`](tupisat_inference/forest_metrics/calibration/cbh_tree_labels.csv):
58 alturas de base de copa marcadas visualmente em P01/P02. **São
insubstituíveis sem re-anotar à mão** — o `.gitignore` tem uma exceção
explícita para eles não caírem na regra `*.csv`.

---

## 9. Testes

```powershell
docker run --rm -v E:\GITHUB\SegmentAnyTree:/w -w /w -e PYTHONPATH=/w `
  --entrypoint python3.8 tupisat:latest -m pytest tests/ -q
```

51 testes.

---

## 10. Limitações conhecidas

- **A etapa 2 não está integrada** ao container do TupiSAT. Um lote grande
  exige rodar as etapas 2 e 3 manualmente por parcela.
- **A calibração é estreita:** 58 árvores, 2 parcelas, uma composição de
  espécies, um scanner, uma data. A validação cruzada tem só 2 dobras — é o
  melhor disponível, mas é evidência fraca. Para outro sítio ou outro
  equipamento, revalide antes de confiar nos números.
- **Falha silenciosa do PointsToWood** (seção 2): sem as flags certas o
  processo trava sem erro, e o orquestrador registraria "running"
  indefinidamente. Se integrar a etapa 2 num lote automático, monitore
  *ausência de progresso*, não exceções — dois dos três modos de falha não
  levantam nada.
- **Fustes muito ocluídos** continuam sem medição em parte das seções. Isso
  é honesto, não bug: quando só há um arco parcial de pontos, nenhum
  diâmetro é confiável, e o pipeline prefere não reportar a chutar.
