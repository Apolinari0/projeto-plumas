# Checklist do escopo → onde cada item é atendido

Baseado no escopo da Profa. Raquel Lobosco (Modelagem Computacional).
Marque `[x]` conforme for concluindo.

## Objetivos (seção 1)
- [ ] Detectar a região ocupada pela pluma → `03_yolo_treino`
- [ ] Segmentar a forma e estimar atributos (área, largura, altura, orientação, centroide) → `04_sam_atributos`
- [ ] Acompanhar posição e forma em sequências → `05_tracking`
- [ ] Associar observações a coordenadas, horários e variáveis meteorológicas → `01_dados`, `06_era5`
- [ ] Comparar evolução visual com o vento → `06_era5`
- [ ] Avaliar desempenho e limites em diferentes cenários → `08_avaliacao`

## Perguntas de pesquisa (seção 2): escolher e justificar
- [ ] P1 Detecção em diferentes luz/distância/contraste/fundo
- [ ] P2 Consistência da segmentação com a forma visível
- [ ] P3 Variação de área, orientação e centroide entre imagens consecutivas
- [ ] P4 Direção do deslocamento compatível com o vento do local/horário
- [ ] P5 Super-resolução melhora detecção/segmentação sem introduzir erros?

Perguntas escolhidas pela dupla: _______________________

## Dados e metadados (seção 3)
- [ ] Vídeo com fonte, data/hora, local, câmera, resolução e condições documentados (`01_dados/metadados.json`)
- [ ] Exemplos variados de pluma **e** de estruturas parecidas que não são pluma (nuvem, neblina, vapor)
- [ ] Modelo validado com o vídeo do projeto
- [ ] Horários em UTC, fuso original registrado, incerteza da localização registrada

## Divisão dos dados (seção 4)
- [ ] Treino/val/teste definidos **antes** de ajustar o modelo
- [ ] Divisão por sequência/local/data; sem quadros quase idênticos dos dois lados (`02_anotacao_split`)

## Pré-processamento e super-resolução (seção 5)
- [ ] Cada transformação registrada; cópia dos originais mantida
- [ ] Original e versão ampliada vinculados; comparação com/sem SR nas mesmas imagens
- [ ] Avaliação em dados de teste independentes; inspeção de artefatos nas bordas

## YOLO (seção 6)
- [ ] Versão do modelo e do código, resolução, épocas, sementes, augmentations registradas
- [ ] Imagens com anotações vs. predições examinadas

## Segmentação (seção 7)
- [ ] Máscaras do SAM revisadas (bordas difusas / baixo contraste)
- [ ] Atributos extraídos só de máscaras validadas
- [ ] Sem calibração: resultados em pixels ou proporções da imagem

## Análise temporal (seção 8)
- [ ] Associação entre quadros (distância de centroides / sobreposição / consistência temporal)
- [ ] Períodos sem detecção registrados (sem interpolar intervalos longos)
- [ ] Distinção explícita: (i) deslocamento aparente na imagem, (ii) vento observado/estimado, (iii) trajetória prevista por modelo físico

## ERA5 (seção 9)
- [ ] Produto, variáveis, unidades, nível, resoluções, período e método de extração registrados
- [ ] Horários em referência comum; associação espacial/temporal descrita
- [ ] Convenções, sinais e unidades de u e v conferidos
- [ ] Limites discutidos: grade grossa, perspectiva da câmera, vento de superfície vs. outros níveis, associação ≠ causa

## Reprodutibilidade (seção 10)
- [ ] Registro dos experimentos (`runs.jsonl`): dados, split, transformações, configuração, métricas, figuras, decisões
- [ ] Pesos e previsões com identificador de versão
- [ ] Código repete preparação, inferência e análise com os mesmos parâmetros
