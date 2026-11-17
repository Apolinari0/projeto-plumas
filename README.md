# Detecção e análise de plumas em vídeo

Projeto final de Modelagem Computacional (UNIFESP, Profa. Raquel Lobosco).
Pipeline: **vídeo → YOLO (detecta) → SAM (segmenta) → atributos → tracking → ERA5 (vento) → avaliação**.

Código no GitHub, execução no **Google Colab**, dados e pesos no **Google Drive**.

## Estrutura

```
config.yaml             parâmetros, sementes e caminhos (único lugar para ajustar)
src/plumas/             código compartilhado e testado (config, video, split, features, tracking, wind)
tests/                  testes dos módulos de src/
docs/CHECKLIST_ESCOPO.md  itens do escopo → onde são atendidos
01_dados/               vídeo, metadados, extração de quadros
02_anotacao_split/      anotação e divisão treino/val/teste (por sequência)
03_yolo_treino/         treino e avaliação do detector (com variantes)
04_sam_atributos/       caixa → máscara SAM → atributos aparentes
05_tracking/            associação entre quadros e deslocamento aparente
06_era5/                vento ERA5 e comparação com o deslocamento
07_super_resolucao/     Real-ESRGAN, com vs. sem
08_avaliacao/           tabelas, figuras e limites do método
```

No **Drive** existe uma pasta espelho (`drive_root` do `config.yaml`) com uma subpasta por etapa
(`01_dados/`, `02_anotacao_split/`, ...) que guarda vídeos, quadros, rótulos, pesos, máscaras e
saídas grandes. Os notebooks criam essas pastas sozinhos.

## Ordem de execução

Rodar em sequência: `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`. Cada notebook começa
declarando o que **lê** (saída da etapa anterior) e o que **grava**. Só GPU em 03, 04 e 07; o resto roda em CPU.

## Primeira configuração (uma vez)

1. Troque `USUARIO/projeto-plumas` pelo caminho real do repositório em todos os notebooks:
   ```bash
   grep -rl "USUARIO/projeto-plumas" . | xargs sed -i 's#USUARIO/projeto-plumas#SEU_USUARIO/SEU_REPO#g'
   ```
2. Se o repositório for **privado**: crie um token do GitHub (Settings → Developer settings →
   Fine-grained tokens, acesso só de leitura ao repositório) e guarde nos **Secrets** do Colab
   (ícone de chave) com o nome `GITHUB_TOKEN`, ativando "Acesso ao notebook".
3. Para o ERA5: crie conta no Copernicus CDS, aceite a licença do dataset e guarde a chave nos
   Secrets do Colab como `CDS_API_KEY` (só o valor da chave).
4. Ajuste `drive_root` e os parâmetros em `config.yaml`.

## Como trabalhar em dupla

- Cada pessoa mexe em **pastas de etapas diferentes** para evitar conflito de merge em `.ipynb`.
- Antes de commitar: *Editar → Limpar todas as saídas* (evita repositório pesado e diffs ruidosos).
- Salvar do Colab: *Arquivo → Salvar uma cópia no GitHub*, ou `git push` pelo terminal do Colab.
- Cada experimento relevante chama `log_experiment(...)`, que grava uma linha em `runs.jsonl` no Drive
  (data, etapa, commit, parâmetros, métricas, decisões).

## O que NÃO vai para o Git

Vídeos, quadros, pesos (`.pt`), ERA5 (`.nc`), segredos (`.cdsapirc`, tokens). Veja `.gitignore`.

## Testes

```bash
pip install -r requirements.txt pytest
python -m pytest -q tests
```

## Cuidados metodológicos (do escopo)

- Medidas na imagem são **aparentes** (pixels ou proporções); sem calibração da câmera não há metros nem m/s.
- Deslocamento aparente, vento do ERA5 e trajetória prevista por modelo físico são coisas **diferentes**.
- Compatibilidade entre direção na imagem e vento é evidência de associação possível, não causa.
- O ERA5 tem grade grossa (≈ 0,25°) e não descreve o microambiente local.
- Nunca separar quadros muito parecidos de uma mesma sequência entre treino e teste.

## Estado do esqueleto (o que já foi testado)

- **Testado com dados sintéticos:** módulos de `src/plumas/` (17 testes) e os notebooks 01, 02, 05, 06 (com ERA5 sintético) e 08, rodados de ponta a ponta.
- **Não executado ainda (precisa de GPU, pesos ou conta):** notebooks 03, 04 e 07 (YOLO, SAM, Real-ESRGAN), o download real do ERA5 e a célula de clone do GitHub no Colab. A sintaxe foi validada, mas ajustes de versão de bibliotecas podem ser necessários no primeiro uso.
- Os blocos marcados no código com variáveis a preencher (`BEST`, `PRE_WEIGHTS`, `ZERO_SHOT_WEIGHTS`, `EXTRA_TRAIN_DIRS`, `meta`) são intencionais.
