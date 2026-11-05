# Triagem de Retinopatia Diabética com Deep Learning

[![CI](https://github.com/richprograming/dr-screening/actions/workflows/ci.yml/badge.svg)](https://github.com/richprograming/dr-screening/actions/workflows/ci.yml)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)](LICENSE)

Classificação de imagens de fundo de olho nos 5 graus de retinopatia diabética (RD), com explicabilidade por Grad-CAM, API de inferência e demo interativa. O modelo foi treinado e avaliado com o dataset APTOS 2019.

> ⚠️ Projeto educacional. Não é dispositivo médico. Veja o [Model Card](MODEL_CARD.md).

**Na validação (733 imagens):** QWK de 0,898 e AUC de encaminhamento de 0,986. O recall nos graus graves ainda é limitado; veja os [resultados](#resultados) e as [limitações](#limitações).

## Abordagem

| Etapa | Decisão |
|---|---|
| Modelo | EfficientNet-B0 pré-treinada no ImageNet, com fine-tuning completo |
| Pré-processamento | Recorte de bordas e realce de lesões pelo método de Ben Graham |
| Desbalanceamento | Perda de entropia cruzada com pesos por classe |
| Otimização | AdamW com taxa de aprendizado cosseno |
| Métrica principal | Quadratic Weighted Kappa (QWK), a métrica oficial da competição |
| Métrica de triagem | AUC de encaminhamento (RD moderada ou pior) |
| Explicabilidade | Grad-CAM implementado do zero, com testes contra gradientes calculados à mão |
| Entrega | API em FastAPI, demo em Gradio, exportação ONNX e Dockerfile |
| Qualidade | Testes automatizados, incluindo ponta a ponta com imagens sintéticas, e CI no GitHub Actions |

## Resultados

Avaliação nos 733 casos de validação (20% do APTOS 2019, divisão estratificada, seed 42). A reavaliação em um segundo ambiente reproduziu exatamente as mesmas métricas. Os dados completos estão em [`docs/metrics.json`](docs/metrics.json).

| Métrica | Valor |
|---|---|
| Quadratic Weighted Kappa | 0,898 |
| AUC de encaminhamento (grau ≥ 2) | 0,986 |
| Acurácia | 0,817 |
| Sensibilidade de encaminhamento (grau previsto ≥ 2) | 0,936 (279 de 298) |
| Especificidade de encaminhamento (grau previsto ≥ 2) | 0,936 (407 de 435) |

| Grau | Recall | Casos |
|---|---|---|
| Sem RD | 0,961 | 361 |
| Leve | 0,581 | 74 |
| Moderada | 0,775 | 200 |
| Grave | 0,462 | 39 |
| Proliferativa | 0,610 | 59 |

![Matriz de confusão](docs/confusion_matrix.png)

## Limitações

**Validação usada na seleção do modelo.** Os mesmos 733 casos escolheram a melhor época e geraram as métricas, sem conjunto de teste separado. Por isso, os valores tendem a ser otimistas.

**Imagens duplicadas.** 47 das 733 imagens de validação (6,4%) têm uma cópia provável no treino (correlação de detalhes finos ≥ 0,9), e 14 dos 54 pares têm graus diferentes. Sem essas 47 imagens, o QWK fica em 0,904 e a AUC em 0,986, sem queda em relação aos valores acima, então o modelo não foi treinado de novo. O método pode deixar passar cópias muito alteradas. Reproduza com `scripts/check_duplicates.py`; os dados estão em [`docs/duplicates_report.json`](docs/duplicates_report.json).

**Grad-CAM.** Na primeira imagem de cada grau na validação, sem seleção manual, houve 3 acertos e 2 erros. Nesses exemplos, o mapa muitas vezes destaca o disco óptico ou a periferia, e não as lesões mais visíveis. Isso não confirma que o modelo se baseia em lesões reais.

Os exemplos de Grad-CAM não são publicados aqui porque contêm imagens do APTOS, e as [regras da competição](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules) proíbem redistribuir os dados a quem não participa dela. Quem tem acesso ao dataset pode gerá-los com `scripts/gradcam_demo.py`; os arquivos ficam em `docs/`, ignorados pelo Git.

Outras limitações estão no [Model Card](MODEL_CARD.md).

## Como executar

Requer Python 3.11.

```bash
pip install -r requirements.txt

# Treino (sem GPU, use o notebook train_colab.ipynb)
python -m src.dr.train --data data/aptos --epochs 15

# Avaliação (grava docs/metrics.json e docs/confusion_matrix.png)
python -m src.dr.evaluate --data data/aptos --weights models/best.pt

# Exemplos de Grad-CAM em docs/
python scripts/gradcam_demo.py --data data/aptos --weights models/best.pt

# API
uvicorn api.main:app --reload
curl -F "file=@exemplo.png" http://localhost:8000/predict

# Demo em Gradio
python app.py
```

Os pesos treinados (`models/best.pt`) não são versionados. Para usar a API ou a demo, treine o modelo antes.

**Testes:** `pytest` usa imagens sintéticas e não baixa nada. Para testar o pipeline sem o dataset real:

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m src.dr.train --data data/synthetic --epochs 1 --bs 8 --out models/smoke.pt
```

**Docker:** `docker build -t dr-screening . && docker run -p 8000:8000 dr-screening`. Coloque `models/best.pt` antes do build; sem os pesos, a API sobe, mas `/predict` responde 503. O build da imagem ainda não foi testado.

### Dataset

O APTOS 2019 tem cerca de 10 GB e não está no repositório.

1. Aceite as [regras da competição](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules); sem isso, o download dá erro 403.
2. Instale o CLI e faça login:
   ```bash
   pip install kaggle six
   kaggle auth login
   ```
   Como alternativa, gere um token em [kaggle.com/settings](https://www.kaggle.com/settings) e defina a variável `KAGGLE_API_TOKEN`.
3. Baixe e extraia só o necessário em `data/aptos/` (o `tar` também funciona no Windows 10 ou superior):
   ```bash
   kaggle competitions download aptos2019-blindness-detection -p data/aptos
   tar -xf data/aptos/aptos2019-blindness-detection.zip -C data/aptos train.csv train_images
   ```

O resultado deve ser `data/aptos/train.csv` e `data/aptos/train_images/`, com 3.662 imagens PNG.

## Estrutura

```
src/dr/            data, model, train, evaluate, gradcam, inference, export_onnx
api/               API em FastAPI
scripts/           gradcam_demo, make_synthetic_data, check_duplicates
tests/             testes do pipeline e de ponta a ponta
app.py             demo em Gradio (veja docs/huggingface_space.md)
train_colab.ipynb  treino no Google Colab
```

## Próximos passos

- [ ] Avaliar em um conjunto de teste separado, com as duplicatas agrupadas na divisão
- [ ] Comparar com ConvNeXt/ViT e ensemble
- [ ] Testar regressão ordinal em vez de classificação
- [ ] Calibrar probabilidades e detectar imagens fora de distribuição
- [ ] Publicar a demo no Hugging Face Spaces

## Licença

O código está sob a [licença MIT](LICENSE). Ela não cobre os dados do APTOS 2019, que seguem as regras da competição no Kaggle.
