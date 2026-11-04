# 👁️ Triagem de Retinopatia Diabética com Deep Learning

Classificação de imagens de fundo de olho em 5 graus de retinopatia diabética (RD), com **explicabilidade (Grad-CAM)**, API de inferência e deploy em Docker.

> ⚠️ Projeto educacional. Não é dispositivo médico. Veja o [Model Card](MODEL_CARD.md).

## Destaques
- Transfer learning com **EfficientNet-B0**
- Pré-processamento de Ben Graham para realçar lesões
- Tratamento de **desbalanceamento de classes**
- Métrica oficial da competição: **Quadratic Weighted Kappa**
- Métrica clínica: AUC de **encaminhamento** (RD moderada ou pior)
- **Grad-CAM** implementado do zero
- API **FastAPI** + exportação **ONNX** + **Docker**

## Resultados
_Preencher após o treino: QWK, AUC, matriz de confusão e exemplos de Grad-CAM (adicione imagens em `docs/`)._

## Como rodar
```bash
pip install -r requirements.txt

# 1. Baixe o APTOS 2019 (Kaggle) em data/aptos/ (train.csv + train_images/)
kaggle competitions download -c aptos2019-blindness-detection -p data/aptos

# 2. Treine
python -m src.dr.train --data data/aptos --epochs 15

# 3. Avalie
python -m src.dr.evaluate --data data/aptos --weights models/best.pt

# 4. Suba a API
uvicorn api.main:app --reload
curl -F "file=@exemplo.png" http://localhost:8000/predict
```

Docker: `docker build -t dr-screening . && docker run -p 8000:8000 dr-screening`

Testes: `pytest`

## Estrutura
```
src/dr/   data, model, train, evaluate, gradcam, export_onnx
api/      FastAPI
tests/    testes do pipeline
```

## Próximos passos
- [ ] Treinar e preencher métricas
- [ ] Comparar com ConvNeXt/ViT e ensemble
- [ ] Regressão ordinal em vez de classificação
- [ ] Calibração de probabilidades e detecção de imagens fora de distribuição
- [ ] Demo no Hugging Face Spaces
- [ ] CI com GitHub Actions
