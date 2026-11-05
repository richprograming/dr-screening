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
Requer Python 3.11.

```bash
pip install -r requirements.txt

# 1. Baixe o APTOS 2019 (veja a seção abaixo)

# 2. Treine (sem GPU, use o notebook train_colab.ipynb)
python -m src.dr.train --data data/aptos --epochs 15

# 3. Avalie (grava docs/metrics.json e docs/confusion_matrix.png)
python -m src.dr.evaluate --data data/aptos --weights models/best.pt

# 4. Gere exemplos de Grad-CAM em docs/
python scripts/gradcam_demo.py --data data/aptos --weights models/best.pt

# 5. Suba a API
uvicorn api.main:app --reload
curl -F "file=@exemplo.png" http://localhost:8000/predict

# 6. Ou abra a demo em Gradio
python app.py
```

Docker: `docker build -t dr-screening . && docker run -p 8000:8000 dr-screening`. Coloque `models/best.pt` antes do build; sem os pesos, a API sobe, mas `/predict` responde 503.

Testes: `pytest` (usam imagens sintéticas e não baixam nada).

Para testar o pipeline sem o dataset real:
```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m src.dr.train --data data/synthetic --epochs 1 --bs 8 --out models/smoke.pt
```

### Baixar o APTOS 2019
São cerca de 10 GB.

1. Aceite as regras da [competição](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules); sem isso, o download dá erro 403.
2. Instale o CLI e faça login:
   ```bash
   pip install kaggle
   kaggle auth login
   ```
   Como alternativa, gere um token em [kaggle.com/settings](https://www.kaggle.com/settings) e defina a variável `KAGGLE_API_TOKEN`.
3. Baixe e extraia só o necessário em `data/aptos/` (o `tar` também funciona no Windows 10 ou superior):
   ```bash
   kaggle competitions download -c aptos2019-blindness-detection -p data/aptos
   tar -xf data/aptos/aptos2019-blindness-detection.zip -C data/aptos train.csv train_images
   ```

O resultado deve ser `data/aptos/train.csv` e `data/aptos/train_images/`, com 3.662 imagens PNG.

## Estrutura
```
src/dr/            data, model, train, evaluate, gradcam, inference, export_onnx
api/               FastAPI
scripts/           gradcam_demo, make_synthetic_data
tests/             testes do pipeline e de ponta a ponta
app.py             demo em Gradio (veja docs/huggingface_space.md)
train_colab.ipynb  treino no Google Colab
```

## Próximos passos
- [ ] Treinar e preencher métricas
- [ ] Comparar com ConvNeXt/ViT e ensemble
- [ ] Regressão ordinal em vez de classificação
- [ ] Calibração de probabilidades e detecção de imagens fora de distribuição
- [ ] Publicar a demo no Hugging Face Spaces
- [x] CI com GitHub Actions
