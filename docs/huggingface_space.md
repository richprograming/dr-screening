# Publicar a demo no Hugging Face Spaces

Faça isto depois do treino real, pois a demo precisa de `models/best.pt`.

## 1. Publique os pesos em um repositório de modelo

```bash
pip install huggingface_hub
hf auth login
hf repo create SEU_USUARIO/dr-screening --repo-type model
hf upload SEU_USUARIO/dr-screening models/best.pt best.pt
```

## 2. Crie o Space

Crie um Space com SDK **Gradio** e hardware CPU e copie para ele `app.py` e a pasta `src/`.

Inclua um `README.md` com este cabeçalho:

```yaml
---
title: Triagem de Retinopatia Diabética
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
---

Projeto educacional. Não é dispositivo médico.
```

Inclua também um `requirements.txt` só com o necessário, usando o PyTorch para CPU:

```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.15.0
torchvision==0.29.2
opencv-python-headless==5.0.0.93
pandas==3.1.1
scikit-learn==1.9.1
huggingface_hub==2.4.0
```

## 3. Configure a origem dos pesos

Em *Settings > Variables*, crie `DR_WEIGHTS_REPO` com o valor `SEU_USUARIO/dr-screening`. A demo baixa `best.pt` desse repositório na primeira previsão.
