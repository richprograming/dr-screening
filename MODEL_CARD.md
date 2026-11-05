# Model Card — Triagem de Retinopatia Diabética

## Uso pretendido
Demonstração educacional de classificação de imagens de fundo de olho em 5 graus de RD (0–4). **Não é dispositivo médico** e não deve apoiar decisões clínicas.

## Dados
APTOS 2019 Blindness Detection (Kaggle), ~3.600 imagens rotuladas. Divisão estratificada 80/20, seed 42.

## Modelo
EfficientNet-B0 pré-treinado (ImageNet), fine-tuning completo, perda com pesos por classe, AdamW + cosine LR.

## Métricas (preencher após o treino)
| Métrica | Valor |
|---|---|
| Quadratic Weighted Kappa | _TODO_ |
| AUC encaminhamento (grau ≥ 2) | _TODO_ |
| Recall por classe | _TODO_ |

As métricas vêm de `python -m src.dr.evaluate` e ficam registradas em `docs/metrics.json`.

## Limitações conhecidas
- As métricas são calculadas nos mesmos 20% de validação usados para escolher a melhor época, sem conjunto de teste separado, então tendem a ser otimistas.
- A divisão não verifica imagens duplicadas ou quase duplicadas entre treino e validação.
- Dataset de uma única origem (Índia): generalização para outras câmeras, populações e protocolos não foi validada.
- Classes intermediárias (leve/moderada) são as mais confundidas, até entre oftalmologistas.
- Grad-CAM mostra regiões influentes, não prova que o modelo "enxerga" lesões reais.
- Sem calibração de incerteza nem detecção de imagens fora de distribuição.

## Considerações éticas
Uso de dados públicos e anonimizados. Qualquer uso real exigiria validação clínica prospectiva e aprovação regulatória (ANVISA/ética em pesquisa).
