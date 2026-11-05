# Model Card — Triagem de Retinopatia Diabética

## Uso pretendido
Demonstração educacional de classificação de imagens de fundo de olho em 5 graus de RD (0–4). **Não é dispositivo médico** e não deve apoiar decisões clínicas.

## Dados
APTOS 2019 Blindness Detection (Kaggle), ~3.600 imagens rotuladas. Divisão estratificada 80/20, seed 42. As regras da competição limitam os dados a fins não comerciais e proíbem redistribuí-los a quem não participa dela, por isso nenhuma imagem do dataset é publicada neste repositório.

## Modelo
EfficientNet-B0 pré-treinado (ImageNet), fine-tuning completo, perda com pesos por classe, AdamW + cosine LR.

## Métricas
Calculadas nos 733 casos de validação.

| Métrica | Valor |
|---|---|
| Quadratic Weighted Kappa | 0,898 |
| AUC encaminhamento (grau ≥ 2) | 0,986 |
| Acurácia | 0,817 |
| Sensibilidade de encaminhamento (grau previsto ≥ 2) | 0,936 (279 de 298; 19 casos não encaminhados) |
| Especificidade de encaminhamento (grau previsto ≥ 2) | 0,936 (407 de 435) |
| Recall: Sem RD (361 casos) | 0,961 |
| Recall: Leve (74 casos) | 0,581 |
| Recall: Moderada (200 casos) | 0,775 |
| Recall: Grave (39 casos) | 0,462 |
| Recall: Proliferativa (59 casos) | 0,610 |

As métricas vêm de `python -m src.dr.evaluate` e ficam registradas em `docs/metrics.json`.

## Limitações conhecidas
- As métricas são calculadas nos mesmos 20% de validação usados para escolher a melhor época, sem conjunto de teste separado, então tendem a ser otimistas.
- A divisão não separa imagens duplicadas: 47 das 733 imagens de validação (6,4%) têm cópia provável no treino (correlação de detalhes finos ≥ 0,9), o que também afeta o treino. Sem elas, o QWK fica em 0,904 e a AUC em 0,986; a alta do QWK não indica melhora, pois a exclusão também remove pares com rótulos conflitantes. A verificação pode deixar passar cópias muito alteradas (`docs/duplicates_report.json`).
- Rótulos inconsistentes: 14 dos 54 pares de cópias prováveis têm graus diferentes, o que limita o desempenho mensurável.
- Dataset de uma única origem (Índia): generalização para outras câmeras, populações e protocolos não foi validada.
- Classes intermediárias (leve/moderada) são as mais confundidas, até entre oftalmologistas.
- Grad-CAM mostra regiões influentes, não prova que o modelo "enxerga" lesões reais. Em 5 exemplos inspecionados, o mapa muitas vezes destaca o disco óptico ou a periferia, e não as lesões mais visíveis. Essa amostra é pequena e não substitui uma avaliação sistemática.
- O recall é baixo nos graus graves (0,462 em Grave e 0,610 em Proliferativa), que têm poucos casos na validação.
- Sem calibração de incerteza nem detecção de imagens fora de distribuição.

## Considerações éticas
Uso de dados públicos e anonimizados. Qualquer uso real exigiria validação clínica prospectiva e aprovação regulatória (ANVISA/ética em pesquisa).
