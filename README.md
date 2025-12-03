# ImageAI
## Kernideen:

## Tensoren: Mehrdimensionale Arrays (ähnlich NumPy), aber mit GPU-Support (.to("cuda")).

## Autograd: Automatische Berechnung von Gradienten -> du musst Ableitungen nicht selbst herleiten.

## nn.Module: Baustein für eigene Modelle (z.B. class MyNet(nn.Module): ...).

## Optimierer: z.B. Adam, SGD für Gewichtsaktualisierung.

## Läuft auf CPU und GPU, wenn du CUDA hast.

## Typische Aufgaben damit:

## CNNs für Bilder (Klassifikation, Objekterkennung, Segmentation)

## RNN/Transformer für Text

## Custom Trainings-Loops (for epoch in range(...): ...)