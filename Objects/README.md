# Nomenclatura dos arquivos:


## Primeiro parâmetro INJ_{}
Consiste no número da injetora. podendo variar de 01 à 15.

## Segundo parâmetro CL_{}-{}-...-{}_
Define quais classes fizeram parte do modelo deste arquivo.
Abaixo a lista com o número correspondente a cada classe.

    01 - Normal
    02 - Falha de Injeção
    03 - Rechupe
    
## Terceiro Parâmetro {} 
Consiste no Classificador utilizado, seguindo os valores abaixo.

    SVM_LINEAR : SVM com Kernel linear;
    SVM_SIGMOID : SVM com Kernel Senoidal;
    SVM_POLY : SVM com Kernel Polinomial;
    SVM_RBF : SVM com Kernel RBF;
    MLP_BACKPROP[{}-...-{}]: MLP com Backpropagation as variaveis sao um array com as camadas.
    
## Quarto parâmetro MOLD-{}
Define o mold a qual os dados pertencem sendo eles:

    01 - PLACA_MONO_LCD_ID_CAV_2_2
    02 - TRAVA_TERMINAIS_MONO_ID_CAV_4_1
    