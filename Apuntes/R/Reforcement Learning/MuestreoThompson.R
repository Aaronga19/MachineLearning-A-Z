#                               Muestreo Thompson 

# Importar los datos
dataset = read.csv("GitHub/MachineLearningAZ/datasets/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv")

# Implementar UCB
d = 10
N = 10000
numberOfRewards1 = integer(d)
numberOfRewards0 = integer(d)

ads_selected = integer(0)
total_reward = 0
for(n in 1:N){
        maxRandom = 0
        ad = 0
        for(i in 1:d){
                randomBeta = rbeta(n = 1, 
                                   shape1 = numberOfRewards1[i]+1,
                                   shape2 = numberOfRewards0[i]+1)
                if(randomBeta > maxRandom){
                        maxRandom = randomBeta
                        ad = i
                }
        }
        ads_selected = append(ads_selected, ad)
        reward = dataset[n, ad]
        if(reward == 1){
                numberOfRewards1[ad] = numberOfRewards1[ad]+1
                
        }else{
                numberOfRewards0[ad] = numberOfRewards0[ad]+1
                
        }
        total_reward = total_reward + reward
}

# Visualización de resultados - Histograma
hist(ads_selected,
     col = "lightblue",
     main = "Histograma de los Anuncios",
     xlab = "ID del Anuncio",
     ylab = "Frecuencia absoluta del anuncio")