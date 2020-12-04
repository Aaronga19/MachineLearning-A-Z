

#                               Upper Confidence Bound

dataset<- read.csv("GitHub/MachineLearningAZ/datasets/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

# Implementar algoritmo UCB
N = 10000
d = 10
number_of_selections = integer(d)
sums_of_rewards = integer(d)
adds_selected = integer(d)
total_reward = 0
for (n in 1:N){
        max_upper_bound = 0
        ad = 0
        for (i in 1:d){
                if(number_of_selections[i]>0){
                        average_reward = sums_of_rewards[i] / number_of_selections[i]
                        delta = sqrt(3/2*log(n)/number_of_selections[i])
                        upper_bound = average_reward + delta
                }
                else{
                        upper_bound = 1e400
                }
                
                if (upper_bound > max_upper_bound){
                        max_upper_bound = upper_bound
                        ad = i
                }
        }
        adds_selected = append(adds_selected, ad)
        number_of_selections[ad] = number_of_selections[ad] + 1
        reward = dataset[n, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad]+ reward
        total_reward = total_reward + reward
}
# Visualización de resultados - Histograma
hist(adds_selected,
     col = "lightblue",
     main = "Histograma de los Anuncios",
     xlab = "ID del Anuncio",
     ylab = "Frecuencia absoluta del anuncio")