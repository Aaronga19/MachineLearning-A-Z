## Tratamiento de los NaN

dataset = read.csv("Data.csv")


dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x)mean(x,na.rm = T)),dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x)mean(x,na.rm = T)),dataset$Salary)