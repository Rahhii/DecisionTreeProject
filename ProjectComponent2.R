library(rpart)
# Read in the dataset
iris <- read.csv("IRIS.csv", header=TRUE)

# We should split the dataset into a training and test dataset.
# For reproducibility
set.seed(97863523)

# According to ChatGPT, the following library and three lines of code will split up the code
library(caret)

# Now we split the dataset (70% in training, 30% in test)
idx <- createDataPartition(iris$species, p=0.7, list=FALSE)
trainset <- iris[idx,]
testset <- iris[-idx,]

# First we created the tree. We used all the columns in the model
rm <- rpart(species ~ ., data = trainset, method = "class")

# now we will plot to see 
library(rpart.plot)
png("rpart.png")
plot <- rpart.plot(rm)
print("hello!")

# Ran a prediction on the training dataset (that we used to create the model). 
pred_rm_train <- predict(rm, trainset, type="class")

# Since we are working off of 3 discrete categories (setosa, vesticolor, virginica),
# we calculate the error using a confusion matrix and not using MSE
# tried conf_matrix_rm_train <- confusionMatrix(data=pred_rm_train, reference=trainset$species)
# Got Error: `data` and `reference` should be factors with the same levels.

# So we have to convert the species column into a factor so it's treated
# as a category with discrete values and the way to do that (according to StackOverflow)
# is to use as.factor().
# Rather than create a new column for this factored category, We are  going to
# overwrite the species column so I don't have to change the code I alread wrote.
iris$species <- as.factor(iris$species)

# Now, resplit the data and make the model
idx <- createDataPartition(iris$species, p=0.7, list=FALSE)
trainset <- iris[idx,]
testset <- iris[-idx,]
rm <- rpart(species ~ ., data=trainset, method="class")
pred_rm_train <- predict(rm, trainset, type="class")
conf_matrix_rm_train <- confusionMatrix(data=pred_rm_train, reference=trainset$species)
conf_matrix_rm_train

# We got results - and this is specifically for our training dataset
#                       Class: Iris-setosa Class: Iris-versicolor Class: Iris-virginica
#...
# Balanced Accuracy                1.0000                 0.9643                0.9500

# Now let's see what the accuracy of the model is on the test dataset
pred_rm_test <- predict(rm, newdata=testset, type="class")
conf_matrix_rm_test <- confusionMatrix(data=pred_rm_test, reference=testset$species)
conf_matrix_rm_test
#                       Class: Iris-setosa Class: Iris-versicolor Class: Iris-virginica
#....
# Balanced Accuracy                1.0000                 0.9667                0.9333

# Interesting! Somehow the model is better on the test dataset than on the training dataset.
# Nevertheless, it's great on iris-setosa, very good on iris-versicolor, and good on iris-virginica

# Naive-Bayes
# Load the library and make the model
library(e1071)
nbm <- naiveBayes(species ~ ., data = trainset)

# The rest of the code is the same as above
pred_nbm_train <- predict(nbm, trainset, type="class")
conf_matrix_nbm_train <- confusionMatrix(data=pred_nbm_train, reference=trainset$species)
conf_matrix_nbm_train
#                           Class: Iris-setosa Class: Iris-versicolor Class: Iris-virginica
# ...
# Balanced Accuracy                1.0000                 0.9500                0.9429

# Let's see what the accuracy of the model is on the test dataset
pred_nbm_test <- predict(nbm, newdata=testset, type="class")
conf_matrix_nbm_test <- confusionMatrix(data=pred_nbm_test, reference=testset$species)
print(conf_matrix_nbm_test)
#                             Class: Iris-setosa Class: Iris-versicolor Class: Iris-virginica
# ...
# Balanced Accuracy                1.0000                 0.9833                0.9667



# Last up, we compare the accuracy of the Naïve Bayes Classification with the rpart accuracy. 
# The overall accuracy of rpart was   Accuracy : 0.9556   = 95.56%
# The overall accuracy of the Naive Bayes is  Accuracy : 0.9778   = 97.78%

# They are both very good, but it seems Naive Bayes is slightly more accurate.



#accuracy values
rpartplot <- c(1.0000, 0.9667, 0.9333)
naiveplot <- c(1.0000, 0.9833, 0.9667)
speciesplot <- c("Setosa", "Versicolor", "Virginica")

# bar

png('bar.png')
barplot(
  rbind(rpartplot, naiveplot),
  beside = TRUE,
  ylim = c(0.9, 1),  # Set y-axis limits
  names.arg = speciesplot,
  xlab = "Species",
  ylab = "Accuracy",
  main = "rpart v naive: Accuracy"
)

#scatter
png('plot.png')
plot(
  1:3, rpartplot,
  pch = 15,
  xaxt = "n",
  xlab = "Species",
  ylab = "Balanced Accuracy",
  ylim = c(0.9, 1),
  main = "rpart v naive: Accuracy"
)
points(
  1:3, naiveplot,
  pch = 15
)
axis(1, at = 1:3, labels = speciesplot)

#histogram
hist(
  c(rpartplot, naiveplot),
  breaks = 5,  # Adjust number of bins as needed
  col = c("red", "yellow"),
  xlab = "Balanced Accuracy",
  ylab = "Frequency",
  main = "rpart v naive: Accuracy",
  xlim = c(0.9, 1)
)


# box
png('box.png')
boxplot(
  list(rpartplot, naiveplot),
  col = c("red", "yellow"),
  names = c("rpart", "Naive Bayes"),
  ylab = "Balanced Accuracy",
  main = "rpart v naive: Accuracy"
)


