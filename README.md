# Practical_ML-Project
# Prompt
The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.

# Prep

Load the required prackages for the machine learning exercise.
```{r}
# Load up dem packaaages
library(caret)
library(e1071)
library(randomForest)
```

Use the urls provided to download and read the data sets, one for the training data, and one for the final test.
```{r}
# Download and read training files
trainingUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(trainingUrl), na.strings=c("NA",""))
testing <- read.csv(url(testingUrl), na.strings=c("NA",""))
```

In order to train the data, lets now create a partition, such that we can use part of the training set to actually train the data, then we can use other part to get a sense of the accuracy of the model.
```{r}
# Create data Partitioon
PreParTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
TrainingDS <- training[PreParTrain, ]
TestingDS <- training[-PreParTrain, ]
```

# Prepping the Data

The first step will be to take out near zero variance, which diagnoses variables that have zero variance (ie unique variables), or near zero variance, which allows us to reduce the number of variables that do not meaningfully impact the result: lowering possible out of sample variance and reduces run time. 
```{r}
# Remove Near Zero Variance variables from training & test sets
NZCol <- nearZeroVar(TrainingDS)
TrainingDS <- TrainingDS[,-NZCol]
TestingDS <- TestingDS[,-NZCol]
```

Remove the 1st column of the dataset, which is simply the row number, which will only negatively impact the accuracy of the alogrithm.
```{r}
# Remove ID column
TrainingDS <- TrainingDS[c(-1)]
```

Remove variables in the training set that have limited number of entries (less than 1/3). This allows us to get better accuracy, as we're able to focus on the variables that matter.
```{r}
# Remove Columns where NAs are greater than 2/3 of the entries
TrainingDS <- TrainingDS[, -which(colMeans(is.na(TrainingDS)) > 0.66)]
```

# Create, Train, and Tune the Model
I will use a Random Forest model, which has better predictability and the negatives, interpreting variable/ variable interaction and long run time considering. Also, I really like the name - super fitting... haha... get it... fitting.... oh geeze. We'll also use a 10 fold cross validation to tune the model, and preprocess the variables to center and scale the data.
```{r}
# Create a Random Forest model, preprocessing w/ centering & scaling, and use K=10 cross validation
modFit <- train(TrainingDS$classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 10),TrainingDS)
print(modFit, digits=10)
```

This tells us that the best parameters are mtry = 79, which are randomly selected predictors. Given the level of accuracy in the training / cross validation, we should expect very good results when applying the algorithm to the partitioned test set. Now lets see how well our model holds up against the partitioned test data set.
```{r}
# Use the Random Forest model created to predict the class variables in the testing data set, so we can get an idea of the out of sample error rates
predictions <- predict(modFit, newdata=TestingDS)
print(confusionMatrix(predictions, TestingDS$classe), digits=5)
```

As we can see, the model does very well in terms of fit - as it has an accuracy of .9994, which means our estimated error rate is ~0.06%. Additionally, it looks as if the A and E classes have the best accuracy, with B least accurate (although not by much). Now that we have a pretty solid model, lets use it on the actual test set and see if we can get full marks on the results.

# Predicting the Final Test Outcomes
```{r}
# Use the Random Forest model to predict the final test set
predictions_FinalTest <- predict(modFit, testing)

# Create files for "easy" submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers <- as.character(predictions_FinalTest)
pml_write_files(answers)
```


