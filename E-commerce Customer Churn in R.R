###########################################

#Author: Ngu Hui En
#Course: Applied Machine Learning (AML)
#Topic: Customer Churn Prediction in E-Commerce Sector using Machine Learning Algorithms
#Task: Predict Customer Churn (Binary Classification)
#Model Implemented: Logistic Regression, Decision Tree, Random Forest
#Conclusion: Random Forest achieved the best performance in predicting customer churn.

###########################################

setwd("C:/Users/huien/Downloads/S1 - Applied Machine Learning/Assignment")

############### IMPORT AND LOAD THE DATA ###############
library(readxl)
mydata = read_excel("E Commerce Dataset.xlsx")

############### EXPLORE THE DATA ###############
library(pastecs)
library(DataExplorer)
str(mydata) #check data structure
summary(mydata) #descriptive statistics
dim(mydata) #dimension of data
head(mydata) #top 6 data
tail(mydata) #last 6 data
names(mydata) #column names of data
length(mydata) #total column of data 
plot_str(mydata)
stat.desc(mydata) #statistics

################################### 

#DATA PRE-PROCESSING 

###################################

############### DATA STRUCTURE CONVERSION ###############
library(dplyr)
mydata <- mydata %>% mutate_if(is.character,as.factor)

#some of the encoded labels as loaded as Numerical
mydata$CustomerID <- as.factor(mydata$CustomerID)
mydata$Churn <- as.factor(mydata$Churn)
mydata$CityTier <- as.factor(mydata$CityTier)
mydata$SatisfactionScore <- as.factor(mydata$SatisfactionScore)
mydata$Complain <- as.factor(mydata$Complain)

############### DATA CLEANING ###############
ds <- mydata

#change column names
library(tidyverse)
names(ds)[names(ds) == "PreferedOrderCat"] <- "PreferredOrderCat"

ds$PreferredLoginDevice <- as.factor(str_replace(ds$PreferredLoginDevice, "Mobile Phone", "Mobile"))
ds$PreferredLoginDevice <- as.factor(str_replace(ds$PreferredLoginDevice, "Phone", "Mobile"))
ds$PreferredPaymentMode <- as.factor(str_replace(ds$PreferredPaymentMode, "CC", "Credit Card"))
ds$PreferredPaymentMode <- as.factor(str_replace(ds$PreferredPaymentMode, "COD", "Cash on Delivery"))
ds$PreferredOrderCat <- as.factor(str_replace(ds$PreferredOrderCat, "Mobile Phone", "Mobile"))

str(ds)
summary(ds)

#check duplicates
sum(duplicated(ds)) 

#drop irrelevant feature 
ds <- select(ds, -CustomerID)

############### MISSING VALUE ###############
library(funModeling)
sum(is.na(ds))
colSums(is.na(ds))
plot_missing(ds)

#imputing missing values under numerical data using median
ds$Tenure[is.na(ds$Tenure)] <- round(median(ds$Tenure,na.rm=TRUE))
ds$WarehouseToHome[is.na(ds$WarehouseToHome)] <- round(median(ds$WarehouseToHome,na.rm=TRUE))
ds$HourSpendOnApp[is.na(ds$HourSpendOnApp)] <- round(median(ds$HourSpendOnApp,na.rm=TRUE))
ds$OrderAmountHikeFromlastYear[is.na(ds$OrderAmountHikeFromlastYear)] <- round(median(ds$OrderAmountHikeFromlastYear,na.rm=TRUE))
ds$CouponUsed[is.na(ds$CouponUsed)] <- round(median(ds$CouponUsed,na.rm=TRUE))
ds$OrderCount[is.na(ds$OrderCount)] <- round(median(ds$OrderCount,na.rm=TRUE))
ds$DaySinceLastOrder[is.na(ds$DaySinceLastOrder)] <- round(median(ds$DaySinceLastOrder,na.rm=TRUE))

#check missing value
sum(is.na(ds))
colSums(is.na(ds))
plot_missing(ds)
str(ds)

############### OUTLIERS ###############
# Filter numeric variables from the dataset
numeric_data <- ds %>%
  select(
    Tenure,
    WarehouseToHome,
    HourSpendOnApp,
    NumberOfDeviceRegistered,
    NumberOfAddress,
    OrderAmountHikeFromlastYear,
    CouponUsed,
    OrderCount,
    DaySinceLastOrder,
    CashbackAmount
  )

boxplot(numeric_data)

# Detection and Treatment
for (var in colnames(numeric_data)) {
  # Calculate quartiles and IQR
  q1 <- quantile(numeric_data[[var]], 0.25)
  q3 <- quantile(numeric_data[[var]], 0.75)
  iqr <- q3 - q1
  
  # Calculate lower and upper bounds
  lower_bound <- q1 - (1.5 * iqr)
  upper_bound <- q3 + (1.5 * iqr)
  
  # Detect outliers
  outliers <- which(numeric_data[[var]] < lower_bound | numeric_data[[var]] > upper_bound)
  # Display outliers
  cat("Outliers for variable", var, ":\n")
  print(numeric_data[outliers, ])
  
  # Winsorize outliers
  numeric_data[[var]] <- pmin(pmax(numeric_data[[var]], lower_bound), upper_bound)
}

# Check the updated dataset after outlier treatment
# Convert numeric columns to integers
for (var in colnames(numeric_data)) {
  numeric_data[[var]] <- as.integer(numeric_data[[var]])
}
# Convert numeric columns to numeric type
numeric_data <- sapply(numeric_data, as.numeric)
str(numeric_data)
boxplot(numeric_data)

# Categorical
categorical <- ds %>%
  select(
    Churn,
    Gender,
    PreferredLoginDevice,
    CityTier,
    PreferredPaymentMode,
    PreferredOrderCat,
    SatisfactionScore,
    MaritalStatus,
    Complain,
  )

# Combine winsorized numeric data with categorical data
combined_data <- cbind(numeric_data, categorical)

dt <- combined_data
############### FREQUENCY OF EACH COLUMN ###############
table(dt$Churn)
table(dt$Tenure)
table(dt$PreferredLoginDevice)
table(dt$CityTier)
table(dt$WarehouseToHome)
table(dt$PreferredPaymentMode)
table(dt$Gender)
table(dt$HourSpendOnApp)
table(dt$NumberOfDeviceRegistered)
table(dt$PreferredOrderCat)
table(dt$SatisfactionScore)
table(dt$MaritalStatus)
table(dt$NumberOfAddress)
table(dt$Complain)
table(dt$OrderAmountHikeFromlastYear)
table(dt$CouponUsed)
table(dt$OrderCount)
table(dt$DaySinceLastOrder)
table(dt$CashbackAmount)

############### EXPLORATORY DATA ANALYSIS (EDA) ###############
summary(dt)
names(dt)
head(dt)
dim(dt)
str(dt)
profiling_num(dt) 

############### Univariate analysis
library(ggplot2)
library(gridExtra)

#plot of numerical variables 
plot_num(dt)

#plot of categorical variables
plot_bar(categorical)

#checking Churn (Class) distribution 
table(dt$Churn) 
prop.table(table(dt$Churn))
ggplot(data = dt, aes(x = factor(Churn, labels = c("No", "Yes")))) + 
  geom_bar(fill="steelblue") + 
  labs(title = "Churn Distribution by Customer", x="Churn", y="Frequency") +
  scale_x_discrete(labels = c("No", "Yes"))

############### Bivariate analysis
############### BOXPLOT
# Convert the numeric_data list to a dataset
numeric_dataset <- data.frame(numeric_data)

# Subset the dataset to include only the 'Churn' variable and numeric columns
dt_subset <- dt[, c("Churn", colnames(numeric_dataset))]

# Set the figure size
options(repr.plot.width = 50, repr.plot.height = 20)

# Create a list to store the boxplot plots
boxplot_list <- list()

# Iterate over each numeric column in the dataset
for (col in colnames(numeric_dataset)) {
  # Create a boxplot for the current column with 'Churn' as the y variable
  boxplot_plot <- ggplot(dt_subset, aes(x = Churn, y = .data[[col]], fill = Churn)) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", col, "by Churn"),
         x = "Churn",
         y = col) +
    theme(plot.title = element_text(size = 10)) + # Set smaller title font size
    scale_fill_manual(values = c("lightblue", "salmon"))  # Set the fill colors
  
  # Append the boxplot plot to the list
  boxplot_list[[col]] <- boxplot_plot
}

# Arrange and display the boxplot plots
grid.arrange(grobs = boxplot_list, nrow = 3, ncol = 4)

################ BARPLOT 
# Define a function to create bar plots
plot_bar <- function(dt, variable) {
  ggplot(dt, aes(x = .data[[variable]], fill = Churn)) +
    geom_bar() +
    labs(
      title = paste("Bar Plot of", variable, "by Churn"),
      x = variable,
      y = "Count"
    ) +
    scale_fill_manual(values = c("salmon", "lightblue"))  # Set the fill color as red and blue for churn
}

# Create a list to store the bar plots
barplot_list <- list()

# Iterate over each categorical variable
for (var in names(categorical)[-1]) {
  # Create the bar plot
  plot <- plot_bar(categorical, var)
  
  # Append the bar plot to the list
  barplot_list[[var]] <- plot
}

# Arrange and display the bar plots
grid.arrange(grobs = barplot_list, nrow = 3, ncol = 3)

############### DATA ENCODING ###############
library(caret)
str(dt)

#One-hot encoding for nominal
dt1 <- dt %>%
  select(
    PreferredPaymentMode,
    PreferredOrderCat,
    MaritalStatus,
  )

dt2 <- dt %>%
  select(
    -PreferredPaymentMode,
    -PreferredOrderCat,
    -MaritalStatus
  )

###label encoding for ordinal
dt2$SatisfactionScore <- as.numeric(factor(dt2$SatisfactionScore))
dt2$CityTier <- as.numeric(factor(dt2$CityTier))
dt2$Complain <- as.numeric(dt2$Complain)
dt2$Complain <- ifelse(dt2$Complain == 2, 1, 0)

dt2$Gender <- ifelse(dt2$Gender == "Female", "1", "0") #female=1, male=0
dt2$Gender <- as.numeric(dt2$Gender)
dt2$PreferredLoginDevice <- ifelse(dt2$PreferredLoginDevice == "Mobile", "1", "0") #mobile=1, computer=0
dt2$PreferredLoginDevice <- as.numeric(dt2$PreferredLoginDevice)

###one-hot encoding
dmy <- dummyVars("~.", data = dt1)
dataset_one_hot <- data.frame(predict(dmy,newdata=dt1))
dataset_one_hot
str(dataset_one_hot)

data_final <- cbind(dt2, dataset_one_hot)
str(data_final)

data_final$Churn <- as.numeric(data_final$Churn)
data_final$Churn <- ifelse(data_final$Churn == 2, 1, 0)

############### NORMALIZATION ###############
#exclude target variable
data_for_normalize <- select(data_final,-Churn) 

# Subset the numeric columns
numeric_cols <- sapply(data_for_normalize, is.numeric)
numeric_data <- data_for_normalize[, numeric_cols]
# Apply min-max scaling using preProcess
preproc_obj <- preProcess(numeric_data, method = "range")
normalized_data <- predict(preproc_obj, numeric_data)
# Replace the normalized numeric columns in the original data frame
data_for_normalize[, numeric_cols] <- normalized_data

# Combine the 'Churn' variable with the normalized data
data_normalized <- cbind(data_for_normalize, Churn = data_final$Churn)

# View the updated dataset
str(data_normalized)

################ CORRELATION ###############
library(reshape2)
correlation_matrix <- cor(data_normalized)
print(correlation_matrix)
ggplot(data = melt(correlation_matrix), aes(x = Var1, y = Var2, fill = value, label = round(value, 2))) +
  geom_tile() +
  geom_text(color = "black", size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Correlation Matrix Heatmap")

library(caret)
highlyCorrelated <- findCorrelation(correlation_matrix, cutoff=0.75)
print(highlyCorrelated)

############### FEATURE SELECTION ###############
library(Boruta)
set.seed(123)
data_boruta <- Boruta(Churn ~., data = data_normalized, doTrace = 2)
print(data_boruta)
plot(data_boruta, las = 2, cex.axis = 0.7)
attStats(data_boruta)

##############################################

#DATA SPLITTING
#Original Data
#80% TRAINING SET, 20% TEST SET

###############################################
data_normalized$Churn <- as.factor(data_normalized$Churn)

library(caTools)
set.seed(123)
split = sample.split(data_normalized$Churn, SplitRatio = 0.8)
training_set = subset(data_normalized, split == TRUE)
test_set = subset(data_normalized, split == FALSE)

#Check class distribution
table(data_normalized$Churn)
prop.table(table(data_normalized$Churn))
prop.table(table(training_set$Churn))
prop.table(table(test_set$Churn))

###############################

#DATA SPLITTING
#Over Sampling Data
#80% TRAINING SET, 20% TEST SET

################################
#Using library ROSE (Random Over-Sampling)
library(ROSE)
set.seed(123)
table(training_set$Churn)
training_set_over <- ovun.sample(Churn ~., data = training_set, 
                                method = "over", N = 7492, seed = 123)$data

table(training_set_over$Churn)
prop.table(table(training_set_over$Churn))

#remain the same test set for oversampling
test_set_over <- test_set
table(test_set_over$Churn)
table(test_set$Churn)

#######################################################

#MODEL IMPLEMENTATION 1: LOGISTIC REGRESSION


#######################################################
#######################################################

#EXPERIMENT1 - Original DATASET 
#LOGISTIC REGRESSION

#######################################################
#Building classifier
set.seed(123)
classifier = glm(Churn ~.,
                 training_set,
                 family = binomial)
summary(classifier)

#Training set results
#-29 to exclude Churn (Target Variable)
pred_prob_training <- predict(classifier, type = 'response', training_set[ ,-29])
pred_prob_training
pred_class_training = ifelse(pred_prob_training > 0.5, 1, 0)
pred_class_training
cm_train_lr <- confusionMatrix(factor(pred_class_training), factor(training_set$Churn), positive = "1")
cm_train_lr

f1_score_train_lr <- cm_train_lr$byClass["F1"]
f1_score_train_lr

#Test set results
pred_prob_test <- predict(classifier, type = 'response', test_set[ ,-29] )
pred_prob_test
pred_class_test = ifelse(pred_prob_test > 0.5, 1, 0)
pred_class_test
cm_lr <- confusionMatrix(factor(pred_class_test), factor(test_set$Churn), positive = "1")
cm_lr

f1_score_lr <- cm_lr$byClass["F1"]
f1_score_lr

### ROC curve
library(ROCR)
pred_lr = prediction(pred_prob_test, test_set$Churn)
perf_lr = performance(pred_lr, "tpr", "fpr")
pred_lr
perf_lr
plot(perf_lr, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1 - Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

# Area Under Curve
auc_lr <- as.numeric(performance(pred_lr, "auc")@y.values)
auc_lr <-  round(auc_lr, 4)
auc_lr

#########################################################

#EXPERIMENT2 - Over-Sampling DATASET 
#LOGISTIC REGRESSION

########################################################
set.seed(123)
classifier_over = glm(Churn ~.,
                      training_set_over,
                      family = binomial)
summary(classifier_over)

#Training set results
pred_prob_training_over <- predict(classifier_over, type = 'response', training_set_over[ ,-29])
pred_prob_training_over
pred_class_training_over <- ifelse(pred_prob_training_over > 0.5, 1, 0)
pred_class_training_over
cm_lr_train_over <- confusionMatrix(factor(pred_class_training_over), factor(training_set_over$Churn), positive = "1")
cm_lr_train_over

f1_score_lr_train_over <- cm_lr_train_over$byClass["F1"]
f1_score_lr_train_over

#Test set results
pred_prob_test_over <- predict(classifier_over, type = 'response', test_set_over[ ,-29] )
pred_prob_test_over
pred_class_test_over = ifelse(pred_prob_test_over > 0.5, 1, 0)
pred_class_test_over
cm_lr_over <- confusionMatrix(factor(pred_class_test_over), factor(test_set_over$Churn), positive = "1")
cm_lr_over

f1_score_lr_over <- cm_lr_over$byClass["F1"]
f1_score_lr_over

### ROC curve
pred_lr_over = prediction(pred_prob_test_over, test_set_over$Churn)
perf_lr_over = performance(pred_lr_over, "tpr", "fpr")
pred_lr_over
perf_lr_over
plot(perf_lr_over, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1 - Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

# Area Under Curve
auc_lr_over <- as.numeric(performance(pred_lr_over, "auc")@y.values)
auc_lr_over <-round(auc_lr_over, 4)
auc_lr_over

########################################################
###Variable Importance by Coefficient
# Extract variable coefficients and names, filtering out missing values
coef_data_lr_over <- data.frame(Variable = c("(Intercept)", names(coef(classifier_over))[-29]),
                                Coefficient = c(coef(classifier_over)[1], coef(classifier_over)[-29]),
                                stringsAsFactors = FALSE)
coef_data_lr_over <- na.omit(coef_data_lr_over)
coef_data_lr_over

# Plot variable importance using ggplot2
ggplot(coef_data_lr_over, aes(x = Variable, y = Coefficient, fill = Coefficient > 0)) +
  geom_bar(stat = "identity", position = "identity", width = 0.5) +
  labs(title = "Logistic Regression Coefficient") +
  xlab("Variable") +
  ylab("Coefficient") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("lightblue", "salmon"), guide = "none")

#########################################################

#EXPERIMENT3 - Over-Sampling DATASET 
#LOGISTIC REGRESSION
#Cross Validation
#Hyperparameter Tuning

########################################################
library(glmnet)

# Define the hyperparameters (regularization)
hyperparameters <- expand.grid(
  alpha = c(0, 0.1, 0.5, 1),
  lambda = c(0.01, 0.1, 1)
)

# Grid search using cross-validation
cv_fit <- cv.glmnet(
  x = as.matrix(training_set_over[, -29]),
  y = training_set_over$Churn,
  family = "binomial",
  alpha = hyperparameters$alpha[1],  # Choose a single alpha value
  lambda = hyperparameters$lambda,
  type.measure = "class",
  nfolds = 10,
  classProbs = TRUE
)

# Get the best hyperparameters
best_alpha <- cv_fit$lambda.1se
best_alpha
best_lambda <- cv_fit$lambda.min
best_lambda

# Train the model with the best hyperparameters
classifier_tuned <- glmnet(
  x = as.matrix(training_set_over[, -29]),
  y = training_set_over$Churn,
  family = "binomial",
  alpha = best_alpha,
  lambda = best_lambda
)
classifier_tuned

# Test set results
pred_prob_test_tuned <- predict(classifier_tuned, newx = as.matrix(test_set_over[, -29]), type = "response")
pred_class_test_tuned <- ifelse(pred_prob_test_tuned > 0.5, 1, 0)

# Evaluate the performance of the tuned model
cm_lr_tuned <- confusionMatrix(factor(pred_class_test_tuned), factor(test_set_over$Churn), positive = "1")
cm_lr_tuned

f1_score_lr_tuned <- cm_lr_tuned$byClass["F1"]
f1_score_lr_tuned

###ROC curve
library(ROCR)
pred_lr_tuned <- prediction(pred_prob_test_tuned,test_set_over$Churn)
perf_lr_tuned <- performance(pred_lr_tuned, "tpr","fpr")
plot(perf_lr_tuned, colorize = TRUE, main = "ROC curve",
     ylab = "Sensitivity", xlab = "1 - Specificity",
     print.auc = TRUE,
     print.cutoffs.at = seq(0, 1, 0.3),
     text.adj = c(-0.2, 1.7))

#AUC
auc_lr_tuned <- as.numeric(performance(pred_lr_tuned, "auc")@y.values)
auc_lr_tuned <- round(auc_lr_tuned, 4)
auc_lr_tuned

###################################################

#MODEL EXPERIMENTATION 2: DECISION TREE

##################################################

#EXPERIMENT1 - ORIGINAL DATASET
#DECISION TREE

##################################################
library(rpart)
library(rpart.plot)
library(party)

# Create the decision tree model
tree_model <- rpart(Churn ~ ., data = training_set, method = "class")
print(tree_model)
prp(tree_model)
prp(tree_model, type = 5, extra = 100)
rpart.plot(tree_model, extra = 101, nn = TRUE)

#split with entropy information
ent_Tree = rpart(Churn ~ ., data=training_set, method="class", parms=list(split="information"))
ent_Tree
prp(ent_Tree)
rpart.plot(tree_model, extra = 101, nn = TRUE) # GINI index
rpart.plot(ent_Tree, extra = 101, nn = TRUE) # ENTROPY index

plotcp(tree_model)
plotcp(ent_Tree)

#Training set result
tree_predict_train <- predict(tree_model, newdata = training_set, type = "class")
head(tree_predict_train)
cm_train_dt <- confusionMatrix(factor(tree_predict_train), factor(training_set$Churn), positive = "1")
cm_train_dt
f1_score_train_dt <- cm_train_dt$byClass["F1"]
f1_score_train_dt

#Test set result
tree_predict <- predict(tree_model, newdata = test_set, type = "class")
head(tree_predict)
cm_dt <- confusionMatrix(factor(tree_predict), factor(test_set$Churn), positive="1")
cm_dt
f1_score_dt <- cm_dt$byClass["F1"]
f1_score_dt

###ROC curve
pred_dt = predict(tree_model, test_set)
pred_dt
pred_dt[,2] #estimate class "1" = churn

pred_dt = prediction(pred_dt[,2], test_set$Churn)
perf_dt = performance(pred_dt, "tpr", "fpr")
pred_dt
perf_dt
plot(perf_dt, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1 - Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

# Area Under Curve (AUC)
auc_dt = as.numeric(performance(pred_dt, "auc")@y.values)
auc_dt = round(auc_dt, 4)
auc_dt

##################################################

#EXPERIMENT2 - OVER SAMPLING DATASET
#DECISION TREE

##################################################
# Create the decision tree model
tree_model_over <- rpart(Churn ~ ., data = training_set_over, method = "class")
print(tree_model_over)
prp(tree_model_over)
prp(tree_model_over, type = 5, extra = 100)
rpart.plot(tree_model_over, extra = 101, nn = TRUE)

#split with entropy information
ent_Tree_over = rpart(Churn ~ ., data=training_set_over, method="class", parms=list(split="information"))
ent_Tree_over
prp(ent_Tree_over)
rpart.plot(tree_model_over, extra = 101, nn = TRUE) # GINI index
rpart.plot(ent_Tree_over, extra = 101, nn = TRUE) # ENTROPY index

plotcp(tree_model_over)
plotcp(ent_Tree_over)

#Training set result
tree_predict_train_over <- predict(tree_model_over, newdata = training_set_over, type = "class")
head(tree_predict_train_over)
cm_train_dt_over <- confusionMatrix(factor(tree_predict_train_over), factor(training_set_over$Churn), positive = "1")
cm_train_dt_over
f1_score_train_dt_over <- cm_train_dt_over$byClass["F1"]
f1_score_train_dt_over

#Test set result
tree_predict_over <- predict(tree_model_over, newdata = test_set_over, type = "class")
head(tree_predict_over)
cm_dt_over <- confusionMatrix(factor(tree_predict_over), factor(test_set_over$Churn), positive="1")
cm_dt_over
f1_score_dt_over <- cm_dt_over$byClass["F1"]
f1_score_dt_over

###ROC curve
pred_dt_over = predict(tree_model_over, test_set_over)
pred_dt_over
pred_dt_over[,2]

pred_dt_over = prediction(pred_dt_over[,2], test_set_over$Churn)
perf_dt_over = performance(pred_dt_over, "tpr", "fpr")
pred_dt_over
perf_dt_over
plot(perf_dt_over, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1 - Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

#AUC
auc_dt_over = as.numeric(performance(pred_dt_over, "auc")@y.values)
auc_dt_over = round(auc_dt_over, 4)
auc_dt_over

###Variable importance
var_importance_over <- tree_model_over$variable.importance
var_importance_over

# Convert variable importance to data frame
var_importance_df <- data.frame(
  Variable = names(var_importance_over),
  Importance = var_importance_over,
  stringsAsFactors = FALSE
)

# Plot variable importance using ggplot2
ggplot(var_importance_df, aes(x = Variable, y = Importance)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Decision Tree Variable Importance") +
  xlab("Variable") +
  ylab("Importance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#########################################################

#EXPERIMENT3 - Over-Sampling DATASET 
#DECISION TREE
#Cross Validation
#Hyperparameter Tuning

########################################################
library(mlr)
library(caret)
library(pROC)

getParamSet("classif.rpart")

# Define the task
task <- makeClassifTask(data = training_set_over, target = "Churn")

# Define the learner
learner <- makeLearner("classif.rpart", predict.type = "prob")

# Define the search space for hyperparameters
param_grid <- makeParamSet(
  makeIntegerParam("minsplit", lower = 10, upper = 20),
  makeIntegerParam("minbucket", lower = 10, upper = 20),
  makeNumericParam("cp", lower = 0.001, upper = 0.01)
)

# Define the resampling strategy
cv <- makeResampleDesc("CV", iters = 10)

# Define the performance measure
measure <- mlr::auc

# Perform grid search with cross-validation
tuner <- mlr::makeTuneControlGrid(resolution = 10)
result <- mlr::tuneParams(learner, task, resampling = cv, measures = measure, par.set = param_grid, control = tuner)

best_index <- which.min(result$y)
best_index
best_params <- result$x[[best_index]]
best_params
result$x

#Use the best parameters
best_params <- list(minsplit = 17, minbucket = 10, cp = 0.001)

learner <- makeLearner("classif.rpart", 
                       predict.type = "prob",
                       minsplit = best_params$minsplit,
                       minbucket = best_params$minbucket,
                       cp = best_params$cp)
learner
subset_indices <- 1:nrow(training_set_over) 
final_model <- mlr::train(learner, task, subset = subset_indices, weights = NULL)
final_model

# Evaluate the model on the test set
predictions <- predict(final_model, newdata = test_set_over)
predictions
cm_dt_tuned <- confusionMatrix(test_set_over$Churn, predictions$data$response, positive="1")
cm_dt_tuned
f1_score_dt_tuned <- cm_dt_tuned$byClass["F1"]
f1_score_dt_tuned

###ROC curve
predictions$data$response <- as.numeric(as.character(predictions$data$response))
pred_prob_dt_tuned <- predictions$data$response
roc_dt_tuned <- roc(test_set_over$Churn, pred_prob_dt_tuned)
plot(roc_dt_tuned, main = "ROC Curve",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate")

#AUC
auc_dt_tuned <- mlr::performance(predictions, measures = measure)
auc_dt_tuned

###################################################

#MODEL EXPERIMENTATION 3: RANDOM FOREST

##################################################

#EXPERIMENT1 - ORIGINAL DATASET
#RANDOM FOREST

##################################################
rf_training_set <- training_set
rf_test_set <- test_set
rf_training_set_over <- training_set_over
rf_test_set_over <- test_set_over
##################################################

library(randomForest)
rf <- randomForest(Churn ~., data = rf_training_set, proximity = TRUE, type = "classification")
print(rf)
attributes(rf)
plot(rf)

#Training test result
p1 <- predict(rf, rf_training_set)
p1
cm_rf <- confusionMatrix(p1, rf_training_set$Churn, positive = "1")
cm_rf
f1_score_train_rf <- cm_rf$byClass["F1"]
f1_score_train_rf

#Test set result
p2 <- predict(rf, rf_test_set)
p2
rf_model <- confusionMatrix(p2, rf_test_set$Churn, positive = "1")
rf_model
f1_score_rf <- rf_model$byClass["F1"]
f1_score_rf

###ROC curve
pred_class_rf <- predict(rf,rf_test_set, type="response")
pred_class_rf
pred_class_rf_num <- as.numeric(pred_class_rf)

roc_obj_rf <- roc(rf_test_set$Churn, pred_class_rf_num)
plot(roc_obj_rf, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

#AUC
auc_rf <- auc(roc_obj_rf)
auc_rf <- round(auc_rf, 4)
auc_rf

##################################################

#EXPERIMENT2 - OVERSAMPLING DATASET
#RANDOM FOREST

##################################################
rf_over <- randomForest(Churn ~., data = rf_training_set_over, proximity = TRUE, type = "classification")
print(rf_over)
attributes(rf_over)
plot(rf_over)

#Training set result
p1_over <- predict(rf_over, rf_training_set_over)
p1_over
cm_rf_train_over <- confusionMatrix(p1_over, rf_training_set_over$Churn, positive = "1")
cm_rf_train_over
f1_score_rf_train_over <- cm_rf_train_over$byClass["F1"]
f1_score_rf_train_over

#Test set result
p2_over <- predict(rf_over, rf_test_set_over)
p2_over
cm_rf_over <- confusionMatrix(p2_over, rf_test_set_over$Churn, positive = "1") 
cm_rf_over
f1_score_rf_over <- cm_rf_over$byClass["F1"]
f1_score_rf_over

###ROC curve
pred_class_rf_over <- predict(rf_over,rf_test_set_over, type="response")
pred_class_rf_over
pred_class_rf_over_num <- as.numeric(pred_class_rf_over)

roc_obj_rf_over <- roc(rf_test_set_over$Churn, pred_class_rf_over_num)
plot(roc_obj_rf_over, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

#AUC
auc_rf_over <- auc(roc_obj_rf_over)
auc_rf_over <- round(auc_rf_over, 4)
auc_rf_over

###Variance Importance
varImp(rf_over)
varImpPlot(rf_over)

##################################################

#EXPERIMENT3 - OVERSAMPLING DATASET
#Random Forest
#Cross Validation
#HYPERPARAMETER TUNING

##################################################
library(mlr)

task_rf <- makeClassifTask(data = rf_training_set_over, target = "Churn")
learner_rf <- makeLearner("classif.randomForest", predict.type = "prob")

param_grid_rf <- makeParamSet(
  makeIntegerParam("mtry", lower = 1, upper = 3),
  makeIntegerParam("ntree", lower = 100, upper = 150),
  makeIntegerParam("nodesize", lower = 5, upper = 10)
)

cv_rf <- makeResampleDesc("CV", iters = 10)

# Define the performance measure
measure_rf <- mlr::auc

# Perform grid search with cross-validation
tuner_rf <- makeTuneControlGrid()
result_rf <- tuneParams(learner_rf, task_rf, resampling = cv_rf, 
                        measures = measure_rf, par.set = param_grid_rf, control = tuner_rf)

# Get the best hyperparameters
best_index_rf <- which.max(result_rf$y)
best_index_rf
best_params_rf <- result_rf$x[[best_index_rf]]
best_params_rf
result_rf$x

#Use the best parameters
best_params_rf <- list(mtry = 3, ntree = 106, nodesize = 5)

# Train the model with the best hyperparameters
learner_rf <- makeLearner("classif.randomForest", predict.type = "prob", mtry = best_params_rf$mtry, ntree = best_params_rf$ntree, nodesize = best_params_rf$nodesize)
learner_rf
set.seed(123)
final_model_rf <- train(learner_rf, task_rf)
final_model_rf

# Evaluate the model on the test set
predictions_rf <- predict(final_model_rf, newdata = rf_test_set_over)
predictions_rf
cm_rf_tuned <- confusionMatrix(rf_test_set_over$Churn, predictions_rf$data$response, positive = "1")
cm_rf_tuned
f1_score_rf_tuned <- cm_rf_tuned$byClass["F1"]
f1_score_rf_tuned

###ROC curve
predictions_rf$data$response <- as.numeric(as.character(predictions_rf$data$response))
pred_prob_rf_tuned <- predictions_rf$data$response
pred_prob_rf_tuned
roc_rf_tuned <- roc(test_set_over$Churn, pred_prob_rf_tuned)
plot(roc_rf_tuned, main = "ROC Curve", xlab = "False Positive Rate", ylab = "True Positive Rate")

#AUC
auc_rf_tuned <- mlr::performance(predictions_rf, measures = measure_rf)
auc_rf_tuned

##################################################

#Comparison with Literature Review

##################################################
library(ggplot2)

# Create a data frame with the model names and their corresponding accuracy values
data <- data.frame(
  Reference = c("In this study", "Alshamsi (2022)", "Alshamsi (2022)", "Alshamsi (2022)", "Awasthi (2022)", "Awasthi (2022)", "Awasthi (2022)", "Awasthi (2022)", "Awasthi (2022)", "Kiran et al. (2022)", "Kiran et al. (2022)", "Kiran et al. (2022)"),
  Model = c("RF with hyperparameter tuning", "RF", "LR", "DT", "Stacking", "DT", "KNN", "SVM", "RF", "LDA", "NB", "SVM"),
  Accuracy = c(96.71, 93.50, 80.50, 84.80, 98.00, 97.02, 97.04, 89.87, 90.58, 92.53, 90.31, 86.94)
)

# Create the bar plot
ggplot(data, aes(x = Reference, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Reference", y = "Accuracy", title = "Comparison of Model Accuracies") +
  theme_bw() +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = paste0(Accuracy, "%")), position = position_dodge(width = 0.9), vjust = -0.5, size = 3)
