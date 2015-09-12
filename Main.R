#install.packages("data.table")
library("data.table")
library("caret")
#install.packages("DMwR")
library("DMwR")

#install.packages("ROSE")
library("ROSE")

library(doMC)

#Set the number of cores to 6 in order to be used by caret
registerDoMC(cores = 5)

#Read testing and training datasets
testing = read.csv(file = "./data/pml-testing.csv")
training = read.csv(file = "./data/pml-training.csv", stringsAsFactors=FALSE)

#Understand the structure
str(training)

#See the summary of data
summary(training)

#Extracting Hour and Day of Week from date
#Training
training$DateTime = strptime(as.character(training$cvtd_timestamp), "%d/%m/%Y %H:%M")
training$Hour = training$DateTime$hour
training$WeekDay = training$DateTime$wday

#Testing
testing$DateTime = strptime(as.character(testing$cvtd_timestamp), "%d/%m/%Y %H:%M")
testing$Hour = testing$DateTime$hour
testing$WeekDay = testing$DateTime$wday


#Create a list of columns that should be removed due to:
# - Can not be imputed due to the high ammount of empty or incorrect values (e.g. "" or " #DIV/0!");
# - Are not directly used, such as timestamps and DateTime 
columnsToRemove = c("X", "kurtosis_yaw_belt", "skewness_yaw_belt", "amplitude_yaw_belt", "kurtosis_yaw_dumbbell", "skewness_yaw_dumbbell", "amplitude_yaw_dumbbell", "kurtosis_yaw_forearm", "skewness_yaw_forearm", "amplitude_yaw_forearm", "kurtosis_roll_belt", "kurtosis_roll_arm", "kurtosis_picth_belt", "skewness_roll_belt", "skewness_roll_belt.1", "max_yaw_belt", "min_yaw_belt", "skewness_roll_arm", "skewness_roll_arm", "skewness_pitch_arm", "skewness_yaw_arm", "kurtosis_roll_dumbbell", "kurtosis_picth_dumbbell", "skewness_roll_dumbbell", "skewness_pitch_dumbbell", "max_yaw_dumbbell", "kurtosis_roll_forearm", "kurtosis_picth_forearm", "skewness_roll_forearm", "skewness_pitch_forearm", "max_yaw_forearm", "min_yaw_forearm", "var_yaw_forearm", "stddev_roll_forearm", "var_roll_forearm", "avg_pitch_forearm", "stddev_pitch_forearm", "var_pitch_forearm", "avg_yaw_forearm", "stddev_yaw_forearm", "var_accel_dumbbell", "avg_roll_dumbbell", "stddev_roll_dumbbell", "var_roll_dumbbell", "max_roll_dumbbell", "max_picth_dumbbell", "min_roll_dumbbell", "min_pitch_dumbbell", "amplitude_roll_dumbbell", "min_pitch_arm", "min_yaw_arm", "amplitude_roll_arm", "amplitude_pitch_arm", "amplitude_yaw_arm", "max_roll_arm", "max_picth_arm", "max_yaw_arm", "min_roll_arm", "avg_pitch_arm", "stddev_pitch_arm", "var_pitch_arm", "avg_yaw_arm", "stddev_yaw_arm", "var_yaw_arm", "var_accel_arm", "avg_roll_arm", "stddev_roll_arm", "var_roll_arm", "stddev_pitch_belt", "var_pitch_belt", "avg_yaw_belt", "stddev_yaw_belt", "var_yaw_belt", "amplitude_roll_belt", "amplitude_pitch_belt", "var_total_accel_belt", "avg_roll_belt", "stddev_roll_belt", "var_roll_belt", "avg_pitch_belt", "max_roll_belt", "max_picth_belt", "min_roll_belt", "min_pitch_belt", "min_pitch_forearm", "amplitude_roll_forearm", "amplitude_pitch_forearm", "var_accel_forearm", "avg_roll_forearm", "max_roll_forearm", "max_picth_forearm", "min_roll_forearm", "avg_pitch_dumbbell", "stddev_pitch_dumbbell", "var_pitch_dumbbell", "avg_yaw_dumbbell", "stddev_yaw_dumbbell", "var_yaw_dumbbell", "amplitude_pitch_dumbbell", "kurtosis_picth_arm", "kurtosis_yaw_arm", "min_yaw_dumbbell",  "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "DateTime")

#Remove the colunms not used
training = training[, !(names(training) %in% columnsToRemove)]
testing = testing[, !(names(testing) %in% columnsToRemove)]


#Formating columsn
#Training
training$new_window = training$new_window == "yes"
training$classe = as.factor(training$classe)
training$user_name = as.factor(training$user_name)
#Testing
testing$new_window = testing$new_window == "yes"
testing$user_name = as.factor(testing$user_name)

table(training$classe)

#----
# Reducing the number of features
#----
#Training
predictors = colnames(training[, !(names(training) %in% c("classe", "new_window", "user_name") )])
columnIndex = which(colnames(training) %in% predictors)
preProcessModel = preProcess(training[, columnIndex], method = c("center", "scale", "pca"),  thresh = 0.95)
preProcessedTraining = predict(preProcessModel, training[, columnIndex])

#Testing
preProcessedTesting = predict(preProcessModel, testing[, columnIndex])

#Inserting non numeric columns
preProcessedTraining = cbind(preProcessedTraining, training[, (names(training) %in% c("classe", "new_window", "user_name"))])
preProcessedTesting = cbind(preProcessedTesting, testing[, (names(testing) %in% c("new_window", "user_name"))])
#----

control = trainControl(method="repeatedcv", number=10, sampling=smotest)
#model = train(classe ~ ., data=training, method="rf", sampling="smote", trControl=control)
#model = train(classe ~ ., data=training, method="gbm",  sampling="smote", preProcess = c("center", "scale"), trControl=control, probs=TRUE)
#model = train(classe ~ ., data=preProcessedTraining, method="gbm")


model = train(classe ~ ., data=training, method="gbm", trControl=control)

model
summary(model)
varImp(model)

predictions = predict(model, newdata=testing)

predictions


# pml_write_files(predictions)
# 
# 
# pml_write_files = function(x){
#     n = length(x)
#     for(i in 1:n){
#         filename = paste0("problem_id_",i,".txt")
#         write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
#     }
# }






