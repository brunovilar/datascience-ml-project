# Practical Machine Learning -- Course Project
Bruno Vilar  
22-08-2015  




# Introduction

This report presents the development of a model to perform Human Activity Prediction based on [HAR dataset](http://groupware.les.inf.puc-rio.br/har). This task is part of the [Practical Machine Learning](https://www.coursera.org/course/predmachlearn) course, from the [Data Science Specialization](https://www.coursera.org/specialization/jhudatascience/1) on [Coursera](https://www.coursera.org).

---

# Loading and Separating Data

In the context of the course, two files were provided: *pml-training.csv* and *pml-testing.csv*.


```r
labeledDataset = read.csv(file = "./data/pml-training.csv", stringsAsFactors=FALSE)
unlabledDataset = read.csv(file = "./data/pml-testing.csv")
```

As we do not have access to the labels of the second dataset, we need to creat a validation set from the *pml-training.csv*. The test set will be automatically extracted by [Caret package](https://topepo.github.io/caret/) during repeated 10 fold cross validation. The *pml-testing.csv* is considered an alternative validation, performed by Coursera's website. For this purpose, we divided the training into training and validation sets, respectivelly, with 75% and 25% of the rows.


```r
courseraValidation = unlabledDataset

trainingIndex = createDataPartition(labeledDataset$classe, p = 0.75, list = FALSE)[, 1]
training = labeledDataset[trainingIndex,]
validation = labeledDataset[-trainingIndex,]

df = data.frame(Training=nrow(training), Validation=nrow(validation))
datatable(df, rownames=FALSE, options = list( searching = FALSE, pageLength = 5 ))
```

<!--html_preserve--><div id="htmlwidget-7847" style="width:100%;height:auto;" class="datatables"></div>
<script type="application/json" data-for="htmlwidget-7847">{"x":{"data":[[14718],[4904]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th>Training</th>\n      <th>Validation</th>\n    </tr>\n  </thead>\n</table>","options":{"searching":false,"pageLength":5,"columnDefs":[{"className":"dt-right","targets":[0,1]}],"order":[],"autoWidth":false,"orderClasses":false,"lengthMenu":[5,10,25,50,100]},"callback":null,"filter":"none"},"evals":[]}</script><!--/html_preserve-->

---

# Data Preparation

As we want to avoid gaining any information from the validation set, only the training set will be analyzed. We start by analyzing its structure. There are columns with a high number of NA and incorrect values (empty or "#DIV/0!"). The list can be seen below.


```r
NAs = sapply(colnames(training), function(item){
    sum(as.numeric(is.na(training[, item])))/nrow(training)
})

blankOrWrong = sapply(colnames(training), function(item){
    sum(as.numeric((training[, item]) == "" | training[, item] == "#DIV/0!"))/nrow(training)
})

results = data.frame(Column=names(NAs), NAs=NAs, BlankOrWrong=blankOrWrong)

datatable(results, rownames=FALSE, options = list( searching = TRUE, pageLength = 15 ))
```

<!--html_preserve--><div id="htmlwidget-233" style="width:100%;height:auto;" class="datatables"></div>
<script type="application/json" data-for="htmlwidget-233">{"x":{"data":[["X","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window","roll_belt","pitch_belt","yaw_belt","total_accel_belt","kurtosis_roll_belt","kurtosis_picth_belt","kurtosis_yaw_belt","skewness_roll_belt","skewness_roll_belt.1","skewness_yaw_belt","max_roll_belt","max_picth_belt","max_yaw_belt","min_roll_belt","min_pitch_belt","min_yaw_belt","amplitude_roll_belt","amplitude_pitch_belt","amplitude_yaw_belt","var_total_accel_belt","avg_roll_belt","stddev_roll_belt","var_roll_belt","avg_pitch_belt","stddev_pitch_belt","var_pitch_belt","avg_yaw_belt","stddev_yaw_belt","var_yaw_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","var_accel_arm","avg_roll_arm","stddev_roll_arm","var_roll_arm","avg_pitch_arm","stddev_pitch_arm","var_pitch_arm","avg_yaw_arm","stddev_yaw_arm","var_yaw_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","kurtosis_roll_arm","kurtosis_picth_arm","kurtosis_yaw_arm","skewness_roll_arm","skewness_pitch_arm","skewness_yaw_arm","max_roll_arm","max_picth_arm","max_yaw_arm","min_roll_arm","min_pitch_arm","min_yaw_arm","amplitude_roll_arm","amplitude_pitch_arm","amplitude_yaw_arm","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","kurtosis_roll_dumbbell","kurtosis_picth_dumbbell","kurtosis_yaw_dumbbell","skewness_roll_dumbbell","skewness_pitch_dumbbell","skewness_yaw_dumbbell","max_roll_dumbbell","max_picth_dumbbell","max_yaw_dumbbell","min_roll_dumbbell","min_pitch_dumbbell","min_yaw_dumbbell","amplitude_roll_dumbbell","amplitude_pitch_dumbbell","amplitude_yaw_dumbbell","total_accel_dumbbell","var_accel_dumbbell","avg_roll_dumbbell","stddev_roll_dumbbell","var_roll_dumbbell","avg_pitch_dumbbell","stddev_pitch_dumbbell","var_pitch_dumbbell","avg_yaw_dumbbell","stddev_yaw_dumbbell","var_yaw_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm","kurtosis_roll_forearm","kurtosis_picth_forearm","kurtosis_yaw_forearm","skewness_roll_forearm","skewness_pitch_forearm","skewness_yaw_forearm","max_roll_forearm","max_picth_forearm","max_yaw_forearm","min_roll_forearm","min_pitch_forearm","min_yaw_forearm","amplitude_roll_forearm","amplitude_pitch_forearm","amplitude_yaw_forearm","total_accel_forearm","var_accel_forearm","avg_roll_forearm","stddev_roll_forearm","var_roll_forearm","avg_pitch_forearm","stddev_pitch_forearm","var_pitch_forearm","avg_yaw_forearm","stddev_yaw_forearm","var_yaw_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z","classe"],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0,0,0,0,0,0,0,0,0,0,0,0,0,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0,0,0,0,0,0,0,0,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0,0,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0,0.979209131675499,0.979209131675499,0,0,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0.979209131675499,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0.979684739774426,0.980975676042941,1,0.979684739774426,0.980975676042941,1,null,null,0.979684739774426,null,null,0.979684739774426,null,null,0.979684739774426,null,null,null,null,null,null,null,null,null,null,0,0,0,0,0,0,0,0,0,0,0,0,0,null,null,null,null,null,null,null,null,null,null,0,0,0,0,0,0,0,0,0,0.983285772523441,0.983421660551705,0.979752683788558,0.983285772523441,0.983421660551705,0.979752683788558,null,null,null,null,null,null,null,null,null,0,0,0,0.979412963717896,0.979277075689632,1,0.979412963717896,0.979277075689632,1,null,null,0.979412963717896,null,null,0.979412963717896,null,null,0.979412963717896,0,null,null,null,null,null,null,null,null,null,null,0,0,0,0,0,0,0,0,0,0,0,0,0.9838293246365,0.983897268650632,1,0.9838293246365,0.983897268650632,1,null,null,0.9838293246365,null,null,0.9838293246365,null,null,0.9838293246365,0,null,null,null,null,null,null,null,null,null,null,0,0,0,0,0,0,0,0,0,0]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th>Column</th>\n      <th>NAs</th>\n      <th>BlankOrWrong</th>\n    </tr>\n  </thead>\n</table>","options":{"searching":true,"pageLength":15,"columnDefs":[{"className":"dt-right","targets":[1,2]}],"order":[],"autoWidth":false,"orderClasses":false,"lengthMenu":[10,15,25,50,100]},"callback":null,"filter":"none"},"evals":[]}</script><!--/html_preserve-->

Due to the high number of NAs and fields with blank or wrong values (more than 97%) in some columns, is not viable to impute the values. Based on this, we remove the columns from all datasets, including validation and test sets, without looking their content.


```r
columnsToRemove = c( row.names(results[which(results$NAs > 0.9), ]), row.names(results[which(results$BlankOrWrong > 0.9), ]))

training = training[, !(names(training) %in% columnsToRemove)]
validation = validation[, !(names(validation) %in% columnsToRemove)]
courseraValidation = courseraValidation[, !(names(courseraValidation) %in% columnsToRemove)]
```

Originally we though about using the timestamp information by extracting features such as week day and hour of the day as a indicative of the users' routines. However, the number of values are limited (as shown below) and it is better for the model to rely only on sensors to recognize the human activities.


```r
dateTime = strptime(as.character(training$cvtd_timestamp), "%d/%m/%Y %H:%M")

df = data.frame(Columns=c("Month", "Day of Month", "Day of Week", "Hour"),
                Values=c(
    paste(unique(dateTime$mon+1), collapse = ", "),
    paste(unique(dateTime$mday), collapse = ", "),
    paste(unique(dateTime$wday), collapse = ", "),
    paste(unique(dateTime$hour), collapse = ", ")
))

datatable(df, rownames=FALSE, options = list( searching = FALSE, pageLength = 5 ))
```

<!--html_preserve--><div id="htmlwidget-3390" style="width:100%;height:auto;" class="datatables"></div>
<script type="application/json" data-for="htmlwidget-3390">{"x":{"data":[["Month","Day of Month","Day of Week","Hour"],["12, 11","5, 2, 28, 30","1, 5, 3","11, 14, 13, 17"]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th>Columns</th>\n      <th>Values</th>\n    </tr>\n  </thead>\n</table>","options":{"searching":false,"pageLength":5,"order":[],"autoWidth":false,"orderClasses":false,"lengthMenu":[5,10,25,50,100]},"callback":null,"filter":"none"},"evals":[]}</script><!--/html_preserve-->

As consequence, we removed timestamp columns (*raw_timestamp_part_1*, *raw_timestamp_part_2* and *cvtd_timestamp*) as well as columns that are used for internal control of the dataset and are not related to the movements (*X*, *new_window* and *num_window*) -- see page 16 of [slides](http://groupware.les.inf.puc-rio.br/public/2012.SBIA.Ugulino.WearableComputing-Presentation.pdf) for more details.


```r
columnsToRemove = c('raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'X', 'new_window', 'num_window')

training = training[, !(names(training) %in% columnsToRemove)]
validation = validation[, !(names(validation) %in% columnsToRemove)]
courseraValidation = courseraValidation[, !(names(courseraValidation) %in% columnsToRemove)]
```

To finish the dataset preparation, the character columns are converted to factors.


```r
training$user_name = as.factor(training$user_name)
training$classe = as.factor(training$classe)

validation$user_name = as.factor(validation$user_name)
validation$classe = as.factor(validation$classe)

courseraValidation$user_name = as.factor(courseraValidation$user_name)
```

At the end, 106 columns were removed. The resulting structure is the following:


```r
datatable(data.frame(Column=sort(colnames(training))), rownames=FALSE, options = list( searching = TRUE, pageLength = 10 ))
```

<!--html_preserve--><div id="htmlwidget-1441" style="width:100%;height:auto;" class="datatables"></div>
<script type="application/json" data-for="htmlwidget-1441">{"x":{"data":[["accel_arm_x","accel_arm_y","accel_arm_z","accel_belt_x","accel_belt_y","accel_belt_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","classe","gyros_arm_x","gyros_arm_y","gyros_arm_z","gyros_belt_x","gyros_belt_y","gyros_belt_z","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z","pitch_arm","pitch_belt","pitch_dumbbell","pitch_forearm","roll_arm","roll_belt","roll_dumbbell","roll_forearm","total_accel_arm","total_accel_belt","total_accel_dumbbell","total_accel_forearm","user_name","yaw_arm","yaw_belt","yaw_dumbbell","yaw_forearm"]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th>Column</th>\n    </tr>\n  </thead>\n</table>","options":{"searching":true,"pageLength":10,"order":[],"autoWidth":false,"orderClasses":false},"callback":null,"filter":"none"},"evals":[]}</script><!--/html_preserve-->

---

# Creating Models

In this section we describe the creation of the models to predict the human activities. Applying the training set to the [Caret package](http://topepo.github.io/caret), we performed the following procedures:

 - Center and Scale of the numeric values as preprocessing steps;
 - 3 algorithms were tested:
    - [CART](https://topepo.github.io/caret/Tree_Based_Model.html); 
    - [Stochastic Gradient Boosting](https://topepo.github.io/caret/Boosting.html);    
    - [Random Forest](https://topepo.github.io/caret/Random_Forest.html);
 - For each algorithm a 10-fold cross with 5 repetitions were performed to select the best model based on parameter tunning;
 - The execution of the models were performed using 5 cores;
 - No down-sampling/up-sampling were applied. There is a difference on the number of classes (A:4185 B:2848 C:2567 D:2412 E:2706 ), but the imbalance is not exaggerated.

The configuration of the preprocessing steps, performed on all experiments, is presented below. 

```r
library(doMC)
registerDoMC(cores = 5)
control = trainControl(method="repeatedcv", number=10, repeats=5, preProcOptions = c("center", "scale"))
```

## CART


```r
rObjectFile = "rpartModel.rds"

if (file.exists(rObjectFile)) {
    cartModel = readRDS(rObjectFile)
} else {    
    cartModel = train(classe ~ ., data=training, method="rpart", trControl=control)
    saveRDS(cartModel, rObjectFile)
}
```

## Stochastic Gradient Boosting


```r
rObjectFile = "gbmModel.rds"

if (file.exists(rObjectFile)) {
    gbmModel = readRDS(rObjectFile)
} else {
    gbmModel = train(classe ~ ., data=training, method="gbm", trControl=control)
    saveRDS(gbmModel, rObjectFile)
}
```

## Random Forest


```r
rObjectFile = "rfModel.rds"

if (file.exists(rObjectFile)) {
    rfModel = readRDS(rObjectFile)
} else {    
    rfModel = train(classe ~ ., data=training, method="rf", trControl=control)
    saveRDS(rfModel, rObjectFile)
}
```

# Selecting and Assessing The Best Model

The main statistics from the models are summarized below.


```r
extractModelData = function(model, modelName){

    resultsSize = length(model$results$Accuracy)

    data.frame(Model=rep(modelName, resultsSize),
               Accuracy=model$results$Accuracy,
               Kappa=model$results$Kappa,
               AccuracySD=model$results$AccuracySD,
               KappaSD=model$results$KappaSD)    
}

models = rbind(
        extractModelData(cartModel, "CART"),    
        extractModelData(gbmModel, "GBM"),
        extractModelData(rfModel, "Random Forest")
    )

datatable(models, rownames=FALSE, options = list( searching = TRUE, pageLength = 10 ))
```

<!--html_preserve--><div id="htmlwidget-4277" style="width:100%;height:auto;" class="datatables"></div>
<script type="application/json" data-for="htmlwidget-4277">{"x":{"data":[["CART","CART","CART","GBM","GBM","GBM","GBM","GBM","GBM","GBM","GBM","GBM","Random Forest","Random Forest","Random Forest"],[0.502417343137768,0.445117903980671,0.320014642954085,0.751053992053709,0.855687149776954,0.894741220942358,0.821580110420318,0.906210676226546,0.941785044251919,0.854410391747501,0.931798018246428,0.961869648990362,0.991914704503752,0.992811240756131,0.988557505713849],[0.350440391948342,0.257810819480977,0.0543680382410672,0.684334941722224,0.817163733998677,0.866755977011309,0.774117331117109,0.881316621683666,0.926346910286407,0.815763061688427,0.913697814218387,0.951762338008385,0.989771195114503,0.990906309731766,0.985525099591897],[0.0142014985421729,0.0606644049917811,0.0391799047894189,0.0120227518559038,0.011447877925441,0.00871252739972929,0.0116224435326718,0.00823878589267505,0.00606032609364307,0.00993494614669752,0.00632997436431579,0.00433653028446463,0.0027591913307435,0.00237093325381416,0.00316098069674703],[0.0205912455342978,0.101992036685577,0.0595849748351974,0.0151394505492233,0.0144944387169813,0.0110325833514154,0.0147040914193207,0.0104165058773236,0.00766172571074186,0.0125564315583976,0.00800693047737858,0.00548499828278092,0.00349148122368023,0.00299981013016855,0.00399865139841838]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th>Model</th>\n      <th>Accuracy</th>\n      <th>Kappa</th>\n      <th>AccuracySD</th>\n      <th>KappaSD</th>\n    </tr>\n  </thead>\n</table>","options":{"searching":true,"pageLength":10,"columnDefs":[{"className":"dt-right","targets":[1,2,3,4]}],"order":[],"autoWidth":false,"orderClasses":false},"callback":null,"filter":"none"},"evals":[]}</script><!--/html_preserve-->

The table shows that Random Forest obtained the best model, including the second and third places. GBM took the second place on the overall results. The worst accuracy was from CART. The results are expected, since GBM and Random Forest ensemble results from multiple classifiers. 

## Selected Model

As result of the experiments, the model chose was Random Forest.


```r
bestModel = rfModel
bestModel
```

```
## Random Forest 
## 
## 14718 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 5 times) 
## Summary of sample sizes: 13246, 13247, 13247, 13247, 13245, 13245, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9919147  0.9897712  0.002759191  0.003491481
##   29    0.9928112  0.9909063  0.002370933  0.002999810
##   57    0.9885575  0.9855251  0.003160981  0.003998651
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```

To estimate the accuracy of the best model, we use our validation set, **created from the training set at the beginning of the report**. Since it was not used during the training, we expect to evaluate the accuracy on unseen data.

## Validating The Selected Model


```r
predictedValues = predict(bestModel, newdata = validation)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
confMatrix = confusionMatrix(predictedValues, testing$classe)
confMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    2    0    0    0
##          B    1  946    0    1    0
##          C    0    1  855    0    2
##          D    0    0    0  803    0
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9971, 0.9994)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9982          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9968   1.0000   0.9988   0.9978
## Specificity            0.9994   0.9995   0.9993   1.0000   1.0000
## Pos Pred Value         0.9986   0.9979   0.9965   1.0000   1.0000
## Neg Pred Value         0.9997   0.9992   1.0000   0.9998   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1929   0.1743   0.1637   0.1833
## Detection Prevalence   0.2847   0.1933   0.1750   0.1637   0.1833
## Balanced Accuracy      0.9994   0.9982   0.9996   0.9994   0.9989
```

As result, we estimate that the accuracy of our model fit is about 99%, while the concordance (Kappa index) was about 98.9%. The Confidence Interval was 95% (0.9973 to 0.9996).

## Check Out of Sample Error

The out of sample error rate can be calculated as follows:


```r
outOfSample = 1.00 - sum(diag(confMatrix$table)) / sum(confMatrix$table)
outOfSample
```

```
## [1] 0.001427406
```

This model was used to answer the part 1 of the Course Project and obtained 100% of the predictions using the *courseraValidation* set.


# Concluding Remarks

The model created has a great precision and could be trained in a relatively short time on a local machine. Considering those aspects, we describe some choices made:

 - PCA could have been used to to reduce the number of features on the sets. It was not used due to the satisfactory execution time on the local machine and the high number of attributes already removed during the preprocessing phase;
 - Random Forest does not require a 10-Fold Cross Validation during the training, considering its operating mode. It was performed in order to keep the same standard as the other models tested;
 - No combination of models were performed after training of the models considering that Random Forest is already a ensemble technique and its results were already satisfactory;
 - The imbalance of the classes were not big enough to reduce the model effectiveness. 
 
  


