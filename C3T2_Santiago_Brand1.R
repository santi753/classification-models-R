################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
library(caret)
library(gbm)
library(C50)
library(corrplot)
library(readr)
library(mlbench)

##############
# Import data 
##############

setwd('C:/Users/santi/OneDrive/Documentos/CURSO DATA ANALYTICS/Cours III/Course 3 Task 2/Ejercicio del Curso 3/Data')

##-- Load Train/Existing data (Dataset 1) --##

CompleteResponses <- read.csv("CompleteResponses.csv", stringsAsFactors = FALSE)
str(CompleteResponses)

##--- Load Predict/New data (Dataset 2) [dv = NA, 0, or Blank] ---##

IncompleteResponses <- read.csv("SurveyIncomplete.csv", stringsAsFactors = FALSE)
str(IncompleteResponses)

################
# Evaluate data
################

##--- CompleteResponses ---##

head(CompleteResponses)
tail(CompleteResponses)

##--- IncompleteResponses ---##

head(IncompleteResponses)
tail(IncompleteResponses)

#############
# Preprocess
#############

##--- CompleteResponses ---##

anyNA(CompleteResponses)
anyDuplicated(CompleteResponses)

##--- IncompleteResponses ---##

anyNA(IncompleteResponses)
anyDuplicated(IncompleteResponses)

#####################
# EDA/Visualizations
#####################

##--- CompleteResponses ---##

# statistics
summary(CompleteResponses)

# plots
hist(CompleteResponses$salary)
plot(CompleteResponses$brand, CompleteResponses$age)
plot(CompleteResponses$brand, CompleteResponses$salary)
plot(CompleteResponses$elevel, CompleteResponses$salary)
qqnorm(CompleteResponses$salary) 


##--- IncompleteResponses ---##

# statistics
summary(IncompleteResponses)

# plots
hist(IncompleteResponses$salary)
plot(IncompleteResponses$age, IncompleteResponses$salary)
plot(IncompleteResponses$elevel, IncompleteResponses$salary)
qqnorm(IncompleteResponses$salary)

##--- Combined datasets ---##

Allresponses <- rbind(CompleteResponses, IncompleteResponses)


################
# Sampling
################

# create 20% sample 
set.seed(123) # set random seed
CROOB20p <- CompleteResponses[sample(1:nrow(CompleteResponses), round(nrow(CompleteResponses)*.2),replace=FALSE),]
nrow(CROOB20p)
head(CROOB20p) # ensure randomness
summary(CROOB20p)

#######################
# Correlation analysis
#######################

corrAll <- cor(CompleteResponses)
corrAll
corrplot(corrAll, method = "circle")
corrplot(corrAll, order = "hclust") 

##################
# Train/test sets
##################

CROOB20p$brand <- as.factor(CROOB20p$brand)
CompleteResponses$brand <- as.factor(CompleteResponses$brand)

set.seed(123) 
inTraining <- createDataPartition(CompleteResponses$brand, p=0.75, list=FALSE)
oobTrain <- CompleteResponses[inTraining,]   
oobTest <- CompleteResponses[-inTraining,]   
nrow(oobTrain)
nrow(oobTest)


################
# Train control
################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1) 


###############
# Train models
###############

modelLookup('rf')
modelLookup('gbm')
modelLookup('C5.0')

## ------- RF ------- ##

set.seed(123)
oobRFfit <- train(brand~., data=oobTrain, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit <- train(brand~.,data=oobTrain,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfit
plot(oobRFfit)
varImp(oobRFfit)

##Random Forest 

##7424 samples
##6 predictor
##2 classes: '0', '1' 

##No pre-processing
##Resampling: Cross-Validated (5 fold, repeated 1 times) 
##Summary of sample sizes: 5938, 5940, 5939, 5940, 5939 
##Resampling results across tuning parameters:
  
##  mtry  Accuracy   Kappa    
##4     0.9170272  0.8239807
##5     0.9152752  0.8202743
##6     0.9133901  0.8163124

##Accuracy was used to select the optimal model using the largest value.
##The final value used for the model was mtry = 4.

##rf variable importance

##        Importance
##salary    100.0000
##age        72.6942
##credit      0.7755
##elevel      0.4856
##car         0.1052
##zipcode     0.0000

## ------- GBM ------- ##

set.seed(99)
oobGBMfit <- train(brand~., data=oobTrain, method="gbm", trControl=fitControl, verbose=FALSE)
oobGBMfit
varImp(oobGBMfit)

##Stochastic Gradient Boosting 

##7424 samples
##6 predictor
##2 classes: '0', '1' 

##No pre-processing
##Resampling: Cross-Validated (5 fold, repeated 1 times) 
##Summary of sample sizes: 5938, 5939, 5939, 5940, 5940 
##Resampling results across tuning parameters:
  
##  interaction.depth  n.trees  Accuracy   Kappa    
##1                   50      0.7270991  0.4243569
##1                  100      0.7273677  0.4229400
##1                  150      0.7264255  0.4201510
##2                   50      0.8353943  0.6524422
##2                  100      0.8840233  0.7576204
##2                  150      0.9053076  0.8001846
##3                   50      0.8821388  0.7549380
##3                  100      0.9031556  0.7972862
##3                  150      0.9214727  0.8341021

##Tuning parameter 'shrinkage' was held constant at a value of 0.1
##Tuning parameter 'n.minobsinnode' was held constant
##at a value of 10
##Accuracy was used to select the optimal model using the largest value.
##The final values used for the model were n.trees = 150, interaction.depth = 3,
##shrinkage = 0.1 and n.minobsinnode = 10.


##gbm variable importance

##          Overall
##salary  100.00000
##age      74.15158
##credit    1.72443
##car       0.17754
##elevel    0.01233
##zipcode   0.00000


## ------- C5.0 ------- ##

set.seed(101)
oobC50fit <- train(brand~., data=oobTrain, method="C5.0", trControl=fitControl)
oobC50fit
varImp(oobC50fit)

##C5.0 

##7424 samples
##6 predictor
##2 classes: '0', '1' 

##No pre-processing
##Resampling: Cross-Validated (5 fold, repeated 1 times) 
##Summary of sample sizes: 5939, 5939, 5940, 5939, 5939 
##Resampling results across tuning parameters:
  
##  model  winnow  trials  Accuracy   Kappa    
##rules  FALSE    1      0.9112345  0.8118427
##rules  FALSE   10      0.9148710  0.8181972
##rules  FALSE   20      0.9158138  0.8207180
##rules   TRUE    1      0.9147368  0.8194061
##rules   TRUE   10      0.9162181  0.8215402
##rules   TRUE   20      0.9174302  0.8245741
##tree   FALSE    1      0.9113695  0.8120946
##tree   FALSE   10      0.9154096  0.8204509
##tree   FALSE   20      0.9191808  0.8283291
##tree    TRUE    1      0.9139288  0.8175321
##tree    TRUE   10      0.9170261  0.8238615
##tree    TRUE   20      0.9194508  0.8288039

##Accuracy was used to select the optimal model using the largest value.
##The final values used for the model were trials = 20, 
##model = tree and winnow = TRUE.

##C5.0 variable importance

##        Overall
##salary   100.00
##car       85.51
##age       85.51
##credit    30.51
##zipcode    0.00
##elevel     0.00



##################
# Model selection
##################

#-- CompleteResponses --# 

Selectmodel <- resamples(list(rf=oobRFfit, gbm=oobGBMfit, C5.0=oobC50fit))

# output summary metrics for tuned models 

summary(Selectmodel)

##Call:
##  summary.resamples(object = Selectmodel)

##Models: rf, gbm, C5.0 
##Number of resamples: 5 

##Accuracy 
##Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
##rf   0.9111709 0.9164420 0.9171717 0.9170272 0.9185185 0.9218329    0
##gbm  0.9090909 0.9172275 0.9238544 0.9214727 0.9278976 0.9292929    0
##C5.0 0.9104377 0.9138047 0.9218329 0.9194508 0.9225589 0.9286195    0

##Kappa 
##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
##rf   0.8117897 0.8214370 0.8252160 0.8239807 0.8272225 0.8342382    0
##gbm  0.8084287 0.8272031 0.8375625 0.8341021 0.8468320 0.8504843    0
##C5.0 0.8088934 0.8190486 0.8328397 0.8288039 0.8347543 0.8484834    0



##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(oobGBMfit, "oobGBMfit.rds")  

# load and name model
SelectModelgbm <- readRDS("oobGBMfit.rds")


############################
# Predict testSet/validation
############################

# predict with RF
GBMPred1 <- predict(oobGBMfit, oobTest)
# performace measurment
postResample(GBMPred1, oobTest$brand)
# plot predicted verses actual
plot(GBMPred1, oobTest$brand)
GBMPred1
plot(GBMPred1)


###############################
# Predict new data (Dataset 2)
###############################

# predict for new dataset with no values for DV

GBMPredOOB <- predict(oobGBMfit, IncompleteResponses)

head(GBMPredOOB)
plot(GBMPredOOB)

###############################
# Predict combined dataset
###############################

GBMPredcombinedOOB <- predict(oobGBMfit, Allresponses)

head(GBMPredcombinedOOB)
plot(GBMPredcombinedOOB)
summary(GBMPredcombinedOOB)
