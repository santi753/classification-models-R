################
# Load packages
################

install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("tidyverse")
library(caret)
library(gbm)
library(C50)
library(corrplot)
library(readr)
library(mlbench)
library(tidyverse)
library(modelr)


##############
# Import data 
##############

setwd('C:/Users/santi/OneDrive/Documentos/CURSO DATA ANALYTICS/Cours III/Course 3 Task 3/data/productattributes')

##-- Load Train/Existing data (Dataset 1) --##

EP <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE)
str(EP)

##--- Load Predict/New data (Dataset 2) [dv = NA, 0, or Blank] ---##

NP <- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE)
str(NP)

##--- Combined datasets ---##

AllP <- rbind(EP, NP)

################
# Evaluate data
################

##--- Existingproducts ---##

head(EP)
tail(EP)

##--- Newproduct ---##

head(NP)
tail(NP)

#############
# Preprocess
#############

##--- Existingproducts ---##

anyNA(EP)
EP[is.na(EP)] <- 0
anyDuplicated(EP)
EP$ProductType <- as.factor(EP$ProductType)
EP$ProductNum <- NULL
EP$ProfitMargin <- NULL
EP$BestSellersRank <- NULL
EP_1 <- dummyVars( " ~ .", data = EP)
EP_2 <- data.frame(predict(EP_1, newdata = EP))
print(EP_2)


##--- Newproduct ---##

anyNA(NP)
anyDuplicated(NP)
NP$ProductType <- as.factor(NP$ProductType)
NP$ProductNum <- NULL
NP$ProfitMargin <- NULL
NP$BestSellersRank <- NULL
NP_1 <- dummyVars( " ~ .", data = NP)
NP_2 <- data.frame(predict(NP_1, newdata = NP))
print(NP_2)

#####################
# EDA/Visualizations
#####################

##--- Existingproducts ---##

# statistics
summary(EP)
unique(EP$ProductType)

# plots
hist(EP$Price)
qplot(EP$ProductType)
plot(EP$ProductType, EP$x5StarReviews)
plot(EP$ProductType, EP$Price)
plot(EP$ProductType, EP$Volume)
plot(EP$x5StarReviews, EP$Volume)
plot(EP$x4StarReviews, EP$Volume)
plot(EP$x3StarReviews, EP$Volume)
plot(EP$x2StarReviews, EP$Volume)
plot(EP$x1StarReviews, EP$Volume)
qqnorm(EP$Volume) 

par(mfrow=c(3,2), cex.main=2, cex.axis=1.5)
plot(EP$x5StarReviews, EP$Volume,  main = "Reviews vs Volume", xlab = "5StarReviews", ylab = "Volume", col="red", cex = 3)
plot(EP$x4StarReviews, EP$Volume, xlab = "4StarReviews", ylab = "Volume", col = "blue", cex = 3)
plot(EP$x3StarReviews, EP$Volume, xlab = "3StarReviews", ylab = "Volume", col = "green", cex = 3)
plot(EP$x2StarReviews, EP$Volume, xlab = "2StarReviews", ylab = "Volume", col = "brown", cex = 3)
plot(EP$x1StarReviews, EP$Volume, xlab = "1StarReviews", ylab = "Volume", col = "dark orange", cex = 3);


##--- Newproducts ---##

# statistics
summary(NP)

# plots
par(mfrow=c(1,1))
hist(NP$Price)
plot(NP$ProductType, NP$Price)
plot(NP$ProductType, NP$x5StarReviews)


#######################
# Correlation analysis
#######################

corrAll <- cor(EP_2)
corrAll
corrplot(corrAll, method = "circle")
corrplot(corrAll, order = "hclust") 

# find IVs that are highly corrected (ideally >0.90)
corrIV <- cor(EP_2)
# create object with indexes of highly corr features 
corrIVhigh <- findCorrelation(corrIV, cutoff=0.8) 
# print indexes of highly correlated attributes
corrIVhigh
# get var name of high corr IV
colnames(EP_2[corrIVhigh]) 
# remove highly correlated features, if applicable
EP_COR <- EP_2[-corrIVhigh] 
str(EP_COR)

# remove highly correlated features for EP_2 dataset
EP_2$x5StarReviews <- NULL



##################
# Train/test sets
##################

# -------- Existingproducts2 -------- #

set.seed(123) 
inTraining <- createDataPartition(EP$Volume, p=0.75, list=FALSE)
oobTrain <- EP[inTraining,]   
oobTest <- EP[-inTraining,]   
# verify number of obs 
nrow(oobTrain) 
nrow(oobTest)  


# -------- Existingproducts2 -------- #
set.seed(123) 
inTraining <- createDataPartition(EP_2$Volume, p=0.75, list=FALSE)
oobTrain_2 <- EP_2[inTraining,]   
oobTest_2 <- EP_2[-inTraining,]   
# verify number of obs 
nrow(oobTrain_2) 
nrow(oobTest_2)  

# -------- Existingproducts_cor -------- #
set.seed(123) 
inTraining <- createDataPartition(EP_COR$Volume, p=0.75, list=FALSE)
oobTrain_COR <- EP_COR[inTraining,]   
oobTest_COR <- EP_COR[-inTraining,]   
# verify number of obs 
nrow(oobTrain_COR) 
nrow(oobTest_COR)  

################
# Train control
################

# set cross validation
fitControl <- trainControl(method="repeatedcv", number=5, repeats=1) 

###############
# Train models
###############

modelLookup('svmLinear2')
modelLookup('rf')
modelLookup('gbm')

## ------- SVM ------- ##

# Existingproducts

set.seed(99)
oobSVMfit <- train(Volume~., data=oobTrain, method="svmLinear2", trControl=fitControl, verbose=FALSE, scale = FALSE)
oobSVMfit
varImp(oobSVMfit)

# Existingproducts2

set.seed(99)
oobSVMfit1 <- train(Volume~., data=oobTrain_2, method="svmLinear2", trControl=fitControl, verbose=FALSE, scale = FALSE)
oobSVMfit1
varImp(oobSVMfit1)

# Existingproducts_cor

set.seed(123)
oobSVMfit2 <- train(Volume~., data=oobTrain_COR, method="svmLinear2", trControl=fitControl, verbose=FALSE,scale = FALSE)
oobSVMfit2
varImp(oobSVMfit2)


## ------- RF ------- ##

# Existingproducts

set.seed(123)
oobRFfit <- train(Volume~., data=oobTrain, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit <- train(Volume~.,data=oobTrain,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfit
plot(oobRFfit)
varImp(oobRFfit)


# Existingproducts2 

set.seed(123)
oobRFfit1 <- train(Volume~., data=oobTrain_2, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit1 <- train(Volume~.,data=oobTrain_2,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfit1
plot(oobRFfit1)
varImp(oobRFfit1)

# Existingproducts_cor

set.seed(123)
oobRFfit2 <- train(Volume~., data=oobTrain_COR, method="rf", importance=T, trControl=fitControl)

# manual grid
rfGrid <- expand.grid(mtry=c(4,5,6))  
set.seed(123)
# fit
oobRFfit2 <- train(Volume~.,data=oobTrain_COR,method="rf",
                  importance=T,
                  trControl=fitControl,
                  tuneGrid=rfGrid)
oobRFfit2
plot(oobRFfit2)
varImp(oobRFfit2)

## ------- GBM ------- ##

# Existingproducts

set.seed(99)
oobGBMfit <- train(Volume~., data=oobTrain, method="gbm", trControl=fitControl, verbose=FALSE)
oobGBMfit
varImp(oobGBMfit)

# Existingproducts2 

set.seed(99)
oobGBMfit1 <- train(Volume~., data=oobTrain_2, method="gbm", trControl=fitControl, verbose=FALSE)
oobGBMfit1
varImp(oobGBMfit1)

# Existingproducts_cor

set.seed(99)
oobGBMfit2 <- train(Volume~., data=oobTrain_COR, method="gbm", trControl=fitControl, verbose=FALSE)
oobGBMfit2
varImp(oobGBMfit2)

##################
# Model selection
##################

#-- CompleteResponses --# 

Selectmodel <- resamples(list(svmLinear=oobSVMfit ,rf=oobRFfit, gbm=oobGBMfit))
Selectmodel1 <- resamples(list(svmLinear=oobSVMfit1 ,rf=oobRFfit1, gbm=oobGBMfit1))
Selectmodel2 <- resamples(list(svmLinear=oobSVMfit2,rf=oobRFfit2, gbm=oobGBMfit2))

# output summary metrics for tuned models 

summary(Selectmodel)
summary(Selectmodel1)
summary(Selectmodel2)

##--- Save/load top performing model ---##

# save top performing model after validation
saveRDS(oobRFfit2, "oobRFfit2.rds")  

# load and name model
SelectModelRF2 <- readRDS("oobRFfit2.rds")

############################
# Predict testSet/validation
############################

# predict with RF
RFPred2 <- predict(oobRFfit2, oobTest_COR)
# performace measurment
postResample(RFPred2, oobTest_COR$Volume)
# plot predicted verses actual
RFPred2
plot(RFPred2, oobTest_COR$Volume)


###############################
# Predict new data (Dataset 2)
###############################

# predict for new dataset with no values for DV

RFPredOOB <- predict(oobRFfit2, NP_2)
RFPredOOB
head(RFPredOOB)
plot(RFPredOOB)

#####################################
# Create new dataset with predictions
#####################################

NP_pred <- add_predictions(NP_2, oobRFfit2, var = "Volume_pred", type = NULL)
summary(NP_pred)
write.csv(NP_pred, file="C2.T3output.csv", row.names = TRUE)


