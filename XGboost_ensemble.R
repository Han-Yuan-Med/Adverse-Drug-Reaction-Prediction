#XG Ensemble
train_positive<-read.csv("train_positive.csv")
train_positive<-train_positive[,-1]
train_negative<-read.csv("train_negative.csv")
train_negative<-train_negative[,-1]
test_positive<-read.csv("test_positive.csv")
test_positive<-test_positive[,-1]
test_negative<-read.csv("test_negative.csv")
test_negative<-test_negative[,-1]

#XGBoost
train_data<-rbind(train_positive, train_negative)
test_data<-rbind(test_positive, test_negative)
testset1 <- data.matrix(test_data[,c(1:188)]) 
testset2 <- Matrix(testset1,sparse=T) 
testset3 <- test_data[,189]
testset4 <- list(data=testset2,label=testset3) 
dtest <- xgb.DMatrix(data = testset4$data, label = testset4$label) 
voteslist<-list() 
fitlist<-list() 

for(i in 1:200){ 
  set.seed(i) 
  negativedata_sample<-train_negative[sample(nrow(train_negative),100),] 
  train_data<-rbind(train_positive,negativedata_sample) 
  traindata1 <- data.matrix(train_data[,c(1:188)]) 
  traindata2 <- Matrix(traindata1,sparse=T) 
  traindata3 <- train_data[,189]
  traindata4 <- list(data=traindata2,label=traindata3) 
  dtrain <- xgb.DMatrix(data = traindata4$data, label = traindata4$label) 
  fit <- xgboost(data = dtrain,max_depth=200, eta=0.5,  objective='binary:logistic', nround=50)
  fitlist[[i]]<-fit 
  print(i)
} 

ensemblematrix<-matrix(nr=9223,nc=1)
for (i in 1:200) {
  pre1<-predict(fitlist[[i]],newdata = dtest) 
  ensemblematrix<-cbind(ensemblematrix, pre1)
  print(i)
}

ensembleresult_1<-rowSums(ensemblematrix[,-1])
ensembleresult_1 = ensembleresult_1 / 200

length(which(ensembleresult_1[1:12]>=0.5))
length(which(ensembleresult_1[13:9223]<=0.5))

write.csv(ensembleresult_1,"XGBoostensemble.csv")