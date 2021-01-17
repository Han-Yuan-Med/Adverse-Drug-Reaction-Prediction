#LR Ensemble
train_positive<-read.csv("train_positive.csv")
train_positive<-train_positive[,-1]
train_negative<-read.csv("train_negative.csv")
train_negative<-train_negative[,-1]
test_positive<-read.csv("test_positive.csv")
test_positive<-test_positive[,-1]
test_negative<-read.csv("test_negative.csv")
test_negative<-test_negative[,-1]

voteslist<-list() 
fitlist<-list() 
for(i in 1:200){ 
  set.seed(i) 
  negativedata_sample<-train_negative[sample(nrow(train_negative),100),] 
  traindata<-rbind(train_positive,negativedata_sample) 
  fit<- glm(ADR~., data = traindata, family = binomial())
  fitlist[[i]]<-fit 
  print(i)
} 

testdata<-rbind(test_positive,test_negative) 
ensemblematrix<-matrix(nr=9223,nc=1)
for (i in 1:200) {
  pre1<-predict(fitlist[[i]],testdata,type="response") 
  ensemblematrix<-cbind(ensemblematrix, pre1)
  print(i)
}

ensembleresult_1<-rowSums(ensemblematrix[,-1])
ensembleresult_1 = ensembleresult_1 / 200

length(which(ensembleresult_1[1:12]>=0.5))
length(which(ensembleresult_1[13:9223]<=0.5))

write.csv(ensembleresult_1,"LRensemble.csv")