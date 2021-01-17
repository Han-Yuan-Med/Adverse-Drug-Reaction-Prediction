library(adabag) 
library(gbm) 
adrdata<-read.csv("adr_data_after_na.csv")
adrdata<-adrdata[,-c(1,2)]
positivedata=adrdata[which(adrdata$ADR==1),] 
negativedata=adrdata[-which(adrdata$ADR==1),] 
set.seed(2) 
positivedata_sample_train<-positivedata[sample(nrow(positivedata),28),] 
positivedata_sample_test<-positivedata[-sample(nrow(positivedata),28),] 
negativedata_sample_test<-negativedata[sample(nrow(negativedata),9211),] 
negativedata_sample_train<-negativedata[-sample(nrow(negativedata),9211),] 

voteslist<-list() 
fitlist<-list() 
for(i in 1:200){ 
  set.seed(i) 
  negativedata_sample<-negativedata_sample_train[sample(nrow(negativedata_sample_train),100),] 
  traindata<-rbind(positivedata_sample_train,negativedata_sample) 
  traindata$ADR<-as.factor(traindata$ADR) 
  fit<-boosting(ADR~.,data =traindata) 
  fitlist[[i]]<-fit 
  voteslist[[i]]<-fit$votes 
  print(i)
} 

testdata<-rbind(positivedata_sample_test,negativedata_sample_test) 
testdata$ADR<-as.factor(testdata$ADR) 
ensemblematrix<-matrix(nr=9223,nc=1)
for (i in 1:200) {
  pre1<-predict(fitlist[[i]],testdata,type="class") 
  ensemblematrix<-cbind(ensemblematrix, pre1$prob[,1])
  print(i)
}

ensembleresult_1<-rowSums(ensemblematrix[,-1])
ensembleresult_1 = ensembleresult_1 / 200

length(which(ensembleresult_1[1:12]<=0.5))
length(which(ensembleresult_1[13:9223]>=0.5))
write.csv(ensembleresult_1,"Adaprob@100@200.csv")

total_train<-rbind(positivedata_sample_train, negativedata_sample_train)
total_train$ADR<-as.factor(total_train$ADR) 
singlefit<-boosting(ADR~.,data = total_train)
cm<-singlefit$importance
order(cm)
cn<-cm[order(cm)]
pre2<-predict(singlefit,testdata,type="class") 
write.csv(pre2$prob,"SingleAdaprob.csv")