% convert double to cell type
X = con2seq(mobilepenetrationrateexp);
T = con2seq(mobiledatausageexp);

% for multi-step ahead prediction
N = 5;
 
% input and target series divided into 2 groups. 1st:training ; 2nd:simulation
inputSeries = X(1:end-N);
targetSeries = T(1:end-N);
inputSeriesVal = X(end-N+1:end);
targetSeriesVal = T(end-N+1:end);
 
% neural net architecture
delay = 2;
neuronsHiddenLayer = 10;
net = narxnet(1:delay,1:delay,neuronsHiddenLayer);

% train the neural net
[Xs,Xi,Ai,Ts] = preparets(net,inputSeries,{},targetSeries);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai); 
perf = perform(net,Ts,Y); % one-step

% now multi-step prediction
[Xs1,Xio,Aio] = preparets(net,inputSeries(1:end-delay),{},targetSeries(1:end-delay));
[Y1,Xfo,Afo] = net(Xs1,Xio,Aio);
[netc,Xic,Aic] = closeloop(net,Xfo,Afo);
[yPred,Xfc,Afc] = netc(inputSeriesVal,Xic,Aic);
multiStepPerformance = perform(net,yPred,targetSeriesVal);
view(netc)
figure;
plot([cell2mat(targetSeries),nan(1,N); 
	nan(1,length(targetSeries)),cell2mat(yPred); 
	nan(1,length(targetSeries)),cell2mat(targetSeriesVal)]')
legend('Original Targets','Network Predictions','Expected Outputs')
xlabel('Time[Q]');
ylabel('Mobile Data Usage')