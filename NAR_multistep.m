% convert double to cell type
T = con2seq(mobiledatausageexp);
 
% for multi-step ahead prediction
N = 5;
 
% input and target series divided into 2 groups. 1st:training ; 2nd:simulation
targetSeries = T(1:end-N);
targetSeriesVal = T(end-N+1:end);

% neural net architecture
delay = 20;
neuronsHiddenLayer = 20;
net = narnet(1:delay,neuronsHiddenLayer);
 
% train the neural net
[Xs,Xi,Ai,Ts] = preparets(net,{},{},targetSeries);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y); % one-step

% now multi-step prediction
[netc,Xic,Aic] = closeloop(net,Xf,Af);
[yPred,Xfc,Afc] = netc(targetSeriesVal,Xic,Aic);
multiStepPerformance = perform(net,yPred,targetSeriesVal);
view(netc)
figure;
plot([cell2mat(targetSeries),nan(1,N); 
	nan(1,length(targetSeries)),cell2mat(yPred); 
	nan(1,length(targetSeries)),cell2mat(targetSeriesVal)]')
legend('Original Targets','Network Predictions','Expected Outputs')
xlabel('Time [Q]');
ylabel('Mobile Data Usage');