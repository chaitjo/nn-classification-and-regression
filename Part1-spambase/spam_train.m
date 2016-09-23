P_train = P_train_std; Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
hiddenLayerSize = [10];
net = fitnet(hiddenLayerSize);
net.trainFcn = 'traingd'; 
net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
net.trainParam.epochs =100;
net.trainParam.max_fail = 25;
[net tr] = train(net,P_train,T_train);
[fields N] = size(T_test);
neuralnetscore = net(Val.P);
missclassification_rate = sum(0.5*abs(T_test - neuralnetscore))/N;
