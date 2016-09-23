%P_train = P_train_n; T_train = T_train_n; Val = Val_n; %%% Use this line to use mnmx preprocessing on the data. IMPORTANT: Run preprocess.m first 
P_train = P_train_std; T_train = T_train_std; Val = Val_std; %%% Use this line to use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
hiddenLayerSize = [10];
net = fitnet(hiddenLayerSize);
net.trainFcn = 'traingd';
net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
net.trainParam.epochs =200;
net.trainParam.max_fail = 50;
[net tr] = train(net,P_train,T_train);
[fields N] = size(T_test);

est = net(Val.P);
%est = postmnmx(est,mint,maxt); %%% Use this line if you use mnmx preprocessing on the data. IMPORTANT: Uncomment the corresponding line above
est = mapstd('apply', est, TS_train_std); %%% Use this line if you use STD or PCA preprocessing on the data. IMPORTANT: Uncomment the corresponding line above 

RMS_Error = perform(net, T_test, est);