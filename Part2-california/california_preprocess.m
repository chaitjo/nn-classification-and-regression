load ('california_data.mat')
Val.P = P_test; 
Val.T = T_test;

[P_train_n, PS_train_minmax] = mapminmax(P_train);
Val_n.P = mapminmax('apply', P_test, PS_train_minmax);	
[T_train_n TS_train_minmax] = mapminmax(T_train);
Val_n.T = mapminmax('apply', T_test, TS_train_minmax);

[P_train_std, PS_train_std] = mapstd(P_train);
Val_std.P = mapstd('apply', P_test, PS_train_std);
[T_train_std TS_train_std] = mapstd(T_train);
Val_std.T = mapstd('apply', T_test, TS_train_std);
