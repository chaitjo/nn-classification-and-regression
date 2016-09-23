load ('spam_data.mat')
Val.P = P_test; 
Val.T = T_test;

[P_train_n, PS_train_minmax] = mapminmax(P_train);
Val_n.P = mapminmax('apply', P_test, PS_train_minmax);
Val_n.T = T_test;

[P_train_std, PS_train_std] = mapstd(P_train);
Val_std.P = mapstd('apply', P_test, PS_train_std);
Val_std.T = T_test;
