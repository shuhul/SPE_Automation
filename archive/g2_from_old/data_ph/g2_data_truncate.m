load('ch6_f004o006_e1_2.pturawdata.mat')
%%
data=data(1:length(data)*3/4,:);
save(strcat('ch6_f004o006_e1_trunc.ptu','rawdata.mat'),'-v7.3','data');