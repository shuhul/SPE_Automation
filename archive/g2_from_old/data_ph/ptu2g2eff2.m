%% Turning .ptu file into MATLAB readable data
clear 
clc
close all
global filename;
global tplot;
global csw;
global data;
  
% g2time=1e3; %nano second
% timebin = 1;  %nano second 
% 
% g2time=g2time*1e3;
% timebin=timebin*1e3;
fprintf('\n Reading .ptu file.')  
g2analyze; %Here is where the magic happens. open the file 'g2analyze.m' to see what happens under the hood
%toc

%% Cleaning and saving the data
fprintf('\n Now cleaning up the data and saving the raw data. \n');
tic
data((data(:,2)==0), :) = []; % remove data points with no time stamp.
save(strcat(filename,'rawdata.mat'),'-v7.3','data');
toc

%% Correlation calculatiion
fprintf('\n Now measuring correlation. \n');
load('C18_e1_1.pturawdata.mat')
filename='C18_e1_1.ptu';
datacopy=data;
tic;
g2time=100; %range of x axis of the g2 function (ns)                
timebin = 0.5;  %resolution of g2 function (ns) (usually 0.1-1 ns)  %%%%%%%%%%%%%%%%%%%%%%%%INPUT%%%%%%%%%%%%%%%%%%%%%

g2time=g2time*1e3;
timebin=timebin*1e3;

I=ceil(g2time/timebin);
tauindex=-I:I;
c=zeros(length(tauindex),1);
tau=tauindex*timebin/1000;


for i=1:length(data)-1
    if data(i,1)==0
        if data(i+1,1)==1
            temp=data(i+1,2)-data(i,2);
            ind=I+1+floor(temp/timebin);
            if temp<=g2time
                c(ind)=c(ind)+1;
            end
        end
    end
    if data(i,1)==1
        if data(i+1,1)==0
            temp=data(i+1,2)-data(i,2);
            ind=I-ceil(temp/timebin)+1;
            if temp<=g2time
                c(ind)=c(ind)+1;
            end
        end
    end
    
end
figure(1)
plot(tau,c)
title('raw data')
xlabel('time delay (ns)')
ylabel('counts')
toc
%% Removing afterflash
fprintf('\n Removing afterflash. \n');
tic;
tempsum=0;
tempind=0;
for i=1:length(c)
    if tau(i)>40
        if tau(i)<90
            tempsum=tempsum+c(i);
            tempind=tempind+1;
        end
    end
end
cavg=tempsum/tempind;

for i=1:length(data)-1
    if data(i,1)==0
        if data(i+1,1)==1
            temp=data(i+1,2)-data(i,2);
            ind=I+1+floor(temp/timebin);
            if temp<g2time
                if abs(tau(ind))>9 
                    if abs(tau(ind))<35
                        u=rand;
                        crat=poissrnd(cavg)/poissrnd(c(ind));
                        if u>crat
                            datacopy(i+1, 2) = 0;
                        end
                    end
                end
            end
        end
    end
    if data(i,1)==1
        if data(i+1,1)==0
            temp=data(i+1,2)-data(i,2);
            ind=I-ceil(temp/timebin)+1;
            if temp<=g2time
                if abs(tau(ind))>9 
                    if abs(tau(ind))<35
                        u=rand;
                        crat=poissrnd(cavg)/poissrnd(c(ind));
                        if u>crat
                            datacopy(i+1, 2) = 0;
                        end
                    end
                end
            end
        end
    end  
end
datacopy((datacopy(:,2)==0), :) = []; % remove data points with no time stamp.
data=datacopy;
save(strcat(filename,'afterflashremoveddata.mat'),'-v7.3','data');
%save(strcat('508nW_f2_e3.ptu','afterflashremoveddata.mat'),'data');
toc
%% Correlation calculation
fprintf('\n Now measuring correlation. \n');
tic;

I=ceil(g2time/timebin);
tauindex=-I:I;
c=zeros(length(tauindex),1);
tau=tauindex*timebin/1000;


for i=1:length(data)-1
    if data(i,1)==0
        if data(i+1,1)==1
            temp=data(i+1,2)-data(i,2);
            ind=floor(temp/timebin);
            if ind<=I
                c(I+ind+1)=c(I+ind+1)+1;
            end
        end
    end
    if data(i,1)==1
        if data(i+1,1)==0
            temp=data(i+1,2)-data(i,2);
            ind=ceil(temp/timebin);
            if ind<=I
                c(I-ind+1)=c(I-ind+1)+1;
            end
        end
    end
    
end
N=length(data);
TT=data(N,2);
N1=0;
N2=0;
for i=1:length(data)
    if data(i,1)==0
        N1=N1+1;
    else
        N2=N2+1;
    end
end
A=(N1*N2)*timebin/(TT);
g2=c/A;
toc
%% This part fits the g2 curve
X=tau;
Y=g2';

% fo = fitoptions('Method','NonlinearLeastSquares',...
%                'Lower',[0,-1,0.1,10,-0.5],...
%                'Upper',[inf,1,inf,inf,0.5], ...
%                'StartPoint', [1,1,10,5000,0]);
% ft = fittype('1 - b*((1+a)*exp(-1*abs(X-t0)/T1)-a*exp(-1*abs(X-t0)/T2))',...
%     'dependent',{'Y'},'independent',{'X'},...
%     'coefficients',{'a','b','T1','T2','t0'},'options',fo)

fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,-1,0.1,10],...
               'Upper',[inf,1,inf,inf], ...
               'StartPoint', [1,0,10,5000]);
ft = fittype('1 - b*((1+a)*exp(-1*abs(X)/T1)-a*exp(-1*abs(X)/T2))',...
    'dependent',{'Y'},'independent',{'X'},...
    'coefficients',{'a','b','T1','T2'},'options',fo)
myfit=fit(X',Y',ft)

tplot=min(tau):(max(tau)-min(tau))/10000:max(tau);



figure(2)
box on
plot(tau,g2,'color', [.8 .8 .8],'linewidth',1)
hold on
%plot(tplot,1 -myfit.b*((1+ myfit.a)*exp(-1*abs(tplot-myfit.t0)/myfit.T1)-myfit.a*exp(-1*abs(tplot-myfit.t0)/myfit.T2)),'black','linewidth',1.5);
plot(tplot,1 -myfit.b*((1+ myfit.a)*exp(-1*abs(tplot)/myfit.T1)-myfit.a*exp(-1*abs(tplot)/myfit.T2)),'black','linewidth',1.5);

plot(tplot,0.5*ones(length(tplot),1),'-.r')
xlabel('\tau (ns)','fontsize', 20)
ylabel('g^2(\tau)','fontsize', 20)
legend('Raw g^{2}(\tau) Data','Fitted g^{2}(\tau) Function',' g^{2}(\tau)=0.5 Threshold', 'fontsize', 14)
%ylim([0,3.5])
xlim([-20,20])
b=myfit.b;
I=N/(TT/1e12);
%t0=myfit.t0;
T1=myfit.T1;
T2=myfit.T2;
a=myfit.a;
%save(strcat(filename,'fittingparameters.mat'),'T1','T2','a','b','I')
%saveFigure(figure(2),strcat(filename,'g2function.bmp'))
%save(strcat('508nW_f2_e3.ptu','fittingparameters.mat'),'T1','T2','a','b','I')