M=size(c);
M=M(2);
J=0;
for i=1:M
    J=J+1;
    if tplot(i)>-30
        if tplot(i)<-10
            J=J-1;
        end
    end
    if tplot(i)<=30
        if tplot(i)>10
            J=J-1;
        end
    end
end
ttplot=zeros(1,J);
cc=zeros(1,J);
j=1;
for i=1:M
    if tplot(i)<=-30
        ttplot(j)=tplot(i);
        cc(j)=c(i);
        j=j+1;
    end
    if tplot(i)>=-10
        if tplot(i)<=10
            ttplot(j)=tplot(i);
            cc(j)=c(i);
            j=j+1; 
        end
    end
    if tplot(i)>=30
        ttplot(j)=tplot(i);
        cc(j)=c(i);
        j=j+1;
    end
end


%% This parts devides the result by average of T>8us
n0=0;
counter=0;
for i=1:M
    if tplot(i)<-8e2
        n0=n0+c(i);
        counter=counter+1;
    end
end
for i=1:M
    if tplot(i)>8e2
        n0=n0+c(i);
        counter=counter+1;
    end
end
avg=n0/counter;
cc=cc/avg;
        
%% This part fits a line to it

% v0=[1,0.1,3,50];
% 
% vL=nlinfit(tplot,c,@g2function,v0);
% lfit=g2function(vL,tplot);
% 
% scatter(tplot,c)
% hold on
% plot(tplot,lfit)
% hold off
X=ttplot;
Y=cc;

fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,0,0,500,-3],...
               'Upper',[10,1,10,5000,3]);
ft = fittype('1 - b*((1+a)*exp(-1*abs(X-t0)/T1)-a*exp(-1*abs(X-t0)/T2))',...
    'dependent',{'Y'},'independent',{'X'},...
    'coefficients',{'a','b','T1','T2','t0'},'options',fo)
myfit=fit(X',Y',ft)

figure(1)
plot(X,Y,'cyan');
hold on
plot(tplot,1 -myfit.b*((1+ myfit.a)*exp(-1*abs(tplot-myfit.t0)/myfit.T1)-myfit.a*exp(-1*abs(tplot-myfit.t0)/myfit.T2)),'linewidth',2);
hold off
%title('Autocorrelation time resolution: 100ps T=1.02ns');
ylabel('g2','Fontsize',20);
xlabel('time delay [ns]','Fontsize',20);
set(gca,'FontSize',20)
%ylim([0 1.5]);
xlim([-10 10]);
    