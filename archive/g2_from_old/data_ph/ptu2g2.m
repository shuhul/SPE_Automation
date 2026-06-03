clear 
clc
global filename;
tic;
global tplot;
global c;
global data;
  
totaltime= 1e6; %nano second
timebin = 500;  %pico second 
g2analyze;
toc
fprintf('\n Now measuring correlation \n');


    T=totaltime*1000;
    tt=ceil(T/timebin);
    ttt=-1*tt:tt;
    t=ttt*timebin;
    c=zeros(1,2*tt+1);
    N=size(data);
    N=N(1);

    for i=1:N
        if data(i,2)==0
            break
        end
        if data(i,1)==0
            for j=i+1:N
                if data(j,1)==1
                    dt=data(j,2)-data(i,2);
                    if abs(dt)>T
                        break
                    end
                    for k=1:2*tt
                        if dt>t(k)  
                            if dt<=t(k+1)
                            	c(k)=c(k)+1;
                            end
                        end
                    end
                end
            end
        end
    end

    for i=1:N
        if data(i,2)==0
            break
        end
        if data(i,1)==0 
            for j=1:i-1
                if data(i-j,1)==1
                    dt=data(i-j,2)-data(i,2);
                    if abs(dt)>T
                        break
                    end
                    for k=1:2*tt
                        if dt>t(k) 
                            if dt<=t(k+1)
                                c(k)=c(k)+1;
                            end
                        end
                    end
                end
            end
        end
    end
    
toc
    tplot=t/1000;
    figure(1)
    plot(tplot,c)
    ylabel('coincident counts','Fontsize',20)
    xlabel('time [ns]','Fontsize',20)
    xlim([-10 10])
    set(gca,'FontSize',20)
    saveas(figure(1),strcat(filename,'.bmp'));
save(strcat(filename,'.mat'),'data','c','tplot');