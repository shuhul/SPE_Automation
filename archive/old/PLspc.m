% Set scan parameters
% Size of scan area in um
xdim =20; %um
ydim =20; %um
% Step size in each direction
dx = 0.5; %um
dy = dx; %um

filename='20251103-PLSPC-PhENOM-Ch10-f014o018-150uw-1300msIntegration-test1';
%filename='20250602-PLSPC-EmFabJune2-f2-150GMM-50uW';
% %%% PL map, Serpent Scanning Pattern 
% Startup delay
pause(3);
% Initialize equipment
if ~exist('instance1')
    Setup_LightField_Environment;
    instance1=lfm(true);
end
[int,wl]=instance1.acquire;
SS=max(size(int));
fprintf('\nInitialize Siglent SDG');
obj1 = InitializeSDG;
wait4action=0.1;
%turn on output
fprintf(obj1, 'C1: OUTP ON'); pause(wait4action);
fprintf(obj1, 'C2: OUTP ON'); pause(wait4action);

% % FSM Calibration
% % y axis: 2.65 V / 20 um
% % x axis: -1.85 V / 20 um
% Due to current optical configuration, the x-axis is reversed
% Conversion factors, Length -> Voltage
XCONV = -1.85/20;
YCONV = 2.65/20;



if(abs(xdim/2 * XCONV) > 10 || (ydim/2 * YCONV) > 10)
    error('Scan area too large for mirror');
end


if(dx > xdim || dy > ydim)
    error('Step size too large');
end

% Starting coordinates (The point (x0,y0) is the center of the square centered at FSM(0V,0V)


X=zeros(1,(xdim/dx)+1);
Y=zeros(1,(ydim/dy)+1);

for i=1:(xdim/dx)+1
    X(i)=-(xdim/2)+(i-1)*dx;
end
for i=1:(ydim/dy)+1
    Y(i)=-(ydim/2)+(i-1)*dy;
    YY(i)=(ydim/2)-(i-1)*dy;
end



z= zeros((ydim / dy)+1 , (xdim / dx)+1 ,SS);
PL=zeros((ydim / dy)+1 , (xdim / dx)+1);

for i = 1:(ydim / dy)+1
    % Update FSM y position
    y = YY(i) * YCONV;
    yvolt = strcat('C1: BSWV OFST, ', num2str(y));
    fprintf(obj1, yvolt); pause(wait4action);
    
    for j = 1:(xdim / dx)+1
        % Update FSM x position
        x = X(j) * XCONV;
        xvolt = strcat('C2: BSWV OFST, ', num2str(x));
        fprintf(obj1, xvolt); pause(wait4action);
        % Measure PL, create map
        int=instance1.acquire;
        % Account for reversing x direction in serpent scan
        
        z(i, j,:) = int;
        PL(i,j)=sum(int);
        
        %%%%%%%%%%%%%%%%%%%%%%%%
    end
    s=round((i/((ydim / dy)+1))*100,1);
    progress=strcat( num2str(s),'% completed')
    

end

% Close the wavefuntion SDG
fprintf(obj1, 'C1: BSWV OFST, 0'); pause(wait4action);
fprintf(obj1, 'C2: BSWV OFST, 0'); pause(wait4action);
fprintf(obj1, 'C1: OUTP OFF'); pause(wait4action);
fprintf(obj1, 'C2: OUTP OFF'); pause(wait4action);
fclose(obj1);
delete(obj1);
clear obj1;


figure(1)
imagesc(X,Y,PL)
title('PLmap-SPC')
ylabel('Y [um]')
xlabel('X [um]')
colorbar



save(strcat(filename,'.mat'),'PL','wl','z','X','Y');
saveas(figure(1),strcat(filename,'.bmp'));