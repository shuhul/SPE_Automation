% Set scan parameters

xc =0; %um
yc =0; %Xum

sc=60; %um
% Step size in each direction
dx = 0.2; %um
dy = dx; %um

%filename='20231107-fineAPD-f14-fullchip';
filename='20260403-bigPLapd-PhENOMCh21-f009o009-500uW-2';
% Startup delay
pause(1);
% Initialize equipment
fprintf('\nInitialize PicoHarp 300');
device = InitializePicoHarp;
Countrate0 = 0;
Countrate1 = 0;
wait4action = 0.1;
fprintf('\nInitialize Siglent SDG');
obj1 = InitializeSDG;

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
% Starting coordinates (The point (x0,y0) is the upper left corner of the square centered at FSM(0V,0V)
% y0=ydim/2 * YCONV;
% x0=-xdim/2 * XCONV;

X=zeros(1,(sc/dx)+1);
Y=zeros(1,(sc/dy)+1);

for i=1:(sc/dx)+1
    X(i)=xc-(sc/2)+(i-1)*dx;
end
for i=1:(sc/dy)+1
    Y(i)=yc-(sc/2)+(i-1)*dy;
    YY(i)=-yc+(sc/2)-(i-1)*dy;
end



z0 = zeros((sc / dy)+1 , (sc / dx)+1 );
z1 = zeros((sc / dy)+1 , (sc / dx)+1 );
for i = 1:(sc / dy)+1
    % Update FSM y position
    y = YY(i) * YCONV;
    yvolt = strcat('C1: BSWV OFST, ', num2str(y));
    fprintf(obj1, yvolt); pause(wait4action);
    
    for j = 1:(sc / dx)+1
        % Update FSM x position
        x = X(j) * XCONV;
        xvolt = strcat('C2: BSWV OFST, ', num2str(x));
        fprintf(obj1, xvolt); pause(wait4action);
        % Measure PL, create map
        [Countrate0, Countrate1] = GetPL(device);
        % Account for reversing x direction in serpent scan
        
        z0(i, j) = Countrate0;
        z1(i, j) = Countrate1;
        %%%%%%%%%%%%%%%%%%%%%%%%
    end
    s=(i/((sc / dy)+1))*100;
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

% Closing the Picoharp300
ret = calllib('PHlib', 'PH_CloseDevice', 0);
closedev;

PL=z0+z1;

figure(2)
imagesc(X,Y,PL)
title('PLmap-APD-fine')
ylabel('Y [um]')
xlabel('X [um]')
colorbar


for j=1:(sc/dx)+1
    for i=1:(sc/dy)+1
        if PL(i,j)==max(max(PL));
            fprintf(strcat('\nx=',num2str(X(j))));
            fprintf(strcat('\ny=',num2str(-YY(i)),'\n'));
        end
    end
end
save(strcat(filename,'.mat'),'PL','X','Y');
saveas(figure(2),strcat(filename,'.bmp'));