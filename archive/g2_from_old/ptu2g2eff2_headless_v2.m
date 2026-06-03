function ptu2g2eff2_headless_v2(ptu_path, g2time_ns, timebin_ns, max_records, out_prefix)
    global filename;
    global data;
    rng(0);  % reproducible afterflash removal

    g2analyze_headless(ptu_path, max_records);

    % Remove zero-timestamp rows (preallocated tail if max_records used)
    data((data(:,2)==0), :) = [];
    fprintf('Records after cleaning: %d\n', size(data,1));

    g2time = g2time_ns * 1e3;     % ps
    timebin = timebin_ns * 1e3;   % ps

    I = ceil(g2time/timebin);
    tauindex = -I:I;
    tau = tauindex * timebin / 1000;   % ns

    % ---- Correlation #1 (raw, start-stop adjacent pairs) ----
    tic;
    c = zeros(length(tauindex),1);
    Nd = length(data);
    for i = 1:Nd-1
        if data(i,1)==0 && data(i+1,1)==1
            temp = data(i+1,2) - data(i,2);
            ind = I+1+floor(temp/timebin);
            if temp <= g2time
                c(ind) = c(ind)+1;
            end
        end
        if data(i,1)==1 && data(i+1,1)==0
            temp = data(i+1,2) - data(i,2);
            ind = I - ceil(temp/timebin) + 1;
            if temp <= g2time
                c(ind) = c(ind)+1;
            end
        end
    end
    fprintf('raw correlation: '); toc

    % Save raw histogram
    tplot = tau;
    c_raw = c;
    save([out_prefix '_raw.mat'], 'tau', 'c_raw', '-v7');

    % ---- Afterflash removal ----
    tic;
    datacopy = data;
    tempsum = 0; tempind = 0;
    for i = 1:length(c)
        if tau(i)>40 && tau(i)<90
            tempsum = tempsum + c(i);
            tempind = tempind + 1;
        end
    end
    cavg = tempsum/max(tempind,1);
    fprintf('cavg = %g\n', cavg);

    for i = 1:Nd-1
        if data(i,1)==0 && data(i+1,1)==1
            temp = data(i+1,2) - data(i,2);
            ind = I+1+floor(temp/timebin);
            if temp < g2time && abs(tau(ind))>9 && abs(tau(ind))<35
                u = rand;
                crat = local_poissrnd(cavg)/local_poissrnd(c(ind));
                if u > crat
                    datacopy(i+1, 2) = 0;
                end
            end
        end
        if data(i,1)==1 && data(i+1,1)==0
            temp = data(i+1,2) - data(i,2);
            ind = I - ceil(temp/timebin) + 1;
            if temp <= g2time && abs(tau(ind))>9 && abs(tau(ind))<35
                u = rand;
                crat = local_poissrnd(cavg)/local_poissrnd(c(ind));
                if u > crat
                    datacopy(i+1, 2) = 0;
                end
            end
        end
    end
    datacopy((datacopy(:,2)==0), :) = [];
    data = datacopy;
    fprintf('afterflash removal: '); toc

    % ---- Correlation #2 (after afterflash removal) ----
    tic;
    c = zeros(length(tauindex),1);
    Nd = length(data);
    for i = 1:Nd-1
        if data(i,1)==0 && data(i+1,1)==1
            temp = data(i+1,2) - data(i,2);
            ind = floor(temp/timebin);
            if ind <= I
                c(I+ind+1) = c(I+ind+1)+1;
            end
        end
        if data(i,1)==1 && data(i+1,1)==0
            temp = data(i+1,2) - data(i,2);
            ind = ceil(temp/timebin);
            if ind <= I
                c(I-ind+1) = c(I-ind+1)+1;
            end
        end
    end
    N = length(data);
    TT = data(N,2);
    N1 = sum(data(:,1)==0);
    N2 = sum(data(:,1)==1);
    A = (N1*N2)*timebin/TT;
    g2 = c/A;
    fprintf('final correlation: '); toc
    fprintf('N1=%d N2=%d TT=%g A=%g\n', N1, N2, TT, A);

    % ---- Fit ----
    fit_ok = false;
    try
        fo = fitoptions('Method','NonlinearLeastSquares', ...
                        'Lower',[0,-1,0.1,10], ...
                        'Upper',[inf,1,inf,inf], ...
                        'StartPoint', [1,0,10,5000]);
        ft = fittype('1 - b*((1+a)*exp(-1*abs(X)/T1)-a*exp(-1*abs(X)/T2))', ...
                     'dependent',{'Y'},'independent',{'X'}, ...
                     'coefficients',{'a','b','T1','T2'},'options',fo);
        myfit = fit(tau', g2, ft);
        a_f = myfit.a; b_f = myfit.b; T1_f = myfit.T1; T2_f = myfit.T2;
        fit_ok = true;
        fprintf('fit: a=%g b=%g T1=%g T2=%g\n', a_f, b_f, T1_f, T2_f);
    catch ME
        warning('fit failed: %s', ME.message);
        a_f = NaN; b_f = NaN; T1_f = NaN; T2_f = NaN;
    end

    save([out_prefix '.mat'], 'tau', 'c', 'g2', 'N1', 'N2', 'TT', 'A', ...
         'a_f', 'b_f', 'T1_f', 'T2_f', 'fit_ok', '-v7');

    % ---- Plot ----
    fig = figure('Visible','off');
    plot(tau, g2, 'color', [.8 .8 .8], 'linewidth', 1); hold on;
    if fit_ok
        tplot_fit = linspace(min(tau), max(tau), 5000);
        yfit = 1 - b_f*((1+a_f)*exp(-abs(tplot_fit)/T1_f) - a_f*exp(-abs(tplot_fit)/T2_f));
        plot(tplot_fit, yfit, 'k', 'linewidth', 1.5);
    end
    plot(tau, 0.5*ones(size(tau)), '-.r');
    xlabel('\tau (ns)', 'fontsize', 16);
    ylabel('g^2(\tau)', 'fontsize', 16);
    xlim([-20 20]);
    if fit_ok
        legend('Raw g^2(\tau)', 'Fit', 'g^2=0.5', 'Location', 'best');
    else
        legend('Raw g^2(\tau)', 'g^2=0.5', 'Location', 'best');
    end
    title(sprintf('ptu2g2eff2 — N=%d, max\\_records=%s', N, num2str(max_records)));
    print(fig, [out_prefix '.png'], '-dpng', '-r150');
    close(fig);
    fprintf('Saved %s.png and %s.mat\n', out_prefix, out_prefix);
end

function k = local_poissrnd(lambda)
    % Knuth's algorithm for small lambda
    if lambda <= 0; k = 0; return; end
    if lambda < 30
        L = exp(-lambda); k = 0; p = 1;
        while true
            k = k + 1;
            p = p * rand;
            if p <= L; k = k - 1; return; end
        end
    else
        % Normal approximation for larger lambda (more than enough here)
        k = max(0, round(lambda + sqrt(lambda) * randn));
    end
end

