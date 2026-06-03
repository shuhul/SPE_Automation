function ptu2g2_headless(ptu_path, totaltime_ns, timebin_ps, max_records, out_mat)
    global filename;
    global data;

    g2analyze_headless(ptu_path, max_records);

    % trim trailing zeros (matlab original uses `if data(i,2)==0; break; end`)
    valid = data(:,2) ~= 0;
    if any(~valid)
        first_zero = find(~valid, 1, 'first');
        data = data(1:first_zero-1, :);
    end

    T  = totaltime_ns * 1000;  % ps
    tt = ceil(T / timebin_ps);
    ttt = -1*tt:tt;
    t = ttt * timebin_ps;
    c = zeros(1, 2*tt+1);
    N = size(data, 1);

    for i = 1:N
        if data(i,2) == 0
            break
        end
        if data(i,1) == 0
            for j = i+1:N
                if data(j,1) == 1
                    dt = data(j,2) - data(i,2);
                    if abs(dt) > T
                        break
                    end
                    for k = 1:2*tt
                        if dt > t(k)
                            if dt <= t(k+1)
                                c(k) = c(k) + 1;
                            end
                        end
                    end
                end
            end
        end
    end

    for i = 1:N
        if data(i,2) == 0
            break
        end
        if data(i,1) == 0
            for j = 1:i-1
                if data(i-j,1) == 1
                    dt = data(i-j,2) - data(i,2);
                    if abs(dt) > T
                        break
                    end
                    for k = 1:2*tt
                        if dt > t(k)
                            if dt <= t(k+1)
                                c(k) = c(k) + 1;
                            end
                        end
                    end
                end
            end
        end
    end

    tplot = t / 1000;
    save(out_mat, 'c', 'tplot', 'data', '-v7');
    fprintf('Saved %s  (N=%d, pairs=%d)\n', out_mat, N, sum(c));
end
