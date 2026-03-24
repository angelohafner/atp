clearvars;
close all;
clc;

set(0, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% Define system parameters
zeta = 0.2;
wn = 2;
wd = wn * sqrt(1 - zeta^2);

% Time axis (now including negative time)
dt = 1e-2;
tt = -10:dt:20;
nn = numel(tt);

% Input: unit step (causal)
x = (tt >= 0);

% Impulse response h(t) (causal)
h_imp = (wn / sqrt(1 - zeta^2)) ...
    * exp(-zeta * wn * tt) ...
    .* sin(wd * tt) ...
    .* (tt >= 0);

% Convolution (full)
y_full = conv(x, h_imp, 'full') * dt;
t_full = (tt(1) + tt(1)):dt:(tt(end) + tt(end));

% Interpolate back to original time axis
y = interp1(t_full, y_full, tt, 'linear', 0);

% Frames to animate
frame_idx = unique([1, 20:20:nn]);

% GIF name
gif_name = 'convolucao_segunda_ordem_subamortecido.gif';

% Create figure
hFig = figure('Color', 'w');
hold on;
grid on;
box on;

% Create plot objects
hShift = plot(tt, nan(size(tt)), 'LineWidth', 1.2);
hInput = plot(tt, x, 'LineWidth', 1.2);
hPoint = scatter(nan, nan, 40, 'filled');
hArea = area(nan, nan, 'FaceAlpha', 0.35, 'LineStyle', 'none');

xlabel('$t$');
ylabel('Amplitude');
title('$y(t)=\int_{-\infty}^{\infty} h(t-\tau)x(\tau)\,d\tau$');

legend( ...
    '$h(t-\tau)$', ...
    '$x(\tau)$', ...
    '$y(t)$', ...
    '$y(t)$ acumulada', ...
    'Location', 'best' ...
);

xlim([tt(1), tt(end)]);

% Dynamic Y-axis
y_all = [x, h_imp, y];
ymin_data = min(y_all);
ymax_data = max(y_all);
ymargin = 0.10 * (ymax_data - ymin_data);

if ymargin == 0
    ymargin = 0.1;
end

ylim([ymin_data - ymargin, ymax_data + ymargin]);

% Animation loop
for m = 1:numel(frame_idx)
    k = frame_idx(m);
    t_now = tt(k);

    % Shifted impulse response
    h_shift = (wn / sqrt(1 - zeta^2)) ...
        * exp(-zeta * wn * (t_now - tt)) ...
        .* sin(wd * (t_now - tt)) ...
        .* ((t_now - tt) >= 0);

    % Update graphics
    set(hShift, 'YData', h_shift);
    set(hPoint, 'XData', t_now, 'YData', y(k));
    set(hArea, 'XData', tt(1:k), 'YData', y(1:k));

    drawnow;

    % Capture frame
    frame = getframe(hFig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);

    % Write GIF
    if m == 1
        imwrite(imind, cm, gif_name, 'gif', 'Loopcount', inf, 'DelayTime', 0.04);
    else
        imwrite(imind, cm, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', 0.04);
    end
end

% Analytical step response (causal)
y_step = 1 - exp(-zeta * wn * tt) .* ...
    (cos(wd * tt) + (zeta / sqrt(1 - zeta^2)) * sin(wd * tt));
y_step(tt < 0) = 0;

figure('Color', 'w');
plot(tt, y, '--', 'LineWidth', 1.2); hold on;
plot(tt, y_step, 'LineWidth', 1.2);
grid on;
box on;
xlabel('$t$');
ylabel('Amplitude');
title('Comparacao entre convolucao numerica e resposta analitica');
legend('Convolucao numerica', 'Resposta analitica', 'Location', 'best');