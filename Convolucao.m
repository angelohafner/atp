clearvars;
close all;
clc;

set(0, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% Time axis
dt = 1e-2;
tt = -10:dt:20;
nn = numel(tt);

% Select functions
x = input_function(tt);
h_imp = impulse_function(tt);

% Convolution
y_full = conv(x, h_imp, 'full') * dt;
t_full = (tt(1) + tt(1)):dt:(tt(end) + tt(end));
y = interp1(t_full, y_full, tt, 'linear', 0);

% Frames to animate
frame_idx = unique([1, 20:20:nn]);

% GIF file name
gif_name = 'convolucao_animada.gif';

% Create figure
hFig = figure('Color', 'w', 'Position', [100, 100, 1300, 500]);
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
    h_shift = impulse_function(t_now - tt);

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

% Analytical response for the active default case
% Default active case:
% - input_function: unit step
% - impulse_function: second-order underdamped

zeta = 0.2;
wn = 2;
wd = wn * sqrt(1 - zeta^2);

y_ref = 1 - exp(-zeta * wn * tt) .* ...
    (cos(wd * tt) + (zeta / sqrt(1 - zeta^2)) * sin(wd * tt));
y_ref(tt < 0) = 0;

% Alternative analytical response for first-order case
% tau = 1.0;
% y_ref = 1 - exp(-tt / tau);
% y_ref(tt < 0) = 0;

figure('Color', 'w', 'Position', [120, 120, 1300, 500]);
plot(tt, y, '--', 'LineWidth', 1.2);
hold on;
plot(tt, y_ref, 'LineWidth', 1.2);
grid on;
box on;
xlabel('$t$');
ylabel('Amplitude');
title('Comparacao entre convolucao numerica e resposta analitica');
legend('Convolucao numerica', 'Resposta analitica', 'Location', 'best');

function x = input_function(t)
    % Original test: unit step
    x = (t >= 0);

    % Alternative test 1: decaying exponential
    % x = exp(-t) .* (t >= 0);

    % Alternative test 2: rectangular pulse
    % x = ((t >= 0) & (t <= 3));

    % Alternative test 3: damped sine
    % x = exp(-0.5 * t) .* sin(3 * t) .* (t >= 0);

    % Alternative test 4: Gaussian pulse
    % x = exp(-t.^2);

    % Alternative test 5: cosine starting at zero
    % x = cos(2 * t) .* (t >= 0);
end

function h = impulse_function(t)
    % Original test: second-order underdamped impulse response
    % Transfer function:
    % H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    zeta = 0.2;
    wn = 2;
    wd = wn * sqrt(1 - zeta^2);

    h = (wn / sqrt(1 - zeta^2)) ...
        .* exp(-zeta * wn * t) ...
        .* sin(wd * t) ...
        .* (t >= 0);

    % Alternative test 1: first-order impulse response
    % Transfer function:
    % H(s) = 1 / (tau*s + 1)
    % tau = 1.0;
    % h = (1 / tau) .* exp(-t / tau) .* (t >= 0);

    % Alternative test 2: simple exponential
    % h = exp(-t) .* (t >= 0);

    % Alternative test 3: sine
    % h = sin(2 * t) .* (t >= 0);

    % Alternative test 4: cosine
    % h = cos(2 * t) .* (t >= 0);

    % Alternative test 5: rectangular pulse
    % h = ((t >= 0) & (t <= 2));

    % Alternative test 6: Gaussian
    % h = exp(-t.^2);
end