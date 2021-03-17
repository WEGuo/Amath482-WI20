%% first video
clear all; close all; clc

video1 = VideoReader('monte_carlo_low.mp4');
vid1 = read(video1);

Nframes = size(vid1);
Nframes = Nframes(4);
duration = video1.Duration;
height = video1.height;
width = video1.width;

frames = zeros(height*width, Nframes);
for i = 1:Nframes 
   f = rgb2gray(vid1(:, :, :, i));
   frames(:, i) = f(:);
end

X = reshape(frames(:, 30), height, width);
imshow(uint8(X))
saveas(gcf, 'orig1.png')
clf

% DMD 
X = frames;

X1 = X(:, 1:end-1);
X2 = X(:, 2:end);
dt = duration / Nframes;

[U, S, V] = svd(X1, 'econ');
%% Singular value plots
sig = diag(S)/sum(diag(S));
subplot(2,2,1), plot(sig,'k.')
title('Singular Values Spectrum/Energy')
xlabel('Singular Values')
ylabel('Energy (%)')
subplot(2,2,2), semilogy(sig,'k.')
ylabel('log(energy)')
xlabel('Singular Values')
title('Log Energy Plot')

subplot(2,2,3), plot(sig(1:25),'k.', 'MarkerSize', 20)
title('First 25 Singular Values/Energy')
ylabel('Energy (%)')
subplot(2,2,4), semilogy(sig(1:25),'k.', 'MarkerSize', 20)
ylabel('log(energy)')
xlabel('Singular Values')
title('First 25 Log Energy Value Plot')

set(gcf,'position',[200, 500, 1500,500])
saveas(gcf, 'SV.png')
clf
%%
k = 2;
U_l = U(:, 1:k);
S_l = S(1:k, 1:k);
V_l = V(:, 1:k);
Stilde = U_l' * X2 * V_l / S_l;
[eV, D] = eig(Stilde);
mu = diag(D);
omega = log(mu)/dt;
Phi = U_l*eV;

t = linspace(0,duration,Nframes+1);
t = t(1:(end-1));
y0 = Phi\X1(:,1);
u_modes = zeros(k,Nframes-1);
for iter = 1:(length(t)-1)
   u_modes(:,iter) = y0.*exp(omega*t(iter)); 
end
u_dmd = Phi*u_modes;

low_rank = uint8(u_dmd);
sparse = X(:,1:Nframes-1) - u_dmd;

min(min(sparse));
% -201.3870
sparse = uint8(X(:,1:Nframes-1) - u_dmd + 200);

background = low_rank;
foreground = sparse;

vidbackg = zeros(height, width, Nframes-1);
vidforeg = zeros(height, width, Nframes-1);

for i = 1: (Nframes - 1)
    vidbackg(:, :, i) = reshape(background(:, i), height, width);
    vidforeg(:, :, i) = reshape(foreground(:, i), height, width);
end

%% visualization
i = 30;
subplot(3, 3, 1)
imshow(uint8(reshape(X(:, i), height, width)))
title('Original Frame 30')
subplot(3, 3, 2)
imshow(reshape(foreground(:, i), height, width))
title('Foreground of Frame 30')
subplot(3, 3, 3)
imshow(reshape(background(:, i), height, width))
title('Background of Frame 30')

i = 100;
subplot(3, 3, 4)
imshow(uint8(reshape(X(:, i), height, width)))
title('Original Frame 100')
subplot(3, 3, 5)
imshow(reshape(foreground(:, i), height, width))
title('Foreground of Frame 100')
subplot(3, 3, 6)
imshow(reshape(background(:, i), height, width))
title('Background of Frame 100')

i = 350;
subplot(3, 3, 7)
imshow(uint8(reshape(X(:, i), height, width)))
title('Original Frame 350')
subplot(3, 3, 8)
imshow(reshape(foreground(:, i), height, width))
title('Foreground of Frame 350')
subplot(3, 3, 9)
imshow(reshape(background(:, i), height, width))
title('Background of Frame 350')

set(gcf,'position',[200, 500, 1000,500])
saveas(gcf, 'Monte.png')
clf
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% second video
clear all; close all; clc

video1 = VideoReader('ski_drop_low.mp4');
vid1 = read(video1);

Nframes = size(vid1);
Nframes = Nframes(4);
duration = video1.Duration;
height = video1.height;
width = video1.width;

frames = zeros(height*width, Nframes);
for i = 1:Nframes 
   f = rgb2gray(vid1(:, :, :, i));
   frames(:, i) = f(:);
end

X = reshape(frames(:, 50), height, width);
imshow(uint8(X))
saveas(gcf, 'orig2.png')
clf
%% DMD 
X = frames;

X1 = X(:, 1:end-1);
X2 = X(:, 2:end);
dt = duration / Nframes;

[U, S, V] = svd(X1, 'econ');

%% Singular value plots
sig = diag(S)/sum(diag(S));
subplot(2,2,1), plot(sig,'k.')
title('Singular Values Spectrum/Energy')
xlabel('Singular Values')
ylabel('Energy (%)')
subplot(2,2,2), semilogy(sig,'k.')
ylabel('log(energy)')
xlabel('Singular Values')
title('Log Energy Plot')

subplot(2,2,3), plot(sig(1:25),'k.', 'MarkerSize', 20)
title('First 25 Singular Values/Energy')
ylabel('Energy (%)')
subplot(2,2,4), semilogy(sig(1:25),'k.', 'MarkerSize', 20)
ylabel('log(energy)')
xlabel('Singular Values')
title('First 25 Log Energy Value Plot')

set(gcf,'position',[200, 500, 1500,500])
saveas(gcf, 'SV2.png')
clf
%%
k = 1;
U_l = U(:, 1:k);
S_l = S(1:k, 1:k);
V_l = V(:, 1:k);
Stilde = U_l' * X2 * V_l / S_l;
[eV, D] = eig(Stilde);
mu = diag(D);
omega = log(mu)/dt;
Phi = U_l*eV;

t = linspace(0,duration,Nframes+1);
t = t(1:(end-1));
y0 = Phi\X1(:,1);
u_modes = zeros(k,Nframes-1);
for iter = 1:(length(t)-1)
   u_modes(:,iter) = y0.*exp(omega*t(iter)); 
end
u_dmd = Phi*u_modes;

low_rank = uint8(u_dmd);
sparse = X(:,1:Nframes-1) - u_dmd;

min(min(sparse));
% -216.5599
sparse = uint8(X(:,1:Nframes-1) - u_dmd + 200);

background = low_rank;
foreground = sparse;

vidbackg = zeros(height, width, Nframes-1);
vidforeg = zeros(height, width, Nframes-1);

for i = 1: (Nframes - 1)
    vidbackg(:, :, i) = reshape(background(:, i), height, width);
    vidforeg(:, :, i) = reshape(foreground(:, i), height, width);
end

%% visualization
i = 50;
subplot(3, 3, 1)
imshow(uint8(reshape(X(:, i), height, width)))
title('Original Frame 50')
subplot(3, 3, 2)
imshow(reshape(foreground(:, i), height, width))
title('Foreground of Frame 50')
subplot(3, 3, 3)
imshow(reshape(background(:, i), height, width))
title('Background of Frame 50')

i = 100;
subplot(3, 3, 4)
imshow(uint8(reshape(X(:, i), height, width)))
title('Original Frame 100')
subplot(3, 3, 5)
imshow(reshape(foreground(:, i), height, width))
title('Foreground of Frame 100')
subplot(3, 3, 6)
imshow(reshape(background(:, i), height, width))
title('Background of Frame 100')

i = 300;
subplot(3, 3, 7)
imshow(uint8(reshape(X(:, i), height, width)))
title('Original Frame 300')
subplot(3, 3, 8)
imshow(reshape(foreground(:, i), height, width))
title('Foreground of Frame 300')
subplot(3, 3, 9)
imshow(reshape(background(:, i), height, width))
title('Background of Frame 300')

set(gcf,'position',[200, 500, 1000,500])
saveas(gcf, 'ski.png')
clf