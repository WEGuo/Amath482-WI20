% Clean workspace
clear all; close all; clc

%%
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
title('Sweet Child O Mine');
p8 = audioplayer(y,Fs); 
playblocking(p8);

%% explore window size and time step
close all; clear all; clc;

[y,Fs] = audioread('GNR.m4a');

y = y(1:185000);
S = y';
tr_gnr = length(y)/Fs;  

L = tr_gnr;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1];  
ks = fftshift(k);

% a = 100, dt = 0.1
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1,4,1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('a = 100, dt = 0.1'); 

% a = 100, dt = 0.01
a = 100;
tau = 0:0.01:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1,4,2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('a = 100, dt = 0.01'); 

% a = 300, dt = 0.1
a = 300;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1,4,3)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('a = 300, dt = 0.1'); 

% a = 300, dt = 0.01
a = 300;
tau = 0:0.01:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1,4,4)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('a = 300, dt = 0.01'); 

%% part 1 : guitar in GNR
% repeat 1
close all; clear all; clc;

[y,Fs] = audioread('GNR.m4a');

y = y(1:185000);
S = y';
tr_gnr = length(y)/Fs;  

L = tr_gnr;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1];  
ks = fftshift(k);

% Spectrogram
a = 300;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1,2,1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram Measures 1 - 2'); 

yline(554.37, 'w'); 
text(4.1, 554.37, sprintf('C#'),'FontSize',10);
yline(415.3, 'w'); 
text(4.1, 415.3, sprintf('G#'),'FontSize',10);
yline(370, 'w'); 
text(4.1, 370, sprintf('F#'),'FontSize',10);
yline(740, 'w'); 
text(4.1, 740, sprintf('F#'),'FontSize',10);
yline(698.5, 'w'); 
text(4.1, 698.5, sprintf('F'),'FontSize',10);
yline(277.18, 'w'); 
text(4.1, 277.18, sprintf('C#'),'FontSize',10);

% repeat 2
[y,Fs] = audioread('GNR.m4a');

y = y(185001 : 365000);
S = y';
tr_gnr = length(y)/Fs; 

L = tr_gnr;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 300;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt)); 
end

subplot(1,2,2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram Measures 3 - 4'); 

yline(554.37, 'w'); 
text(4.1, 554.37, sprintf('C#'),'FontSize',10);
yline(415.3, 'w'); 
text(4.1, 415.3, sprintf('G#'),'FontSize',10);
yline(370, 'w'); 
text(4.1, 370, sprintf('F#'),'FontSize',10);
yline(740, 'w'); 
text(4.1, 740, sprintf('F#'),'FontSize',10);
yline(698.5, 'w'); 
text(4.1, 698.5, sprintf('F'),'FontSize',10);
yline(311.13, 'w'); 
text(4.1, 311.13, sprintf('D#'),'FontSize',10);

saveas(gcf, 'Sweet1.png')

% repeat 3
close all; clear all; clc;

[y,Fs] = audioread('GNR.m4a');

y = y(365001 : 550000);
S = y';
tr_gnr = length(y)/Fs; 

L = tr_gnr;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1];
ks = fftshift(k); 

% Spectrogram
a = 300;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt)); 
end

subplot(1,2,1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram Measures 5 - 6'); 

yline(554.37, 'w'); 
text(4.1, 554.37, sprintf('C#'),'FontSize',10);
yline(415.3, 'w'); 
text(4.1, 415.3, sprintf('G#'),'FontSize',10);
yline(370, 'w'); 
text(4.1, 370, sprintf('F#'),'FontSize',10);
yline(740, 'w'); 
text(4.1, 740, sprintf('F#'),'FontSize',10);
yline(698.5, 'w'); 
text(4.1, 698.5, sprintf('F'),'FontSize',10);

% repeat 4
[y,Fs] = audioread('GNR.m4a');

y = y(550001 : end);
S = y';
tr_gnr = length(y)/Fs; 

L = tr_gnr;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 300;
tau = 0:0.1:L+2;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt)); 
end

subplot(1,2,2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [200 800], 'Xlim', [0 3.9]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram Measures 7 - 8'); 

yline(554.37, 'w'); 
text(4.1, 554.37, sprintf('C#'),'FontSize',10);
yline(415.3, 'w'); 
text(4.1, 415.3, sprintf('G#'),'FontSize',10);
yline(370, 'w'); 
text(4.1, 370, sprintf('F#'),'FontSize',10);
yline(740, 'w'); 
text(4.1, 740, sprintf('F#'),'FontSize',10);
yline(698.5, 'w'); 
text(4.1, 698.5, sprintf('F'),'FontSize',10);
yline(277.18, 'w'); 
text(4.1, 277.18, sprintf('C#'),'FontSize',10);

saveas(gcf, 'Sweet2.png')
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part 2 : bass in Floyd
close all; clear all; clc;

%%  Shannon filter

[y, Fs] = audioread('Floyd.m4a');

S = [y', 0, 0, 0];
tr_floyd = length(S)/Fs; 

L = tr_floyd;
n = length(S);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

Sf = fft(S);
[Ny, Nx] = size(S);
wy = 10000;
filter = ones(size(S));
filter(wy+1 : Nx-wy+1) = zeros(1, Nx-2*wy+1);

Sfiltered = Sf .* filter;
Sg_f = ifft(Sfiltered);

tr_gnr = length(y)/Fs; 
plot((1:length(y))/Fs, y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
hold on
plot((1:length(S))/Fs,Sg_f, 'r');
title('Amplitude of Isolated Bass Part'); 

saveas(gcf, 'Floyd Comparison.png')

% play the isolated bass
% the melody played by the guitar disappears 

% p8 = audioplayer(Sg_f,Fs); 
% playblocking(p8);

%% repeat 1
leng = length(Sg_f);

y = Sg_f(1:leng/2);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.2:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(2, 1, 1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [60 150]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the First Half'); 

yline(123.5,'w'); 
text(30.5,123.5,sprintf('B'));
yline(110,'w'); 
text(30.5,110,sprintf('A'));
yline(98,'w');
text(30.5,98,sprintf('G'));
yline(92.5,'w'); 
text(30.5,92.5,sprintf('F#'));
yline(82.4,'w'); 
text(30.5,82.4,sprintf('B'));

% repeat 2
y = Sg_f(leng/2+1:end);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.2:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(2, 1, 2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [60 150]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the Second Half'); 

yline(123.5,'w'); 
text(30.5,123.5,sprintf('B'));
yline(110,'w'); 
text(30.5,110,sprintf('A'));
yline(98,'w');
text(30.5,98,sprintf('G'));
yline(92.5,'w'); 
text(30.5,92.5,sprintf('F#'));
yline(82.4,'w'); 
text(30.5,82.4,sprintf('B'));

saveas(gcf, 'floyd.png')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% part 3 : guitar in Floyd
close all; clear all; clc;

%% Shannon filter

[y, Fs] = audioread('Floyd.m4a');

S = y(1:end-1)';
tr_floyd = length(S)/Fs; 

L = tr_floyd;
n = length(S);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

Sf = fft(S);
[Ny,Nx] = size(S);
wy = 10000;
filter = zeros(size(S));
filter(wy+1 : Nx-wy+1) = ones(1, Nx-2*wy+1);
wy = 100000;
filter(wy+1 : Nx-wy+1) = zeros(1, Nx-2*wy+1);

Sfiltered = Sf .* filter;
Sg_f = ifft(Sfiltered);

tr_gnr = length(y)/Fs; 
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); 
ylabel('Amplitude');
hold on
plot((1:length(S))/Fs,((Sg_f)), 'r');
title('Amplitude of Isolated Guitar Part'); 

saveas(gcf, 'Floyd Guitar Comparison.png')
%%
% the sound of bass disappears 
p8 = audioplayer(Sg_f,Fs); 
playblocking(p8);

%% repeat 1
leng = length(Sg_f);

y = Sg_f(1:leng/6);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 1000;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1, 3, 1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1300]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the 1 - 10 Seconds'); 

yline(740,'w'); 
text(10.2,740,sprintf('F#'));
yline(660,'w'); 
text(10.2,660,sprintf('E'));
yline(587,'w');
text(10.2,587,sprintf('D'));
yline(494,'w'); 
text(10.2,494,sprintf('B'));
yline(880,'w'); 
text(10.2,880,sprintf('A'));
% yline(1174.7,'w'); 
text(10.2,1174.7,sprintf('D'));

% repeat 2
y = Sg_f(leng/6+1:leng/3);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 1000;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1, 3, 2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1300]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the 10 - 20 Seconds'); 

yline(587,'w');
text(10.2,587,sprintf('D'));
yline(494,'w'); 
text(10.2,494,sprintf('B'));

% repeat 3
y = Sg_f(leng/3+1:leng/2);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 1000;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1, 3, 3)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1300]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the 20 - 30 Seconds'); 

yline(660,'w'); 
text(10.2,660,sprintf('E'));
yline(587,'w');
text(10.2,587,sprintf('D'));
yline(987.77,'w'); 
text(10.2,987.77,sprintf('B'));
% yline(1174.7,'w'); 
text(10.2,1174.7,sprintf('D'));

saveas(gcf, 'guitar1.png')

%% repeat 4
clf

y = Sg_f(leng/2+1:4*leng/6);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 1000;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1, 3, 1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1300]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the 30 - 40 Seconds'); 

% repeat 5
y = Sg_f(4*leng/6+1:5*leng/6);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 1000;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1, 3, 2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1300]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the 40 - 50 Seconds'); 

yline(660,'w'); 
text(10.2,660,sprintf('E'));
yline(587,'w');
text(10.2,587,sprintf('D'));
yline(987.77,'w'); 
text(10.2,987.77,sprintf('B'));
% yline(1174.7,'w'); 
text(10.2,1174,sprintf('D'));

% repeat 6
y = Sg_f(5*leng/6+1:end);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 1000;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(1, 3, 3)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1300]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the 50 - 60 Seconds'); 

yline(987.77,'w'); 
text(10.2,987.77,sprintf('B'));
yline(740,'w'); 
text(10.2,740,sprintf('F#'));

saveas(gcf, 'guitar2.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% repeat 1
leng = length(Sg_f);

y = Sg_f(1:leng/4);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 500;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(2, 2, 1)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 900]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the First Quarter'); 

% yline(123.5,'w'); 
% text(30.5,123.5,sprintf('B'));
% yline(110,'w'); 
% text(30.5,110,sprintf('A'));
% yline(98,'w');
% text(30.5,98,sprintf('G'));
% yline(92.5,'w'); 
% text(30.5,92.5,sprintf('F#'));
% yline(82.4,'w'); 
% text(30.5,82.4,sprintf('B'));



% repeat 2
leng = length(Sg_f);

y = Sg_f(leng/4+1:leng/2);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 500;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(2, 2, 2)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 900]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the Second Quarter');


% repeat 3

y = Sg_f(leng/2+1:3*leng/4);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 500;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(2, 2, 3)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 900]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the Third Quarter');

% repeat 4

y = Sg_f(3*leng/4+1:end);
S = y;
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 500;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

subplot(2, 2, 4)

pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 900]); 
xlabel('Time [sec]'); 
ylabel('Frequency'); 
title('Spectrogram for the Fourth Quarter');







%% repeat 2
% figure(2)
close all; clear all; clc;
[y, Fs] = audioread('Floyd.m4a');

%
y = y(700001:1400000);
S = y';
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
    Sgt(2401:660000) = 0;
    Sgt_spec(j,:) = fftshift(Sgt);
end

%
clf
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [50 200]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

yline(123.5,'w'); 
text(2.3,123.5,sprintf('B'));
yline(110,'w'); 
text(2.3,110,sprintf('A'));
yline(98,'w');
text(2.3,98,sprintf('G'));
yline(92.5,'w'); 
text(2.3,92.5,sprintf('F#'));
yline(82.4,'w'); 
text(2.3,82.4,sprintf('B'));
% saveas(gcf, 'floyd2.png')


%% repeat 3
% figure(2)
close all; clear all; clc;
[y, Fs] = audioread('Floyd.m4a');

y = y(1400001:2100000);
S = y';
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
    Sgt(2401:660000) = 0;
    Sgt_spec(j,:) = fftshift(Sgt);
end

clf
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [50 200]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

yline(123.5,'w'); 
text(2.3,123.5,sprintf('B'));
yline(110,'w'); 
text(2.3,110,sprintf('A'));
yline(98,'w');
text(2.3,98,sprintf('G'));
yline(92.5,'w'); 
text(2.3,92.5,sprintf('F#'));
yline(82.4,'w'); 
text(2.3,82.4,sprintf('B'));
% saveas(gcf, 'floyd3.png')


%% repeat 4
% figure(2)
close all; clear all; clc;
[y, Fs] = audioread('Floyd.m4a');

%
y = y(2100001:end);
y(length(y)+1) = 0;
S = y';
tr_floyd = length(y)/Fs; 

L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
    Sgt(2001:500000) = 0;
    Sgt_spec(j,:) = fftshift(Sgt);
end

clf
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [50 200]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

yline(123.5,'w'); 
text(2.3,123.5,sprintf('B'));
yline(110,'w'); 
text(2.3,110,sprintf('A'));
yline(98,'w');
text(2.3,98,sprintf('G'));
yline(92.5,'w'); 
text(2.3,92.5,sprintf('F#'));
yline(82.4,'w'); 
text(2.3,82.4,sprintf('B'));
% saveas(gcf, 'floyd4.png')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% part 3
%%
%% repeat 1

[y, Fs] = audioread('Floyd.m4a');

y = y(1:400000);
S = y';
tr_floyd = length(y)/Fs; 
L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
filter = ones(1, 400000);
filter(1:2401) = 0;
filter(10000:end) = 0;
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
%     Sgt(1:2401) = 0;
%     Sgt(660000:end) = 0;
    Sgt = Sgt .* filter;
    Sgt_spec(j,:) = fftshift(Sgt);
end
%
subplot(2, 3, 1)
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1000]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

% yline(740,'w'); 
% % text(2.3,123.5,sprintf('B'));
% yline(494,'w'); 
% % text(2.3,110,sprintf('A'));
% yline(622.25,'w');
% % text(2.3,98,sprintf('G'));
% yline(587.33,'w'); 
% % text(2.3,92.5,sprintf('F#'));
% 
% yline(880,'w'); 
% yline(659.26,'w'); 
% % text(2.3,82.4,sprintf('B'));


% repeat 2

[y, Fs] = audioread('Floyd.m4a');

y = y(400001:800000);
S = y';
tr_floyd = length(y)/Fs; 
L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
filter = ones(1, 400000);
filter(1:2401) = 0;
filter(10000:end) = 0;
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
%     Sgt(1:2401) = 0;
%     Sgt(660000:end) = 0;
    Sgt = Sgt .* filter;
    Sgt_spec(j,:) = fftshift(Sgt);
end
%
subplot(2, 3, 2)
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1000]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

% yline(740,'w'); 
% % text(2.3,123.5,sprintf('B'));
% yline(494,'w'); 
% % text(2.3,110,sprintf('A'));
% yline(622.25,'w');
% % text(2.3,98,sprintf('G'));
% yline(587.33,'w'); 
% % text(2.3,92.5,sprintf('F#'));
% 
% yline(880,'w'); 
% yline(659.26,'w'); 
% % text(2.3,82.4,sprintf('B'));


% repeat 3

[y, Fs] = audioread('Floyd.m4a');

y = y(800001:1200000);
S = y';
tr_floyd = length(y)/Fs; 
L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
filter = ones(1, 400000);
filter(1:2401) = 0;
filter(10000:end) = 0;
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
%     Sgt(1:2401) = 0;
%     Sgt(660000:end) = 0;
    Sgt = Sgt .* filter;
    Sgt_spec(j,:) = fftshift(Sgt);
end
%
subplot(2, 3, 3)
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1000]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

% yline(740,'w'); 
% % text(2.3,123.5,sprintf('B'));
% yline(494,'w'); 
% % text(2.3,110,sprintf('A'));
% yline(622.25,'w');
% % text(2.3,98,sprintf('G'));
% yline(587.33,'w'); 
% % text(2.3,92.5,sprintf('F#'));
% 
% yline(880,'w'); 
% yline(659.26,'w'); 
% % text(2.3,82.4,sprintf('B'));


% repeat 4

[y, Fs] = audioread('Floyd.m4a');

y = y(1200001:1600000);
S = y';
tr_floyd = length(y)/Fs; 
L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
filter = ones(1, 400000);
filter(1:2401) = 0;
filter(10000:end) = 0;
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
%     Sgt(1:2401) = 0;
%     Sgt(660000:end) = 0;
    Sgt = Sgt .* filter;
    Sgt_spec(j,:) = fftshift(Sgt);
end
%
subplot(2, 3, 4)
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1000]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

% yline(740,'w'); 
% % text(2.3,123.5,sprintf('B'));
% yline(494,'w'); 
% % text(2.3,110,sprintf('A'));
% yline(622.25,'w');
% % text(2.3,98,sprintf('G'));
% yline(587.33,'w'); 
% % text(2.3,92.5,sprintf('F#'));
% 
% yline(880,'w'); 
% yline(659.26,'w'); 
% % text(2.3,82.4,sprintf('B'));


% repeat 5

[y, Fs] = audioread('Floyd.m4a');

y = y(1600001:2000000);
S = y';
tr_floyd = length(y)/Fs; 
L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
filter = ones(1, 400000);
filter(1:2401) = 0;
filter(10000:end) = 0;
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
%     Sgt(1:2401) = 0;
%     Sgt(660000:end) = 0;
    Sgt = Sgt .* filter;
    Sgt_spec(j,:) = fftshift(Sgt);
end
%
subplot(2, 3, 5)
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1000]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

% yline(740,'w'); 
% % text(2.3,123.5,sprintf('B'));
% yline(494,'w'); 
% % text(2.3,110,sprintf('A'));
% yline(622.25,'w');
% % text(2.3,98,sprintf('G'));
% yline(587.33,'w'); 
% % text(2.3,92.5,sprintf('F#'));
% 
% yline(880,'w'); 
% yline(659.26,'w'); 
% % text(2.3,82.4,sprintf('B'));

% repeat 6

[y, Fs] = audioread('Floyd.m4a');

y = y(2000001:2400000);
S = y';
tr_floyd = length(y)/Fs; 
L = tr_floyd;
n = length(y);
t2 = linspace(0, L, n+1);
t = t2(1:n);
k = (1 / L) * [0:n/2-1 -n/2:-1]; 
ks = fftshift(k);

% Spectrogram
a = 100;
tau = 0:0.1:L;
Sgt_spec = [];
filter = ones(1, 400000);
filter(1:2401) = 0;
filter(10000:end) = 0;
for j=1:length(tau)
    g = exp(-a * (t-tau(j)).^2);
    Sg = g .* S;
    Sgt = fft(Sg);
%     Sgt_spec(j,:) = fftshift(abs(Sgt));
    Sgt = abs(Sgt);
%     Sgt(1:2401) = 0;
%     Sgt(660000:end) = 0;
    Sgt = Sgt .* filter;
    Sgt_spec(j,:) = fftshift(Sgt);
end
%
subplot(2, 3, 6)
pcolor(tau, ks, Sgt_spec.');
shading interp;
colormap(hot);
set(gca,'Ylim', [400 1000]); 
xlabel('Time (Seconds)'); 
ylabel('Frequency'); 
title('Spectrogram'); 

% yline(740,'w'); 
% % text(2.3,123.5,sprintf('B'));
% yline(494,'w'); 
% % text(2.3,110,sprintf('A'));
% yline(622.25,'w');
% % text(2.3,98,sprintf('G'));
% yline(587.33,'w'); 
% % text(2.3,92.5,sprintf('F#'));
% 
% yline(880,'w'); 
% yline(659.26,'w'); 
% % text(2.3,82.4,sprintf('B'));



% saveas(gcf, 'guitar1.png')










