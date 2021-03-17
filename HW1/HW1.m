clear all; close all; clc

load subdata.mat 
% Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L, L, n+1);
x = x2(1 : n);
y = x;
z = x;
k = (2 * pi / (2 * L)) * [0:(n/2 - 1) -n/2:-1];
ks = fftshift(k);

[X, Y, Z] = meshgrid(x, y, z);
[Kx, Ky, Kz] = meshgrid(ks, ks, ks); 

% averaging
avg = zeros(n, n, n); 
for j = 1:49
    avg(:, :, :) = avg + abs(fftn(reshape(subdata(:, j), n, n, n))); 
end
avg = fftshift(avg)/49;

% figure(1)
isosurface(Kx, Ky, Kz, avg/max((avg(:))), 0.5)
axis([-15 15 -15 15 -15 15])
grid on
drawnow 
xlabel("Kx")
ylabel ("Ky")
zlabel ("Kz") 
title ("The Signature Frequency isoV = 0.5") 
saveas(gcf, 'Signature0.5.png')
clf

isosurface(Kx, Ky, Kz, avg/max((avg(:))), 0.8)
axis([-15 15 -15 15 -15 15])
grid on
drawnow 
xlabel("Kx")
ylabel ("Ky")
zlabel ("Kz") 
title ("The Signature Frequency isoV = 0.8") 
saveas(gcf, 'Signature0.8.png')
clf

[maximum, index] = max(avg(:)); 
[i1, i2, i3] = ind2sub([n, n, n], index); 
xx = ks(i2); 
yy = ks(i1); 
zz = ks(i3);

% Gaussian filtering
% figure(2)
clf
a = 3;
filter = exp(-a * ((Kx - xx).^2 + (Ky - yy).^2 + (Kz - zz).^2)) ; 
locX = zeros(1,49);
locY = zeros(1,49);
locZ = zeros(1,49);
for j = 1:49
    Un(:, :, :) = reshape(subdata(:, j), n, n, n); 
    utn = fftn(Un);
    unft = filter .* fftshift(utn);
    unf = ifftn(unft);
    [M, I] = max(unf(:));
    [i1, i2, i3] = ind2sub(size(unf), I);
    locX(j) = x(i1);
    locY(j) = y(i2);
    locZ(j) = z(i3);
    isosurface(X, Y, Z, abs(unf) / max(abs(unf(:))), 0.5) 
    axis([-20 20 -20 20 -20 20])
    grid on
    drawnow 
    xlabel("X")
    ylabel ("Y")
    zlabel ("Z") 
    title ("Detected Submarine Trajectory t = 3") 
end
saveas(gcf, 'Trajectory3.png')
clf

a = 1;
filter = exp(-a * ((Kx - xx).^2 + (Ky - yy).^2 + (Kz - zz).^2)) ; 
locX = zeros(1,49);
locY = zeros(1,49);
locZ = zeros(1,49);
for j = 1:49
    Un(:, :, :) = reshape(subdata(:, j), n, n, n); 
    utn = fftn(Un);
    unft = filter .* fftshift(utn);
    unf = ifftn(unft);
    [M, I] = max(unf(:));
    [i1, i2, i3] = ind2sub(size(unf), I);
    locX(j) = x(i1);
    locY(j) = y(i2);
    locZ(j) = z(i3);
    isosurface(X, Y, Z, abs(unf) / max(abs(unf(:))), 0.5) 
    axis([-20 20 -20 20 -20 20])
    grid on
    drawnow 
    xlabel("X")
    ylabel ("Y")
    zlabel ("Z") 
    title ("Detected Submarine Trajectory t = 1") 
end
saveas(gcf, 'Trajectory1.png')
clf

a = 0.5;
filter = exp(-a * ((Kx - xx).^2 + (Ky - yy).^2 + (Kz - zz).^2)) ; 
locX = zeros(1,49);
locY = zeros(1,49);
locZ = zeros(1,49);
for j = 1:49
    Un(:, :, :) = reshape(subdata(:, j), n, n, n); 
    utn = fftn(Un);
    unft = filter .* fftshift(utn);
    unf = ifftn(unft);
    [M, I] = max(unf(:));
    [i1, i2, i3] = ind2sub(size(unf), I);
    locX(j) = x(i2);
    locY(j) = y(i1);
    locZ(j) = z(i3);
    isosurface(X, Y, Z, abs(unf) / max(abs(unf(:))), 0.5) 
    axis([-20 20 -20 20 -20 20])
    grid on
    drawnow 
    xlabel("X")
    ylabel ("Y")
    zlabel ("Z") 
    title ("Detected Submarine Trajectory t = 0.5") 
end
saveas(gcf, 'Trajectory0.5.png')
clf

% figure(3)
plot3(locX, locY, locZ, '-o', 'LineWidth', 1)
axis ([ -10 10 -10 10 -10 10])
xlabel("X")
ylabel ("Y")
zlabel ("Z") 
grid on
hold on
plot3(locX(49), locY(49), locZ(49), 'rs', 'MarkerSize', 15)
title ("Path of the Submarine") 
saveas(gcf, 'plot3.png')
clf

% print the coordinates
% figure(4)
coord = {'X';'Y';'Z'};
T = table(locX', locY', locZ');
uitable('Data', T{1:25,:}, 'ColumnName', coord, 'RowName', ...
    1:25 ,'Units', 'Normalized', 'Position', [0, 0, 1, 1]);
saveas(gcf, 'table1.png')

uitable('Data', T{26:49,:}, 'ColumnName', coord, 'RowName', ...
    26:49 ,'Units', 'Normalized', 'Position', [0, 0, 1, 1]);
saveas(gcf, 'table2.png')
