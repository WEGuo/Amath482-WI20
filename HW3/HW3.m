clear all; close all; clc

load('cam1_1.mat')
load('cam1_2.mat')
load('cam1_3.mat')
load('cam1_4.mat')
load('cam2_1.mat')
load('cam2_2.mat')
load('cam2_3.mat')
load('cam2_4.mat')
load('cam3_1.mat')
load('cam3_2.mat')
load('cam3_3.mat')
load('cam3_4.mat')
%%
implay(vidFrames1_1)

%% Test 1: Ideal case
numFrames1 = size(vidFrames1_1, 4);
numFrames2 = size(vidFrames2_1, 4);
numFrames3 = size(vidFrames3_1, 4);

x_1 = [];
y_1 = [];

crop = zeros(480,640);
crop(200:430,300:400) = ones(231,101);

for j = 1:numFrames1
    X1 = rgb2gray(vidFrames1_1(:,:,:,j));
    X1 = uint8(crop) .* X1;
    [M, I] = max(double(X1(:))); 
    [row, col] = find(X1 >= 0.92 * M); 
    x_1(j) = mean(col);
    y_1(j) = mean(row);
end

[~, i] = max(y_1(1:50)); 
x_1 = x_1 (i:end);
y_1 = y_1 (i:end);

y_2 = [];
x_2 = [];

crop = zeros(480,640);
crop(100:400,250:350) = ones(301,101);

for j = 1:numFrames2
    X2 = rgb2gray(vidFrames2_1(:,:,:,j));
    X2 = uint8(crop) .* X2;
    [M, I] = max(double(X2(:))); 
    [row, col] = find(X2 >= 0.92 * M); 
    x_2(j) = mean(col);
    y_2(j) = mean(row);
end

[~, i] = max(y_2(1:50)); 
x_2 = x_2(i:end);
y_2 = y_2(i:end);

x_3 = [];
y_3 = [];

crop = zeros(480,640);
crop(250:350,250:450) = ones(101,201);

for j = 1:numFrames3
    X3 = rgb2gray(vidFrames3_1(:,:,:,j));
    X3 = uint8(crop) .* X3;
    [M, I] = max(double(X3(:))); 
    [row, col] = find(X3 >= 0.92 * M); 
    x_3(j) = mean(col);
    y_3(j) = mean(row);
end
[~, i] = max(y_3(1:50)); 
x_3 = x_3(i:end);
y_3 = y_3(i:end);

minLength = min([length(y_1), length(y_2), length(y_3)]);
X = [x_1(1:minLength); y_1(1:minLength); x_2(1:minLength); 
    y_2(1:minLength); x_3(1:minLength); y_3(1:minLength)]; 
[~, n] = size(X);
avg = mean(X, 2);
X = X - repmat(avg,1,n);
[U, S, V] = svd(X, 'econ');

figure()
subplot(1, 3, 1)
plot(diag(S)./sum(diag(S)), 'k.', 'MarkerSize', 20)
xlabel('Principal Component')
ylabel('Energy')
title('Ideal Case Energy in Each Component')
xlim([1 7])

subplot(1, 3, 2)
plot(V(:,1)); 
hold on;
plot(V(:,2));
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Ideal Case - Displacement across Principle Component')
legend('PC1', 'PC2') 

subplot(1, 3, 3)
V = V * S;
plot(V(:,1)); 
hold on;
plot(V(:,2));
plot(V(:,3));
plot(V(:,4));
plot(V(:,5));
plot(V(:,6)); 
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Ideal Case - Variance across Principle Component')
legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')

%% Cropping visualization
figure()
subplot(1, 2, 1)
X1 = rgb2gray(vidFrames1_1(:,:,:,50));
imshow(X1); 

subplot(1, 2, 2)
crop = zeros(480,640);
crop(200:430,300:400) = ones(231,101);
X1 = uint8(crop) .* X1;
imshow(X1); 

%% XY postion plot for the ideal case
figure()
subplot(1, 3, 1)
plot(x_1, y_1,'k.')
set(gca, 'YDir','reverse') 
xlim([275 375])
ylim([250 400])
xlabel('X position')
ylabel('Y position')
title('Camera 1')

subplot(1, 3, 2)
plot(x_2,y_2,'k.')
set(gca, 'YDir','reverse') 
xlim([250 350])
ylim([150 350])
xlabel('X position')
ylabel('Y position')
title('Camera 2')

subplot(1, 3, 3)
plot(x_3,y_3,'k.')
set(gca, 'YDir','reverse') 
xlim([300 500])
ylim([260 310])
xlabel('X position')
ylabel('Y position')
title('Camera 3')

%% Test 2: Noisy Case
numFrames1 = size(vidFrames1_2, 4);
numFrames2 = size(vidFrames2_2, 4);
numFrames3 = size(vidFrames3_2, 4);

x_1 = [];
y_1 = [];

crop = zeros(480,640);
crop(200:430,250:450) = ones(231,201);

for j = 1:numFrames1
    X1 = rgb2gray(vidFrames1_2(:,:,:,j));
    X1 = uint8(crop) .* X1;
    [M, I] = max(double(X1)); 
    [~,indx] = max(M);
    indy = I(indx);
    x_1(j) = (indx);
    y_1(j) = (indy);
end

[~,i] = max(y_1(1:50)); 
x_1 = x_1 (i:end);
y_1 = y_1 (i:end);

x_2 = [];
y_2 = [];

crop = zeros(480,640);
crop(80:430,150:450) = ones(351,301);

for j = 1:numFrames2
    X2 = rgb2gray(vidFrames2_2(:,:,:,j));
    X2 = uint8(crop) .* X2;
    [M, I] = max(double(X2)); 
    [~,indx] = max(M);
    indy = I(indx);
    x_2(j) = (indx);
    y_2(j) = (indy);
end

[~,i] = max(y_2(1:50)); 
x_2 = x_2(i:end);
y_2 = y_2(i:end);

x_3 = [];
y_3 = [];

crop = zeros(480,640);
crop(200:430,250:450) = ones(231,201);

for j = 1:numFrames3
    X3 = rgb2gray(vidFrames3_2(:,:,:,j));
    X3 = uint8(crop) .* X3;
    [M, I] = max(double(X3));
    [~,indx] = max(M);
    indy = I(indx);
    x_3(j) = (indx);
    y_3(j) = (indy);
end

[~,i] = max(y_3(1:50)); 
x_3 = x_3(i:end);
y_3 = y_3(i:end);

minLength = min([length(y_1), length(y_2), length(y_3)]);
X = [x_1(1:minLength); y_1(1:minLength); x_2(1:minLength); 
    y_2(1:minLength); x_3(1:minLength); y_3(1:minLength)]; 
[~, n] = size(X);
avg = mean(X, 2);
X = X - repmat(avg,1,n);
[~, S, V] = svd(X, 'econ');

figure()
subplot(1, 3, 1)
plot(diag(S)./sum(diag(S)), 'k.', 'MarkerSize', 20)
xlabel('Principal Component')
ylabel('Energy')
title('Noisy Case Energy in Each Component')
xlim([1 7])

subplot(1, 3, 2)
plot(V(:,1)); 
hold on;
plot(V(:,2));
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Noisy Case - Displacement across Principle Component')
legend('PC1', 'PC2') 

subplot(1, 3, 3)
V = V * S;
plot(V(:,1)); 
hold on;
plot(V(:,2));
plot(V(:,3));
plot(V(:,4));
plot(V(:,5));
plot(V(:,6)); 
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Noisy Case - Variance across Principle Component')
legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')

%% XY postion plot for the noisy case
figure()
subplot(1, 3, 1)
plot(x_1,y_1,'k.')
set(gca, 'YDir','reverse') 
xlim([100 600])
ylim([100 500])
xlabel('X position')
ylabel('Y position')
title('Camera 1')

subplot(1, 3, 2)
plot(x_2,y_2,'k.')
set(gca, 'YDir','reverse') 
xlim([0 600])
ylim([0 600])
xlabel('X position')
ylabel('Y position')
title('Camera 2')

subplot(1, 3, 3)
plot(x_3,y_3,'k.')
set(gca, 'YDir','reverse') 
xlim([100 600])
ylim([150 350])
xlabel('X position')
ylabel('Y position')
title('Camera 3')


%% Test 3: Horizontal Displacement
numFrames1 = size(vidFrames1_3, 4);
numFrames2 = size(vidFrames2_3, 4);
numFrames3 = size(vidFrames3_3, 4);

x_1 = [];
y_1 = [];

crop = zeros(480,640);
crop(200:380,280:400) = ones(181,121);

for j = 1:numFrames1
    X1 = rgb2gray(vidFrames1_3(:,:,:,j));
    X1 = uint8(crop) .* X1;
    [M, I] = max(double(X1)); 
    [~,indx] = max(M);
    indy = I(indx);
    y_1(j) = (indy);
    x_1(j) = (indx);
end

[~,i] = max(y_1(1:50)); 
x_1 = x_1 (i:end);
y_1 = y_1 (i:end);

x_2 = [];
y_2 = [];

crop = zeros(480,640);
crop(170:400,200:400) = ones(231,201);

for j = 1:numFrames2
    X2 = rgb2gray(vidFrames2_3(:,:,:,j));
    X2 = uint8(crop) .* X2;
    [M, I] = max(double(X2)); 
    [~,indx] = max(M);
    indy = I(indx);
    x_2(j) = (indx);
    y_2(j) = (indy);
end

[~,i] = max(y_2(1:50)); 
x_2 = x_2(i:end);
y_2 = y_2(i:end);

x_3 = [];
y_3 = [];

crop = zeros(480,640);
crop(200:300,250:450) = ones(101,201);

for j = 1:numFrames3
    X3 = rgb2gray(vidFrames3_3(:,:,:,j));
    X3 = uint8(crop) .* X3;
    [M, I] = max(double(X3)); 
    [~,indx] = max(M);
    indy = I(indx);
    x_3(j) = (indx);
    y_3(j) = (indy);
end

[~,i] = max(y_3(1:50)); 
x_3 = x_3(i:end);
y_3 = y_3(i:end);

minLength = min([length(y_1), length(y_2), length(y_3)]);
X = [x_1(1:minLength); y_1(1:minLength); x_2(1:minLength); 
    y_2(1:minLength); x_3(1:minLength); y_3(1:minLength)]; 
[~, n] = size(X);
avg = mean(X, 2);
X = X - repmat(avg,1,n);
[~, S, V] = svd(X, 'econ');

figure()
subplot(1, 3, 1)
plot(diag(S)./sum(diag(S)), 'k.', 'MarkerSize', 20)
xlabel('Principal Component')
ylabel('Energy')
title('Horizontal Case Energy in Each Component')
xlim([1 7])

subplot(1, 3, 2)
plot(V(:,1)); 
hold on;
plot(V(:,2));
 
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Horizontal Case - Displacement across Principle Component')
legend('PC1', 'PC2') 

subplot(1, 3, 3)
V = V * S;
plot(V(:,1)); 
hold on;
plot(V(:,2));
plot(V(:,3));
plot(V(:,4));
plot(V(:,5));
plot(V(:,6)); 
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Horizontal Case - Variance across Principle Component')
legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')


%% XY postion plot for the horizontal case
figure()
subplot(1, 3, 1)
plot(x_1,y_1,'k.')
set(gca, 'YDir','reverse') 
xlim([100 600])
ylim([100 500])
xlabel('X position')
ylabel('Y position')
title('Camera 1')

subplot(1, 3, 2)
plot(x_2,y_2,'k.')
set(gca, 'YDir','reverse') 
xlim([0 600])
ylim([0 600])
xlabel('X position')
ylabel('Y position')
title('Camera 2')

subplot(1, 3, 3)
plot(x_3,y_3,'k.')
set(gca, 'YDir','reverse') 
xlim([100 600])
ylim([150 350])
xlabel('X position')
ylabel('Y position')
title('Camera 3')

%% Test 4: Horizontal Displacement and Rotation
numFrames1 = size(vidFrames1_4, 4);
numFrames2 = size(vidFrames2_4, 4);
numFrames3 = size(vidFrames3_4, 4);

y1 = [];
x1 = [];

crop = zeros(480,640);
crop(200:380,300:500) = ones(181,201);

for j = 1:numFrames1
    X1 = rgb2gray(vidFrames1_4(:,:,:,j));
    X1 = uint8(crop) .* (X1);
    [M, I] = max(double(X1(:))); 
    [row, col] = find(X1 >= 0.95*M); 
    x_1(j) = mean(col);
    y_1(j) = mean(row);
end

[~,i] = max(y_1(1:50)); 
x_1 = x_1(i:end);
y_1 = y_1(i:end);

y2 = [];
x2 = [];

crop = zeros(480,640);
crop(100:320,230:430) = ones(221,201);

for j = 1:numFrames2
    X2 = rgb2gray(vidFrames2_4(:,:,:,j));
    X2 = uint8(crop) .* (X2);
    [M, I] = max(double(X2(:))); 
    [row,col]=find(X2 >= 0.95*M); 
    x_2(j) = mean(col);
    y_2(j) = mean(row);
end

[~,i] = max(y_2(1:50)); 
x_2 = x_2(i:end);
y_2 = y_2(i:end);

x_3 = [];
y_3 = [];

crop = zeros(480,640);
crop(150:270,250:450) = ones(121,201);

for j = 1:numFrames3
    X3 = (rgb2gray(vidFrames3_4(:,:,:,j)));
    X3 = uint8(crop) .* (X3);
    [M, I] = max(double(X3(:))); 
    [row, col]=find(X3 >= 0.9*M);
    x_3(j) = mean(col);
    y_3(j) = mean(row);
end

[~,i] = max(y_3(1:50)); 
x_3 = x_3(i:end);
y_3 = y_3(i:end);

minLength = min([length(y_1), length(y_2), length(y_3)]);
X = [x_1(1:minLength); y_1(1:minLength); x_2(1:minLength); 
    y_2(1:minLength); x_3(1:minLength); y_3(1:minLength)]; 
[~, n] = size(X);
avg = mean(X, 2);
X = X - repmat(avg,1,n);
[~, S, V] = svd(X, 'econ');

figure()
subplot(1, 3, 1)
plot(diag(S)./sum(diag(S)), 'k.', 'MarkerSize', 20)
xlabel('Principal Component')
ylabel('Energy')
title('Horizontal Displacement and Rotation Case Energy in Each Component')
xlim([1 7])

subplot(1, 3, 2)
plot(V(:,1)); 
hold on;
plot(V(:,2));
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Horizontal and Rotation Case - Displacement across Principle Component')
legend('PC1', 'PC2')

subplot(1, 3, 3)
V = V * S;
plot(V(:,1)); 
hold on;
plot(V(:,2));
plot(V(:,3));
plot(V(:,4));
plot(V(:,5));
plot(V(:,6)); 
hold off;
xlabel('Time (frames)')
ylabel('Displacement')
title('Horizontal and Rotation Case - Variance across Principle Component')
legend('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')

%% XY postion plot for the Horizontal Displacement and Rotation case
figure()
subplot(1, 3, 1)
plot(x_1,y_1,'k.')
set(gca, 'YDir','reverse') 
xlim([100 600])
ylim([100 500])
xlabel('X position')
ylabel('Y position')
title('Camera 1')

subplot(1, 3, 2)
plot(x_2,y_2,'k.')
set(gca, 'YDir','reverse') 
xlim([0 600])
ylim([0 600])
xlabel('X position')
ylabel('Y position')
title('Camera 2')

subplot(1, 3, 3)
plot(x_3,y_3,'k.')
set(gca, 'YDir','reverse') 
xlim([100 600])
ylim([150 350])
xlabel('X position')
ylabel('Y position')
title('Camera 3')

