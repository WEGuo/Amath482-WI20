clear all; close all; clc
%%

[images, labels] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');

%%
image = zeros(784, 60000);

for i = 1:60000
    Im = images(:, :, i);
    image(:, i) = Im(:);
end
image = double(image);
% %%
avg = mean(image, 2);
image = image - repmat(avg,1,60000);
%%
[U,S,V] = svd(image,'econ');

%% Plot first ten principal components
figure()
for k = 1:10
   subplot(2,5,k)
   ut1 = reshape(U(:,k),28,28);
   ut2 = rescale(ut1);
   imshow(ut2)
end
saveas(gcf, 'PC.png')
clf
%% Singular Value Spectrum
plot(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 80])
title('SVD - Singular Value Spectrum')
saveas(gcf, 'Singular Value Spectrum.png')
clf

%%
energy = diag(S)./sum(diag(S));
plot(energy, 'k.', 'MarkerSize', 20)
xlabel('Principal Component')
ylabel('Energy')
title('Energy from Each Principle Component')
saveas(gcf, 'Energy.png')
clf

energyCap = [0, cumsum(energy)'];
plot(energyCap, 'k-')
hold on
plot(1:784, ones(784, 1)*0.9)

xlabel('Principal Component')
ylabel('Energy')
title('Cumulative Energy from Principle Components')
saveas(gcf, 'Cumulative Energy.png')
clf

rank = find(cumsum(energy)>0.9);
rank = rank(1);
% rank = 350
%% image reconstruction
rank = [1, 10, 50, 70, 80, 100];
for i = 1:6
    subplot(2,3,i)
    r = rank(i);
    newU = U(:, 1:r);
    newS = S(1:r, 1:r);
    newV = V(1, 1:r);
    reconstruction = newU * newS * newV';
    ut1 = reshape(reconstruction,28,28);
    imshow(rescale(ut1))
end
saveas(gcf, 'rank.png')
clf
%%
zero = find(labels == 0);
one = find(labels == 1);
two = find(labels == 2);
three = find(labels == 3);
four = find(labels == 4);
five = find(labels == 5);
six = find(labels == 6);
seven = find(labels == 7);
eight = find(labels == 8);
nine = find(labels == 9);

clf
v = V';
plot3(v(3, zero), v(2, zero), v(5, zero), '.'); 
hold on
plot3(v(3, one), v(2, one), v(5, one), '.'); 
plot3(v(3, two), v(2, two), v(5, two), '.'); 
plot3(v(3, three), v(2, three), v(5, three), '.'); 
plot3(v(3, four), v(2, four), v(5, four), '.'); 
plot3(v(3, five), v(2, five), v(5, five), '.'); 
plot3(v(3, six), v(2, six), v(5, six), '.'); 
plot3(v(3, seven), v(2, seven), v(5, seven), '.'); 
plot3(v(3, eight), v(2, eight), v(5, eight), '.'); 
plot3(v(3, nine), v(2, nine), v(5, nine), '.'); 
legend('Digit 0', 'Digit 1', 'Digit 2', 'Digit 3', 'Digit 4', 'Digit 5', 'Digit 6', 'Digit 7', 'Digit 8', 'Digit 9')
xlabel('mode 3')
ylabel('mode 2')
zlabel('mode 5')
title('3D projection of digits onto three selected V-modes')
saveas(gcf, '3D.png')
clf

%% test error matrix
[testimages, testlabels] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
%%
testimage = zeros(784, 10000);

for i = 1:10000
    Im = testimages(:, :, i);
    testimage(:, i) = Im(:);
end
testimage = double(testimage);

avg = mean(testimage, 2);
testimage = testimage - repmat(avg,1,10000);

TestMat = U'*testimage; % PCA projection


%%
ErrorMat = zeros(45, 4);
count = 1;

feature = 70;

digits = S*V';

for i = 0:8
    for j = (i+1) :9
        first = find(labels == i);
        second = find(labels == j);

        D1 = digits(1:feature,first);
        D2 = digits(1:feature,second);

        m1 = mean(D1,2);
        m2 = mean(D2,2);

        Sw = 0; 
        for k = 1:length(first)
            Sw = Sw + (D1(:,k) - m1)*(D1(:,k) - m1)';
        end
        for k = 1:length(second)
           Sw =  Sw + (D2(:,k) - m2)*(D2(:,k) - m2)';
        end

        Sb = (m1-m2)*(m1-m2)'; 

        [V2, D] = eig(Sb,Sw); 
        [lambda, ind] = max(abs(diag(D)));
        w = V2(:,ind);
        w = w/norm(w,2);

        v1 = w'*D1;
        v2 = w'*D2;

        if mean(v1) > mean(v2)
            w = -w;
            v1 = -v1;
            v2 = -v2;
        end

        sort1 = sort(v1);
        sort2 = sort(v2);

        t1 = length(sort1);
        t2 = 1;
        while sort1(t1) > sort2(t2)
            t1 = t1 - 1;
            t2 = t2 + 1;
        end
        threshold = (sort1(t1) + sort2(t2))/2;

        TrainError1 = sum(sort1 > threshold);
        TrainError2 = sum(sort2 < threshold);
        error = (TrainError1 + TrainError2)/(length(sort1)+length(sort2));

        % test data
        lookfor = [i, j];
        exist = ismember(testlabels, lookfor);
        testInd = find(exist);

        curTestMat = TestMat(1:feature, testInd);

        pval = w' * curTestMat;

        ResVec = (pval > threshold);

        digitcategory = ResVec * j;
        digitcategory(digitcategory == 0) = i;

        err = (digitcategory' ~= testlabels(testInd));
        errNum = sum(err);
        testError = errNum/length(testInd);

        ErrorMat(count, 1) = i;
        ErrorMat(count, 2) = j;
        ErrorMat(count, 3) = error;
        ErrorMat(count, 4) = testError;
        count = count+1;
    end
end

%%
plot(v1,zeros(1, length(v1)),'ob','Linewidth',2)
hold on
plot(v2,ones(1, length(v2)),'dr','Linewidth',2)
ylim([0 1.2])
plot([threshold, threshold], [0, 1.2], 'r')
legend('Digit 8', 'Digit 9', 'threshold', 'Location','southeast')
saveas(gcf, 'LDA.png')
clf

%%
[M, I] = max(ErrorMat(:, 3));
ErrorMat(I, [1, 2])
% 3 & 5
%%
[M, I] = min(ErrorMat(:, 3));
ErrorMat(I, [1, 2])
% 6 & 7
%%
[M, I] = max(ErrorMat(:, 4));
ErrorMat(I, [1, 2, 4])
% 5 & 8 error = 0.0477
%%
[M, I] = min(ErrorMat(:, 4));
ErrorMat(I, [1, 2, 4])
% 0 & 4 error = 0.0015
%%
feature = 70;

firstDigit = image(:, labels == 0);
secondDigit = image(:, labels == 6);
thirdDigit = image(:, labels == 1);

[U,S,V,threshold1,threshold2,w,sortd1,sortd2, sortd3] = three_digits_trainer(firstDigit,secondDigit,thirdDigit,feature);
%%
plot(sortd1, ones(length(firstDigit), 1), 'b.', 'MarkerSize', 0.3)
ylim([-1.2 1.2])
hold on
plot(sortd2, zeros(length(secondDigit), 1), 'r.', 'MarkerSize', 0.3)
plot(sortd3, ones(length(thirdDigit), 1) * -1, 'g.', 'MarkerSize', 0.3)
plot([threshold1, threshold1], [-1, 1], 'r')
plot([threshold2, threshold2], [-1, 1], 'r')
title('LDA for three digits 0, 6, and 1')
saveas(gcf, 'three digits.png')
clf


TrainErrorN1 = sum(sortd1 < threshold2);
TrainErrorN2 = sum(sortd2 < threshold1) + sum(sortd2 > threshold2);
TrainErrorN3 = sum(sortd3 > threshold1);

error = (TrainErrorN1 + TrainErrorN2 + TrainErrorN3)/(length(firstDigit)+length(secondDigit)+length(thirdDigit));
% 0.1363

%% SVM and decision tree
tree = fitctree(image', labels);
%% trainning error
pred = predict(tree, image');
sum(pred ~= labels) / 60000;
% 0.0321
%% test error
pred = predict(tree, testimage');
sum(pred ~= testlabels) / 10000;
% 0.3738

%% most difficult digit pair 5 & 8
i = 5;
j = 8;

lookfor = [i, j];
exist = ismember(labels, lookfor);
ind = find(exist);
y = labels(ind);
X = image(:, ind)';

SVMModel = fitcsvm(X,y);

exist = ismember(testlabels, lookfor);
ind = find(exist);
truelabel = testlabels(ind);
test = testimage(:, ind)';
test_labels = predict(SVMModel, test);

sum(test_labels ~= truelabel) / length(ind)
% 0.1249

%% easier digit pair 0 & 4
i = 0;
j = 4;

lookfor = [i, j];
exist = ismember(labels, lookfor);
ind = find(exist);
y = labels(ind);
X = image(:, ind)';

SVMModel = fitcsvm(X,y);

exist = ismember(testlabels, lookfor);
ind = find(exist);
truelabel = testlabels(ind);
test = testimage(:, ind)';
test_labels = predict(SVMModel, test);

sum(test_labels ~= truelabel) / length(ind)
% 0.0046

%% three digits trainer function

function [U,S,V,threshold1,threshold2,w,sortd1,sortd2, sortd3] = three_digits_trainer(digit1,digit2,digit3,feature)

    n1 = length(digit1);
    n2 = length(digit2);
    n3 = length(digit3);
    [U,S,V] = svd([digit1 digit2 digit3],'econ'); 
    digits = S*V';
    U = U(:,1:feature); 
    
    D1 = digits(1:feature,1:n1);
    D2 = digits(1:feature,n1+1:n1+n2);
    D3 = digits(1:feature,n1+n2+1:n1+n2+n3);
    m1 = mean(D1,2);
    m2 = mean(D2,2);
    m3 = mean(D3,2);
    
    Sw = 0;
    for k=1:n1
        Sw = Sw + (D1(:,k)-m1)*(D1(:,k)-m1)';
    end
    for k=1:n2
        Sw = Sw + (D2(:,k)-m2)*(D2(:,k)-m2)';
    end
    for k=1:n3
        Sw = Sw + (D3(:,k)-m3)*(D3(:,k)-m3)';
    end

    meanAll = mean(digits(1:feature, :), 2);
    mean3 = [m1, m2, m3];
    Sb = 0;
    for j = 1:3
        Sb = (mean3(:, j) - meanAll) * (mean3(:, j) - meanAll)';
    end

    [V2,D] = eig(Sb,Sw);
    [~,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    
    vd1 = w'*D1;
    vd2 = w'*D2;
    vd3 = w'*D3;

    sortd1 = sort(vd1);
    sortd2 = sort(vd2);
    sortd3 = sort(vd3);
    
    msortd1 = mean(sortd1);
    msortd2 = mean(sortd2);
    msortd3 = mean(sortd3);
    
    sortedmean = [msortd1, msortd2, msortd3];
    msort = sort(sortedmean);
    
    threshold1 = (msort(1)+msort(2))/2;
    threshold2 = (msort(2)+msort(3))/2;
 
end