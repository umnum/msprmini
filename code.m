% Property of Michael Castanieto
% All Rights Reserved 
% Do Not Copy -- GET YOUR OWN CODE!!!
close all
clear

prpath = 'prtools';
addpath(prpath);

fid = fopen('Skin_NonSkin.txt');
data = textscan(fid,'%f%f%f%f');
fclose(fid);

R = data{1};
G = data{2};
B = data{3};
skinClass = data{4};

sampleLength = length(skinClass);
sampleSkinLength = length(skinClass(skinClass==1));
sampleNonSkinLength = length(skinClass(skinClass==2));

skinR = R(skinClass==1);
skinG = G(skinClass==1);
skinB = B(skinClass==1);
nonSkinR = R(skinClass==2);
nonSkinG = G(skinClass==2);
nonSkinB = B(skinClass==2);

data = [skinR skinG skinB; nonSkinR nonSkinG nonSkinB];

labs = genlab([sampleSkinLength, sampleNonSkinLength]);
z = prdataset(data,labs);
z = setlablist(z, char('Non Skin','Skin'));
z = setfeatlab(z, char('R','G','B'));
z = setprior(z,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
z = setname(z, 'Skin Segmentation');

% plot feature vectors against each other and
% calculate the correlation to see relations among the features
corrRGB = corr(data)
dataRG = [skinR skinG; nonSkinR nonSkinG];
zRG = prdataset(dataRG,labs);
zRG = setlablist(zRG, char('Non Skin','Skin'));
zRG = setfeatlab(zRG, char('R','G'));
zRG = setprior(zRG,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
zRG = setname(zRG, 'Skin Segmentation');
figure
scatterd(zRG,'legend')

dataRB = [skinR skinB; nonSkinR nonSkinB];
zRB = prdataset(dataRB,labs);
zRB = setlablist(zRB, char('Non Skin','Skin'));
zRB = setfeatlab(zRB, char('R','B'));
zRB = setprior(zRB,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
zRB = setname(zRB, 'Skin Segmentation');
figure
scatterd(zRB,'legend')

dataGB = [skinG skinB; nonSkinG nonSkinB];
zGB = prdataset(dataGB,labs);
zGB = setlablist(zGB, char('Non Skin','Skin'));
zGB = setfeatlab(zGB, char('G','B'));
zGB = setprior(zGB,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
zGB = setname(zGB, 'Skin Segmentation');
figure
scatterd(zGB,'legend')

[x, y] = gendat(z,[round(sampleNonSkinLength*0.80) round(sampleSkinLength*0.80)]);

w1 = qdc(y);
w2 = ldc(y);
w3 = nmsc(y);

% principle component analysis
S = cov(data); % calculate covariance matrix
[V D] = eig(S); % perform eigenvalue decomposition
eigval = diag(D); [eigval idx] = sort(eigval,'descend');
V = V(:,idx);
f = figure; plot(eigval);
set(gca, 'XTick', [1 2 3], 'XTickLabel', {'1.','2.','3.'})

f = figure;
linmod = {'magenta','red:','black-.'}
hold on;
for j = 1:3
	plot(V(:,j),linmod{j})
end
set(gca,'XTick',[1 2 3],'XTickLabel',{'R','G','B'})
legend('1. Eigvec','2. Eigvec','3. Eigvec')

[row,col] = size(data);
datac = data - ones(row,1)*mean(data); % centering data
scores = datac*V(:,1:2); % projection on eigenvectors with highest eigval
datar = scores*V(:,1:2)' + ones(row,1)*mean(data); % reconstructed data
z = prdataset(scores,skinClass); % construct pr data
f = figure; scatterd(z);

% TODO
% display reconstructed data
%labs = genlab([sampleSkinLength, sampleNonSkinLength]);
%z = prdataset(datar,labs);
z = prdataset(datar(:,1:2),skinClass); % construct pr data
%z = setlablist(z, char('Non Skin','Skin'));
%z = setfeatlab(z, char('R','G','B'));
%z = setprior(z,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
%z = setname(z, 'Skin Segmentation');
f = figure; scatterd(z);

relcumeig = sum(eigval(1:2))/sum(eigval);
cs = cumsum(eigval)/sum(eigval);
f = figure; plot(cs);
set(gca, 'XTick', [1 2 3], 'XTickLabel', {'1.','2.','3.'})


% test and training index
tst_idx = randperm(row,round(row*0.20));
tr_idx = setdiff(1:row,tst_idx);
% create test and training data
data = [R G B];
tst_data = prdataset(data(tst_idx,:),skinClass(tst_idx));
tr_data = prdataset(data(tr_idx,:),skinClass(tr_idx));
% create a reduced data set
idx_reduced = randperm(row,round(row*0.10)); % randomly choose 10% of the data
data_reduced = data(idx_reduced,:); 

% quadratic discriminant analysis
disp('quadratic discriminant analysis')
w = qdc(tr_data);
% display the confusion matrix
d = tst_data*w;
confmat(d)
cm = confmat(d);
% get true/false positive/negative values
tp = cm(1,1) % true positive
fn = cm(1,2) % false negative
fp = cm(2,1) % false positive
tn = cm(2,2) % true negative
% calculate precision, recall, f-measure,
% false alarm rate, and accuracy
p = tp/(tp+fp) % precision
r = tp/(tp+fn) % recall
f = (2*p*r)/(p+r) % f-measure
fa = fp/(tn+fp) % false alarm rate
acc = (tp+tn)/(tp+fp+fn+tn) % accuracy

pred_lab = tst_data*w*labeld;
errors = pred_lab~=skinClass(tst_idx);
total_errors = sum(errors)

% 10-fold cross validation on qdc
z = prdataset(data_reduced,skinClass(idx_reduced));
w = qdc([]);
e = prcrossval(z,w,10);
disp('10-fold qdc accurracy')
a = 1 - e

% linear discriminant analysis
disp('linear discriminant analysis')
w = ldc(tr_data);
% display the confusion matrix
d = tst_data*w;
confmat(d)
cm = confmat(d);
% get true/false positive/negative values
tp = cm(1,1) % true positive
fn = cm(1,2) % false negative
fp = cm(2,1) % false positive
tn = cm(2,2) % true negative
% calculate precision, recall, f-measure,
% false alarm rate, and accuracy
p = tp/(tp+fp) % precision
r = tp/(tp+fn) % recall
f = (2*p*r)/(p+r) % f-measure
fa = fp/(tn+fp) % false alarm rate
acc = (tp+tn)/(tp+fp+fn+tn) % accuracy

pred_lab = tst_data*w*labeld;
errors = pred_lab~=skinClass(tst_idx);
total_errors = sum(errors)

% minimum distance classifier
disp('minimum distance classifier')
w = nmsc(tr_data);
% display the confusion matrix
d = tst_data*w;
confmat(d)
cm = confmat(d);
% get true/false positive/negative values
tp = cm(1,1) % true positive
fn = cm(1,2) % false negative
fp = cm(2,1) % false positive
tn = cm(2,2) % true negative
% calculate precision, recall, f-measure,
% false alarm rate, and accuracy
p = tp/(tp+fp) % precision
r = tp/(tp+fn) % recall
f = (2*p*r)/(p+r) % f-measure
fa = fp/(tn+fp) % false alarm rate
acc = (tp+tn)/(tp+fp+fn+tn) % accuracy

pred_lab = tst_data*w*labeld;
errors = pred_lab~=skinClass(tst_idx);
total_errors = sum(errors)

% k-nearest neighbor classifier
disp('k-nearest neighbor classifier')
datak = data(randperm(row,round(row*0.10)),:); % randomly choose 10% of the data
[row,col] = size(datak);
tst_idx_k = randperm(row,round(row*0.20));
tr_idx_k = setdiff(1:row,tst_idx_k);
tst_datak = prdataset(datak(tst_idx_k,:),skinClass(tst_idx_k));
tr_datak = prdataset(datak(tr_idx_k,:),skinClass(tr_idx_k));
w = knnc(tr_datak);
% display the confusion matrix
d = tst_data*w;
confmat(d)
cm = confmat(d);
% get true/false positive/negative values
tp = cm(1,1) % true positive
fn = cm(1,2) % false negative
fp = cm(2,1) % false positive
tn = cm(2,2) % true negative
% calculate precision, recall, f-measure,
% false alarm rate, and accuracy
p = tp/(tp+fp) % precision
r = tp/(tp+fn) % recall
f = (2*p*r)/(p+r) % f-measure
fa = fp/(tn+fp) % false alarm rate
acc = (tp+tn)/(tp+fp+fn+tn) % accuracy

disp('knnc');
pred_lab = tst_data*w*labeld;
errors = pred_lab~=skinClass(tst_idx);
total_errors = sum(errors)

% quadratic discriminant analysis on 2 highest eigenvectors
disp('quadratic discriminant analysis on 2 highest eigenvectors')
tst_scores = prdataset(scores(tst_idx,:),skinClass(tst_idx));
tr_scores = prdataset(scores(tr_idx,:),skinClass(tr_idx));

w = qdc(tr_scores);
% display the confusion matrix
d = tst_scores*w;
confmat(d)
cm = confmat(d);
% get true/false positive/negative values
tp = cm(1,1) % true positive
fn = cm(1,2) % false negative
fp = cm(2,1) % false positive
tn = cm(2,2) % true negative
% calculate precision, recall, f-measure,
% false alarm rate, and accuracy
p = tp/(tp+fp) % precision
r = tp/(tp+fn) % recall
f = (2*p*r)/(p+r) % f-measure
fa = fp/(tn+fp) % false alarm rate
acc = (tp+tn)/(tp+fp+fn+tn) % accuracy

pred_lab = tst_scores*w*labeld;
errors = pred_lab~=skinClass(tst_idx);
total_errors = sum(errors)

% quadratic discriminant analysis on the reconstructed data
disp('quadratic discriminant analysis on the reconstructed data')
tst_datar = prdataset(datar(tst_idx,1:2),skinClass(tst_idx));
tr_datar = prdataset(datar(tr_idx,1:2),skinClass(tr_idx));


% 10-fold cross validation on qdc
z = prdataset(datar(idx_reduced,1:2),skinClass(idx_reduced)); % construct pr data
w = qdc([]);
e = prcrossval(z,w,10);
disp('10-fold qdc accurracy of reconstructed data')
a = 1 - e

% plot qdc on reconstructed data
w = qdc(tr_datar);
h = figure;
scatterd(z);
hold on
plotm(w)
title('qdc classification on the reconstructed data')
text(0,200,['accuracy = '  num2str(acc*100) '%'], 'FontSize', 16);

pred_lab = tst_datar*w*labeld;
errors = pred_lab~=skinClass(tst_idx);
total_errors = sum(errors)
corr_rec = corr(datar)
