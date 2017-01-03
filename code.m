% Property of Michael Castanieto
% All Rights Reserved 
% Do Not Copy -- GET YOUR OWN CODE!!!
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
dataRG = [skinR skinG; nonSkinR nonSkinG];
corrRG = corr(dataRG)
zRG = prdataset(dataRG,labs);
zRG = setlablist(zRG, char('Non Skin','Skin'));
zRG = setfeatlab(zRG, char('R','G'));
zRG = setprior(zRG,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
zRG = setname(zRG, 'Skin Segmentation');
figure
scatterd(zRG,'legend')

dataRB = [skinR skinB; nonSkinR nonSkinB];
corrRB = corr(dataRB)
zRB = prdataset(dataRB,labs);
zRB = setlablist(zRB, char('Non Skin','Skin'));
zRB = setfeatlab(zRB, char('R','B'));
zRB = setprior(zRB,[sampleSkinLength; sampleNonSkinLength]/sampleLength);
zRB = setname(zRB, 'Skin Segmentation');
figure
scatterd(zRB,'legend')

dataGB = [skinG skinB; nonSkinG nonSkinB];
corrGB = corr(dataGB)
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
